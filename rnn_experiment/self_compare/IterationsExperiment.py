from functools import partial
from typing import Tuple, List, Union

from rnn_algorithms.GurobiBased import AlphasGurobiBasedMultiLayer, AlphasGurobiBased
from rnn_algorithms.Update_Strategy import Relative_Step
from rnn_experiment.self_compare.generate_random_points import POINTS_PATH

# BASE_FOLDER = "/home/yuval/projects/Marabou/"
import sys

sys.path.insert(0, BASE_FOLDER)
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from maraboupy.keras_to_marabou_rnn import adversarial_query, get_out_idx
from rnn_experiment.self_compare.create_sbatch_iterations_exp import BASE_FOLDER, OUT_FOLDER
# BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
MODELS_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/models/")
# OUT_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/out/")

IN_SHAPE = (40,)


def run_experiment(in_tensor, radius, idx_max, other_idx, h5_file, gurobi_ptr, n_iterations, steps):
    queries_stats = {}
    start = timer()
    try:
        res, queries_stats, alpha_history = adversarial_query(in_tensor, radius, idx_max, other_idx, h5_file,
                                                              gurobi_ptr, n_iterations, steps)
    except ValueError as e:
        # row_result = {'point': in_tensor, 'error': e, 'error_traceback': traceback.format_exc(), 'result' : False}
        res = False
    except TimeoutError as e:
        res = False
        queries_stats['FFNN_Timeout'] = True

    end = timer()

    if queries_stats is not None:
        queries_stats['total_time'] = end - start
    return {'time': end - start, 'result': res, 'stats': queries_stats}


def run_all_experiments(net_options, points, t_range, other_idx_method, gurobi_ptr, radius=0.01, steps_num=1500,
                        save_results=True, continue_pickle=None):
    # assert len(points) > 20
    results = defaultdict(list)
    if len(net_options) == 1:
        net_name = ''.join(net_options[0].split('.')[:-1]).split('/')[-1]
    else:
        net_name = ''
    pickle_path = os.path.join(OUT_FOLDER, 'gurobi' + str(datetime.now()).replace('.', '').replace(' ', '') + "{}.pkl".format(net_name))
    if continue_pickle is not None and os.path.exists(continue_pickle):
        partial_results = pickle.load(open(continue_pickle, "rb"))
        pickle_path = continue_pickle
        results = partial_results
    else:
        print("starting fresh experiment", "\n", "#" * 100)
        partial_results = {}

    print("#" * 100, "\nwriting results to: {}".format(pickle_path), "\n", "#" * 100)
    counter = 0
    pbar = tqdm(total=len(other_idx_method) * len(points) * len(net_options) * len(t_range))
    for method in other_idx_method:
        for idx, point in enumerate(points):
            for path in net_options:
                if not os.path.exists(path):
                    path = os.path.join(MODELS_FOLDER, path)
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                for t in t_range:
                    if counter < 0:
                        counter += 1
                        pbar.update(1)
                        have_point = True
                    else:
                        have_point = False
                        net_name = ''.join(path.split('.')[:-1]).split('/')[-1]
                        name = "{}_{}_{}".format(net_name, radius, t)

                        if name in partial_results:
                            for res in partial_results[name]:
                                if not have_point and res['t'] == t and \
                                        (('in_tensor' in res and np.all(res['in_tensor'] == point)) or
                                         ('in_tesnor' in res and np.all(res['in_tesnor'] == point))):
                                    # already have this result
                                    pbar.update(1)
                                    results[name].append(res)
                                    have_point = True
                    if not have_point:
                        idx_max, other_idx = get_out_idx(point, t, path, method)
                        net_name = ''.join(path.split('.')[:-1]).split('/')[-1]

                        result = run_experiment(point, radius, idx_max, other_idx, path, gurobi_ptr, t, steps=steps_num)
                        result.update({'h5_file': net_name, 't': t, 'other_idx': other_idx, 'in_tensor': point,
                                       'steps_num': steps_num})
                        results[name].append(result)
                        pbar.update(1)
                        if save_results:
                            pickle.dump(results, open(pickle_path, "wb"))
    parse_results_file(pickle_path)
    return results


def parse_results_file(name_path_map: Union[List[Tuple[str, str]], str], t_range=range(2, 20), print_latex=False):
    if isinstance(name_path_map, str):
        os.makedirs("temp/", exist_ok=True)
        results = pickle.load(open(name_path_map, "rb"))
        name_path_map = []
        new_files = defaultdict(dict)
        for k, v in results.items():
            model_name = '_'.join(k.split('_')[:-1])
            new_files[model_name].update({k: v})
        for k, v in new_files.items():
            pickle.dump(v, open("temp/{}".format(k), "wb"))
            name_path_map.append((k.replace("model_20classes_", ""), "temp/{}".format(k)))

    x = PrettyTable()
    # names = set([p[0] for p in name_path_map])
    x.field_names = ['Tmax'] + [p[0] for p in name_path_map]  # if p[0] not in names]

    rows = [[t] for t in t_range]

    total_time = 0
    total_points = 0
    total_timeout = 0
    for p in name_path_map:
        d = pickle.load(open(p[1], "rb"))
        for key, value in d.items():
            net_name = "_".join(key.split("_")[:-2])
            t = int(key.split("_")[-1])
            res = parse_dictionary(value)
            success_rate = int(res['success_rate'] * 100)
            if res['num_invariant_avg_success'] > 1:
                # TODO: Check this out, we improve an invariant using our heuristic
                assert False

            total_success = res['total_success']
            avg_run_time = res['avg_total_time_no_timeout']

            total_time += avg_run_time * res['total']
            total_points += res['total']
            total_timeout += res['len_timeout']
            # assert timeout == 0
            gurobi_time = res['avg_step_time_no_timeout']
            ffnn_time = res['avg_invariant_time_no_timeout'] + res['avg_property_time_no_timeout']
            assert ffnn_time + gurobi_time < avg_run_time, "{}, {}, {}".format(ffnn_time, gurobi_time, avg_run_time)
            rows[t - t_range[0]].append("%.2f (%.2f,%.2f) success_rate: %d/%d ffnn_timeout: %d)" %
                                        (avg_run_time, ffnn_time, gurobi_time, total_success, res['total'],
                                         res['len_timeout']))
            # print("net: {}, time: {} \n".format(net_name, time), parse_dictionary(value), "\n" + "#" * 100, "\n")

    for row in rows:
        x.add_row(row)
    # for k, v in rows.items():
    #     x.add_row([k] + ["{}({})".format(v[net_name][0], v[net_name][1]) for net_name in x.field_names])
    print("Format is: time (#success/#total) (#timeout)")
    print(x)
    if print_latex:
        for t, row in enumerate(rows):
            print("\t{} &&&".format(t + 2))
            print(" &&& ".join(row[1:]).replace("  ", " "), "&")
            print("\t\\\\")
    print("#" * 100)
    print("Average run time {} seconds, over {} points, timeout: {}".format(total_time / total_points, total_points,
                                                                            total_timeout))
    print("#" * 100)


def parse_dictionary(exp):
    # This is an experiment entry strcture:
    #    { 'time', 'result', 'h5_file', 't', other_idx', 'in_tesnor',  'steps_num'
    #   'stats':
    #       {'property_times': {'avg', 'median', 'raw'},
    #       'invariant_times': {'avg', 'median', 'raw'},
    #       'step_times': {'avg', 'median', 'raw'},
    #       'step_querites', 'property_queries', 'invariant_queries', 'number_of_updates', 'total_time'},
    #   }

    success_exp = [e for e in exp if e['result']]
    timeout_exp = []
    no_timeout_exp = []
    for e in exp:
        if 'number_of_updates' in e['stats'] and e['stats']['number_of_updates'] == e['stats']['property_queries']\
                and not e['result']:
            timeout_exp.append(e)
        elif 'FFNN_Timeout' in e['stats']:
            timeout_exp.append(e)
        else:
            no_timeout_exp.append(e)

    safe_mean = lambda x: np.mean(x) if len(x) > 0 else 0

    if len(exp) == len(timeout_exp):
        avg_total_time_no_timeout = None
    else:
        avg_total_time_no_timeout = (sum([e['time'] for e in exp]) - sum([e['time'] for e in timeout_exp])) / (
                len(exp) - len(timeout_exp))
    assert np.abs(avg_total_time_no_timeout - safe_mean([e['time'] for e in no_timeout_exp])) < 2*10**-2, \
        "{}, {}".format(avg_total_time_no_timeout, safe_mean([e['time'] for e in no_timeout_exp]))

    d = {
        'total': len(exp),
        'total_success': len(success_exp),
        'success_rate': len(success_exp) / len(exp),
        'len_timeout': len(timeout_exp),
        'avg_total_time': safe_mean([e['time'] for e in exp]),
        'avg_total_time_no_timeout': avg_total_time_no_timeout,
        'avg_invariant_time_no_timeout': safe_mean(
            [e['stats']['invariant_times']['avg'] for e in no_timeout_exp if 'invariant_times' in e['stats']]),
        'avg_property_time_no_timeout': safe_mean(
            [e['stats']['property_times']['avg'] for e in no_timeout_exp if 'property_times' in e['stats']]),
        'avg_step_time_no_timeout': safe_mean(
            [e['stats']['step_times']['avg'] for e in no_timeout_exp if 'step_times' in e['stats']]),
        'avg_invariant_time': safe_mean(
            [e['stats']['invariant_times']['avg'] for e in exp if 'invariant_times' in e['stats']]),
        'avg_property_time': safe_mean(
            [e['stats']['property_times']['avg'] for e in exp if 'property_times' in e['stats']]),
        'avg_step_time': safe_mean([e['stats']['step_times']['avg'] for e in exp if 'step_times' in e['stats']]),
        'num_invariant_avg': safe_mean(
            [e['stats']['invariant_queries'] for e in exp if 'invariant_queries' in e['stats']]),
        'num_property_avg': safe_mean(
            [e['stats']['property_queries'] for e in exp if 'property_queries' in e['stats']]),
        'num_step_avg': safe_mean([e['stats']['step_queries'] for e in exp if 'step_queries' in e['stats']]),
        'avg_total_time_success': safe_mean([e['time'] for e in success_exp]),
        'avg_invariant_time_success': safe_mean([e['stats']['invariant_times']['avg'] for e in success_exp]),
        'avg_property_time_success': safe_mean([e['stats']['property_times']['avg'] for e in success_exp]),
        'avg_step_time_success': safe_mean([e['stats']['step_times']['avg'] for e in success_exp]),
        'num_invariant_avg_success': safe_mean([e['stats']['invariant_queries'] for e in success_exp]),
        'num_property_avg_success': safe_mean([e['stats']['property_queries'] for e in success_exp]),
        'num_step_avg_success': safe_mean([e['stats']['step_queries'] for e in success_exp])
    }

    gurobi_time = d['avg_step_time_no_timeout']
    ffnn_time = d['avg_invariant_time_no_timeout'] + d['avg_property_time_no_timeout']
    avg_run_time = d['avg_total_time_no_timeout']

    if ffnn_time + gurobi_time > avg_run_time:
        print("{}, {}, {}".format(ffnn_time, gurobi_time, avg_run_time))
    return d


def parse_inputs(t_range, net_options, points, other_idx_method):
    save_results = True

    if sys.argv[1] == 'analyze':
        parse_results_file(sys.argv[2])
    if sys.argv[1] == 'exp':
        if len(sys.argv) > 2:
            if str.isnumeric(sys.argv[2]):
                net = [net_options[int(sys.argv[2])]]
            elif sys.argv[2] == 'all':
                net = net_options
            else:
                net = [sys.argv[2]]
            if len(sys.argv) > 3:
                continue_pickle = sys.argv[3]
            else:
                continue_pickle = ''

            run_all_experiments(net, points, t_range, other_idx_method, gurobi_ptr, steps_num=10,
                                continue_pickle=continue_pickle, save_results=save_results)
        else:
            for net in net_options:
                run_all_experiments([net], points, t_range, other_idx_method, gurobi_ptr, steps_num=30,
                                    continue_pickle='')
    exit(0)


if __name__ == "__main__":
    # parse_results_file([
    #     ('rnn2_fc0', "pickles/final_gurobi_exps/rnn2_fc0.pkl"),
    #     ('rnn2_fc1', "pickles/final_gurobi_exps/rnn2_fc1.pkl"),
    #     ('rnn4_fc1', "pickles/final_gurobi_exps/rnn4_fc1.pkl"),
    #     ('rnn4_fc2', "pickles/final_gurobi_exps/rnn4_fc2.pkl"),
    #     ('rnn4_rnn2', "pickles/final_gurobi_exps/rnn4_rnn2.pkl"),
    #     ('rnn4_rnn4_fc4', "pickles/final_gurobi_exps/rnn4_rnn4_fc4.pkl"),
    #     ('rnn4_rnn4_rnn4_fc1', "pickles/final_gurobi_exps/rnn4_rnn4_rnn4_fc1.pkl"),
    #     ('rnn8_fc1', "pickles/final_gurobi_exps/rnn8_fc1.pkl"),
    #
    # ], print_latex=False)
    # parse_results_file('gurobi_19night_run2.pkl')
    # exit(0)
    net_options = ["./model_20classes_rnn2_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn8_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn8_rnn6_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn16_fc16_epochs200.h5",
                   "./model_20classes_rnn12_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn4_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn16_fc16_epochs50.h5",
                   "./model_20classes_rnn10_fc32_fc32_fc32_fc32_fc32_epochs50.h5",
                   "./model_20classes_rnn6_rnn6_fc32_fc32_fc32_fc32_fc32_epochs50.h5"]

    # other_idx_method = [lambda x: np.argmin(x)]

    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 7)]

    t_range = range(2, 21)
    points = pickle.load(open(POINTS_PATH, "rb"))[:5]

    gurobi_ptr = partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20000,
                        use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    if len(sys.argv) > 1:
        parse_inputs(t_range, net_options, points, other_idx_method)

    # run_all_experiments(['models/AUTOMATIC_MODELS/model_20classes_rnn4_rnn4_fc32_fc320002.ckpt'], points, t_range,
    #                     other_idx_method, gurobi_ptr, steps_num=10)
    #run_all_experiments([net_options[0]], points, t_range, other_idx_method, gurobi_ptr, steps_num=10)
    exit(0)
    # other_idx_method = [lambda x: np.argmin(x)]
    point = np.array([-1.90037058, 2.80762335, 5.6615233, -3.3241606, -0.83999373, -4.67291775,
                      -2.37340524, -3.94152213, 1.78206783, -0.37256191, 1.07329743, 0.02954765,
                      0.50143303, -3.98823161, -1.05437203, -1.98067338, 6.12760627, -2.53038902,
                      -0.90698131, 4.40535622, -3.30067319, -1.60564116, 0.22373327, 0.43266462,
                      5.45421917, 4.11029858, -4.65444165, 0.50871269, 1.40619639, -0.7546163,
                      3.68131841, 1.18965503, 0.81459484, 2.36269942, -2.4609835, -1.14228611,
                      -0.28604645, -6.39739288, -3.54854402, -3.21648808])
    # net = "model_20classes_rnn4_rnn2_fc16_epochs3.h5"
    net = "model_20classes_rnn4_rnn4_rnn4_fc32_epochs50.h5"
    t_range = range(8, 10)
    #gurobi_multi_ptr = partial(AlphasGurobiBasedMultiLayer, update_strategy_ptr=Relative_Step, random_threshold=20000,
    #                           use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    #gurobi_ptr = partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20000,
    #                     use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    #run_all_experiments([net], points[:5], t_range, other_idx_method, gurobi_multi_ptr, save_results=0, steps_num=2)
    # run_all_experiments([net_options[3]], points[:2], t_range, other_idx_method, gurobi_multi, save_results=0, steps_num=2)
