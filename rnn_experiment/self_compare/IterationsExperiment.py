# sys.path.insert(0, BASE_FOLDER)
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from timeit import default_timer as timer
from typing import Tuple, List, Union

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from rnn_experiment.self_compare.create_sbatch_iterations_exp import BASE_FOLDER, OUT_FOLDER
from rnn_experiment.self_compare.generate_random_points import POINTS_PATH

MODELS_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/models/")

IN_SHAPE = (40,)
NUM_SAMPLE_POINTS = 25
NUM_RUNNER_UP = 1

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
    except AssertionError as e:
        res = False

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
    pickle_path = os.path.join(OUT_FOLDER,
                               'gurobi' + str(datetime.now()).replace('.', '').replace(' ', '') + "{}.pkl".format(
                                   net_name))
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
    if save_results:
        if len(net_options) == 1:
            parse_results_file(pickle_path, t_range=t_range)
        else:
            parse_results_file([(net_name, pickle_path)], t_range=t_range)
    return results


def parse_results_file(name_path_map: Union[List[Tuple[str, str]], str], t_range=range(2, 20), print_latex=False):
    # TODO: Can I detect t_range from the pickle? using different values cases problems
    if isinstance(name_path_map, str) and name_path_map.endswith('pkl'):
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
    if isinstance(name_path_map, str) and os.path.isdir(name_path_map):
        tuples = []
        for f in sorted(os.listdir(name_path_map)):
            if not f.endswith('.pkl'):
                continue
            p = os.path.join(name_path_map, f)
            if not os.path.isfile(p):
                continue
            m, _ = extract_model_name_ephocs(f)
            tuples.append((m, p))
        name_path_map = tuples

    x = PrettyTable()
    # names = set([p[0] for p in name_path_map])
    if len(set(name_path_map)) != len(name_path_map):
        print('PROBLEM')
    x.field_names = ['Tmax'] + [p[0] for p in name_path_map]  # if p[0] not in names]

    rows = [[t] for t in t_range]

    total_time = 0
    total_points = 0
    total_timeout = 0
    total_gurobi_time = 0
    total_invariant_time = 0
    total_property_time = 0
    sum_success = 0
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
            sum_success += total_success
            avg_run_time = res['avg_total_time_no_timeout']

            total_time += avg_run_time * res['total']
            assert np.abs(total_time == res['avg_total_time']) < 10**-3
            total_points += res['total']
            total_timeout += res['len_timeout']
            total_gurobi_time +=  res['avg_step_time'] * res['total']
            total_invariant_time += res['avg_invariant_time'] * res['total']
            total_property_time += res['avg_property_time'] * res['total']
            # assert timeout == 0
            gurobi_time = res['avg_step_time_no_timeout']
            ffnn_time = res['avg_invariant_time'] + res['avg_property_time'] - gurobi_time
            assert ffnn_time < avg_run_time, "{}, {}, {}".format(ffnn_time, gurobi_time, avg_run_time)


            avg_gurobi_invariant = res['avg_step_time']
            # avg_marabou_invariant = res['avg_invariant_time_no_timeout'] - avg_gurobi_invariant
            # print("Format is: time (#success/#total) (#timeout)")
            # rows[t - t_range[0]].append("%.2f (%.2f,%.2f) %d/%d (%d)" % (avg_run_time, ffnn_time, gurobi_time,
            #                                                              total_success, res['total'],
            #                                                              res['len_timeout']))
            rows[t - t_range[0]].append("%.2f(%d/%d)" % (avg_run_time, total_success, res['total']))

    for row in rows:
        x.add_row(row)
    # for k, v in rows.items():
    #     x.add_row([k] + ["{}({})".format(v[net_name][0], v[net_name][1]) for net_name in x.field_names])
    print("Format is: time (#success/#total)")
    print(x)
    if print_latex:
        for t, row in enumerate(rows):
            print("\t{} &&&".format(t + 2))
            print(" &&& ".join(row[1:]).replace("  ", " "), "&")
            print("\t\\\\")

    avg_time = total_time / total_points
    avg_gurobi = total_gurobi_time / total_points
    avg_marabou_invariant = (total_invariant_time - total_gurobi_time) / total_points
    avg_property = total_property_time / total_points
    print("#" * 100)

    print("Average run time {} seconds, over {} points, timeout: {}, success: {} ({:.2f})"
          .format(avg_time, total_points, total_timeout, sum_success, sum_success / total_points))
    print("avg Time in Gurobi: {}({:.2f}%), avg Time proving Invariant in Marabou {} ({:.2f}%),"
          "avg Time proving property: {} ({:.2f}%)"
          .format(avg_gurobi, (avg_gurobi) / avg_time,
                avg_marabou_invariant, (avg_marabou_invariant) / avg_time,
                avg_property, (avg_property) / avg_time))
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
        if 'number_of_updates' in e['stats'] and e['stats']['number_of_updates'] == e['stats']['property_queries'] \
                and not e['result']:
            timeout_exp.append(e)
        elif 'FFNN_Timeout' in e['stats']:
            timeout_exp.append(e)
        else:
            no_timeout_exp.append(e)

    safe_mean = lambda x: np.mean(x) if len(x) > 0 else 0

    d = {
        'total': len(exp),
        'total_success': len(success_exp),
        'success_rate': len(success_exp) / len(exp),
        'len_timeout': len(timeout_exp),
        'avg_total_time': safe_mean([e['time'] for e in exp]),
        'avg_total_time_no_timeout': safe_mean([e['time'] for e in no_timeout_exp]),
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
        'num_step_avg_success': safe_mean([e['stats']['step_queries'] for e in success_exp]),
        # 'initialize_query_time': safe_mean([e['stats']['query_initialize'] for e in success_exp])
    }

    gurobi_time = d['avg_step_time_no_timeout']
    ffnn_time = d['avg_invariant_time_no_timeout'] + d['avg_property_time_no_timeout'] - gurobi_time
    avg_run_time = d['avg_total_time_no_timeout']

    assert ffnn_time < avg_run_time, "{}, {}, {}".format(ffnn_time, gurobi_time, avg_run_time)
    assert d['num_invariant_avg'] <= 1, "Found point that needed more then one invariant to prove!!! THIS IS GOOD"
    return d


def parse_inputs(t_range, net_options, points, other_idx_method):
    save_results = True

    if sys.argv[1] == 'analyze':
        parse_results_file(sys.argv[2], t_range, print_latex=1)
    if sys.argv[1] == 'compare':
        compare_ephocs(sys.argv[2], t_range)
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


def extract_model_name_ephocs(name):
    m = 'rnn'
    ephocs = -1
    fc_count = 0
    for w in name.split('_')[2:]:
        if 'rnn' in w:
            m += w.replace('rnn', '')
        elif 'fc' in w:
            fc_count += 1
        else:
            m += '_{}fc32'.format(fc_count)
            ephocs = w.replace('.out', '')
    return m, ephocs


def compare_ephocs(pkl_dir: str, t_range):
    models = defaultdict(list)
    models_to_ephocs = defaultdict(set)  # in the ephocs pickle there is timestep, make sure no duplicates
    for f in os.listdir(pkl_dir):
        if not f.endswith('pkl'):
            continue
        m, e = extract_model_name_ephocs(f)
        if e in models_to_ephocs[m]:
            print('file {} is duplicate of the ephocs'.format(f))
            continue
        models_to_ephocs[m].update(set([e]))
        models[m].append((e, os.path.join(pkl_dir, f)))
    for k, v in models.items():
        print("Results for model: {}".format(k))

        v = sorted(v, key=lambda x: x[0])
        parse_results_file(v, t_range)
        print("\n\n")


if __name__ == "__main__":
    # TODO: Write test, to demonstrate entry point to the experiment (for every parse option)

    t_range = range(2, 21)
    # name = os.path.join("ATVA_EXP", "out_e6629c3", "gurobi2020-04-2922\:07\:34914800model_20classes_rnn8_fc32_fc32_fc32_fc32_fc32_epochs50.pkl")
    parse_results_file('ATVA_EXP/out_8eb20ee/filter/', t_range, print_latex=1)
    exit(0)
    FMCAD_networks = ['model_20classes_rnn4_rnn4_rnn4_fc32_fc32_fc32_0200.pkl',
                      'model_20classes_rnn4_rnn4_rnn4_rnn4_fc32_fc32_fc32_0200.pkl',
                      'model_20classes_rnn8_rnn8_fc32_fc32_0200.pkl',
                      'model_20classes_rnn12_rnn12_fc32_fc32_fc32_fc32_0200.pkl',
                      'model_20classes_rnn16_fc32_fc32_fc32_fc32_0100.pkl',
                      'model_20classes_rnn8_rnn4_rnn4_fc32_fc32_fc32_fc32_0150.pkl']
    # parse_results_file('FMCAD_EXP/out_filter/second_filter', t_range)
    # exit(-1)

    # other_idx_method = [lambda x: np.argmin(x)]
    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 2 + NUM_RUNNER_UP)]

    points = pickle.load(open(POINTS_PATH, "rb"))[:NUM_SAMPLE_POINTS]

    gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    if len(sys.argv) > 1:
        net_options = None
        parse_inputs(t_range, FMCAD_networks, points, other_idx_method)

    # run_all_experiments(['models/AUTOMATIC_MODELS/model_20classes_rnn4_rnn4_fc32_fc320002.ckpt'], points, t_range,
    #                     other_idx_method, gurobi_ptr, steps_num=10)
    # exit(0)

    point = np.array([-1.90037058, 2.80762335, 5.6615233, -3.3241606, -0.83999373, -4.67291775,
                      -2.37340524, -3.94152213, 1.78206783, -0.37256191, 1.07329743, 0.02954765,
                      0.50143303, -3.98823161, -1.05437203, -1.98067338, 6.12760627, -2.53038902,
                      -0.90698131, 4.40535622, -3.30067319, -1.60564116, 0.22373327, 0.43266462,
                      5.45421917, 4.11029858, -4.65444165, 0.50871269, 1.40619639, -0.7546163,
                      3.68131841, 1.18965503, 0.81459484, 2.36269942, -2.4609835, -1.14228611,
                      -0.28604645, -6.39739288, -3.54854402, -3.21648808])
    net = "model_20classes_rnn4_rnn4_rnn4_rnn4_fc32_fc32_fc32_fc32_0200.ckpt"
    # net = "model_20classes_rnn16_fc32_fc32_fc32_fc32_0100.ckpt"
    t_range = range(2, 21)
    run_all_experiments([net], points, t_range, other_idx_method, gurobi_ptr, steps_num=10)
    exit(0)
    gurobi_multi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                               use_counter_example=True)
    run_all_experiments([net], points[:5], t_range, other_idx_method, gurobi_multi_ptr, save_results=0, steps_num=2)
    # run_all_experiments([net_options[3]], points[:2], t_range, other_idx_method, gurobi_multi, save_results=0, steps_num=2)
