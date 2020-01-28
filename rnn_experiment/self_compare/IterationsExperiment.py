BASE_FOLDER = "/home/yuval/projects/Marabou/"
import sys
sys.path.insert(0, BASE_FOLDER)
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from timeit import default_timer as timer

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from maraboupy.keras_to_marabou_rnn import adversarial_query, get_out_idx
from rnn_algorithms.GurobiBased import AlphasGurobiBased, AlphasGurobiBasedMultiLayer
from rnn_algorithms.Update_Strategy import Relative_Step

# BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
MODELS_FOLDER = os.path.join(BASE_FOLDER, "models/")

POINTS_PATH = "points.pkl"
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
    pickle_path = 'gurobi' + str(datetime.now()).replace('.', '').replace(' ', '') + "{}.pkl".format(net_name)
    if continue_pickle is not None and os.path.exists(continue_pickle):
        partial_results = pickle.load(open(continue_pickle, "rb"))
        pickle_path = continue_pickle
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
                        result.update(
                            {'h5_file': net_name, 't': t, 'other_idx': other_idx, 'in_tensor': point,
                             'steps_num': steps_num})
                        results[name].append(result)
                        pbar.update(1)
                        if save_results:
                            pickle.dump(results, open(pickle_path, "wb"))
    return results


def generate_points(models_folder, number=500, max_t=20):
    mean = 0
    var = 3
    points = []
    models = ['model_classes20_1rnn8_1_32_4.h5', 'model_20classes_rnn2_fc32_epochs200.h5',
              'model_20classes_rnn4_fc32_epochs40.h5', ]
    # ,'model_classes20_1rnn2_0_64_4.h5', 'model_20classes_rnn4_fc32_epochs100.h5'] # os.listdir(models_folder)
    pbar = tqdm(total=number)
    while len(points) <= number:
        fail = 0
        candidate = np.random.normal(mean, var, IN_SHAPE)
        for file in models:
            if os.path.isfile(os.path.join(models_folder, file)):
                try:
                    y_idx_max, other_idx = get_out_idx(candidate, max_t, os.path.join(models_folder, file))
                    if y_idx_max is None or other_idx is None and y_idx_max == other_idx:
                        fail = True
                        continue
                except ValueError:
                    # model with different input shape, it does not matter
                    pass
        if not fail:
            points.append(candidate)
            pbar.update(1)
    pbar.close()

    pickle.dump(points, open(POINTS_PATH, "wb"))


def parse_results_file(pickle_paths, t_range=range(2, 20)):
    x = PrettyTable()

    x.field_names = ['Tmax'] + [p[0] for p in pickle_paths]
    rows = [[t] for t in t_range]

    total_time = 0
    total_points = 0
    for p in pickle_paths:
        d = pickle.load(open(p[1], "rb"))
        for key, value in d.items():
            net_name = "_".join(key.split("_")[:-2])
            t = int(key.split("_")[-1])
            res = parse_dictionary(value)
            success_rate = int(res['success_rate'] * 100)
            total_success = res['total_success']
            run_time = round(res['avg_total_time_no_timeout'], 2)
            total_time += run_time * res['total']
            total_points += res['total']
            timeout = res['len_timeout']
            assert timeout == 0
            rows[t - t_range[0]].append("%.2f (%d/%d)" % (run_time, total_success, res['total']))
            # print("net: {}, time: {} \n".format(net_name, time), parse_dictionary(value), "\n" + "#" * 100, "\n")

    for row in rows:
        x.add_row(row)
    # for k, v in rows.items():
    #     x.add_row([k] + ["{}({})".format(v[net_name][0], v[net_name][1]) for net_name in x.field_names])
    print("Format is: time (#success/#total) (#timeout)")
    print(x)

    for t, row in enumerate(rows):
        print("\t{} &&&".format(t + 2))
        print(" &&& ".join(row[1:]).replace("  ", " "), "&")
        print("\t\\\\")
    print("#"*100)
    print("Average run time {} seconds, over {} points".format(total_time / total_points , total_points))
    print("#"*100)


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
    for e in exp:
        if 'number_of_updates' in e['stats']:
            if e['stats']['number_of_updates'] == e['stats']['property_queries'] and not e['result']:
                timeout_exp.append(e)

    safe_mean = lambda x: np.mean(x) if len(x) > 0 else 0

    if len(exp) == len(timeout_exp):
        avg_total_time = None
    else:
        avg_total_time = (sum([e['time'] for e in exp]) - sum([e['time'] for e in timeout_exp])) / (
                    len(exp) - len(timeout_exp))
    return {
        'total': len(exp),
        'total_success': len(success_exp),
        'success_rate': len(success_exp) / len(exp),
        'len_timeout': len(timeout_exp),
        'avg_total_time': safe_mean([e['time'] for e in exp]),
        'avg_total_time_no_timeout': avg_total_time,
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


if __name__ == "__main__":
    parse_results_file([
        ('rnn2_fc0', "final_gurobi_exps/rnn2_fc0.pkl"),
        ('rnn2_fc1', "final_gurobi_exps/rnn2_fc1.pkl"),
        ('rnn4_fc1', "final_gurobi_exps/rnn4_fc1.pkl"),
        ('rnn4_fc2', "final_gurobi_exps/rnn4_fc2.pkl"),
        ('rnn4_rnn2', "final_gurobi_exps/rnn4_rnn2.pkl")
        ('rnn8_fc1', "final_gurobi_exps/rnn8_fc1.pkl"),
        # ('rnn4_rnn2_1', "final_gurobi_exps/runnerup/rnn4_rnn2.pkl"),
    ])
    exit(0)


    # import time
    # time.sleep(60 * 180)
    # import time
    # wait = 0*60*200
    # print(str(datetime.now()), "sleeping for {} minuts".format(wait / 60))
    # time.sleep(wait)

    net_options = ['model_20classes_rnn2_fc32_epochs200.h5', 'model_20classes_rnn4_fc32_epochs40.h5',
                   'model_classes20_1rnn8_1_32_4.h5', 'model_classes20_1rnn2_0_64_4.h5',
                   'model_20classes_rnn4_fc16_fc32_epochs3.h5', 'model_20classes_rnn4_rnn2_fc16_epochs3.h5']

    other_idx_method = [lambda x: np.argmin(x)]
    # other_idx_method = [lambda x: np.argsort(x)[-2]]
    # gurobi_ptr = partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20000,
    #                      use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    gurobi_ptr = partial(AlphasGurobiBasedMultiLayer, update_strategy_ptr=Relative_Step, random_threshold=20000,
                         use_relu=True, add_alpha_constraint=True, use_counter_example=True)

    t_range = range(2, 20)
    points = pickle.load(open(POINTS_PATH, "rb"))

    # pbar = tqdm(total=len(net_options[2:]) * 5 * len(t_range))
    # counter = 0
    # for p in net_options[2:]:
    #     for i, point in enumerate(points[:5]):
    #         for t in t_range:
    #             path = os.path.join("models", p)
    #             idx_max, _ = get_out_idx(point, t, path, other_idx_method[0])
    #             if t == t_range[0]:
    #                 first_idx = idx_max
    #             elif first_idx != idx_max:
    #                 print(path, t, first_idx, idx_max, i)
    #                 counter += 1
    #             pbar.update(1)
    # print('total different', counter)
    # exit(0)


    # net = ['rnn4_try.h5']
    # run_all_experiments([net_options[2]], points[:1], [2], other_idx_method, gurobi_ptr, steps_num=1000, save_results=False)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            generate_points(MODELS_FOLDER)
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

                save_results = True

                # print("*"*100, "\nDEBUG DEBUG DEUBG")
                # print("*"*100)
                # t_range = [3]
                # save_results = False

                run_all_experiments(net, points[:5], t_range, other_idx_method, gurobi_ptr, steps_num=10,
                                    continue_pickle=continue_pickle, save_results=save_results)
            else:
                print('running all nets, one after the other')
                for net in ['model_20classes_rnn4_fc32_epochs40.h5', 'model_20classes_rnn4_fc16_fc32_epochs3.h5',
                            'model_classes20_1rnn8_1_32_4.h5', 'model_classes20_1rnn2_0_64_4.h5']:
                    run_all_experiments([net], points[:5], t_range, other_idx_method, gurobi_ptr, steps_num=30,
                                        continue_pickle='')
        exit(0)

    point = np.array([-1.90037058, 2.80762335, 5.6615233, -3.3241606, -0.83999373, -4.67291775,
                      -2.37340524, -3.94152213, 1.78206783, -0.37256191, 1.07329743, 0.02954765,
                      0.50143303, -3.98823161, -1.05437203, -1.98067338, 6.12760627, -2.53038902,
                      -0.90698131, 4.40535622, -3.30067319, -1.60564116, 0.22373327, 0.43266462,
                      5.45421917, 4.11029858, -4.65444165, 0.50871269, 1.40619639, -0.7546163,
                      3.68131841, 1.18965503, 0.81459484, 2.36269942, -2.4609835, -1.14228611,
                      -0.28604645, -6.39739288, -3.54854402, -3.21648808])
    net = "model_20classes_rnn4_rnn2_fc16_epochs3.h5"
    t_range = range(2,4)
    gurobi_multi_ptr = partial(AlphasGurobiBasedMultiLayer, update_strategy_ptr=Relative_Step, random_threshold=20000,
                         use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    run_all_experiments([net], points[:5], t_range, other_idx_method, gurobi_multi_ptr, save_results=0, steps_num=2)
    # run_all_experiments([net_options[2]], points[:5], t_range, other_idx_method, gurobi_ptr, save_results=0)
