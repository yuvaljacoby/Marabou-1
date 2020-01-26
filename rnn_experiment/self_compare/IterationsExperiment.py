import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm

from maraboupy.keras_to_marabou_rnn import adversarial_query, get_out_idx
from rnn_algorithms.GurobiBased import AlphasGurobiBased
from rnn_algorithms.Update_Strategy import Relative_Step

# BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
BASE_FOLDER = "/home/yuval/projects/Marabou/"
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
                        save_results=True):
    assert len(points) > 20
    results = defaultdict(list)
    pickle_path = 'gurobi' + str(datetime.now()).replace('.', '').replace(' ', '') + ".pkl"
    pbar = tqdm(total = len(other_idx_method) * len(points) * len(net_options) * len(t_range))
    for method in other_idx_method:
        for idx, point in enumerate(points):
            for path in net_options:
                if not os.path.exists(path):
                    path = os.path.join(MODELS_FOLDER, path)
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                for t in t_range:
                    idx_max, other_idx = get_out_idx(point, t, path, method)
                    net_name = ''.join(path.split('.')[:-1]).split('/')[-1]
                    name = "{}_{}_{}_{}".format(net_name, radius, other_idx, t)
                    result = run_experiment(point, radius, idx_max, other_idx, path, gurobi_ptr, t, steps=steps_num)
                    result.update(
                        {'h5_file': net_name, 't': t, 'other_idx': other_idx, 'in_tesnor': point,
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


def parse_results_file(pickle_path):
    d = pickle.load(open(pickle_path, "rb"))
    for key, value in d.items():
        print("{}\n".format(key), parse_dictionary(value), "\n" + "#" * 100)


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

    return {
        'total': len(exp),
        'success_rate': len(success_exp) / len(exp),
        'avg_total_time': np.mean([e['time'] for e in success_exp]),
        'avg_invariant_time': np.mean([e['stats']['invariant_times']['avg'] for e in success_exp]),
        'avg_property_time': np.mean([e['stats']['property_times']['avg'] for e in success_exp]),
        'avg_step_time': np.mean([e['stats']['step_times']['avg'] for e in success_exp]),
        'num_invariant_avg': np.mean([e['stats']['invariant_queries'] for e in success_exp]),
        'num_property_avg': np.mean([e['stats']['property_queries'] for e in success_exp]),
        'num_step_avg': np.mean([e['stats']['step_queries'] for e in success_exp])
    }


if __name__ == "__main__":
    net_options = ['model_20classes_rnn2_fc32_epochs200.h5', 'model_20classes_rnn4_fc32_epochs40.h5',
                   'model_classes20_1rnn8_1_32_4.h5']
    other_idx_method = [lambda x: np.argmin(x)]
    default_gurobi = gurobi_ptr = partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20000,
                                          use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    t_range = range(2, 15)
    points = pickle.load(open(POINTS_PATH, "rb"))

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            generate_points(MODELS_FOLDER)
        if sys.argv[1] == 'analyze':
            parse_results_file(sys.argv[2])
        if sys.argv[1] == 'exp':
            if str.isnumeric(sys.argv[2]):
                net = [net_options[int(sys.argv[2])]]
            else:
                net = [sys.argv[2]]
            run_all_experiments(net, points, t_range, other_idx_method, gurobi_ptr, steps_num=100)
        exit(0)

    run_all_experiments([net_options[1]], points, t_range, other_idx_method, gurobi_ptr)

