import pickle
import sys
from functools import partial

import numpy as np
from tqdm import tqdm

from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from rnn_experiment.self_compare.generate_random_points import POINTS_PATH


def find_first_point(net_path, points_path, other_idx_method, max_n_iter=10):
    gurobi_ptr = partial(GurobiMultiLayer, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    points = pickle.load(open(points_path, "rb"))
    i = 0
    start_point = 10
    pbar = tqdm(total=(max_n_iter + 1 - start_point) * len(points) * len(other_idx_method))
    for n in range(start_point, max_n_iter + 1):
        for point in points:
            m = 0
            for method in other_idx_method:
                idx_max, other_idx = get_out_idx(point, n, net_path, method)
                res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                                      gurobi_ptr, n)
                pbar.update(1)
                if not res:
                    print("$" * 100)
                    print(m, n)
                    print(net_path)
                    print(point)
                    print(idx_max, other_idx)
                    print("$" * 100)
                    exit(0)
                i += 1
            m += 1


if __name__ == '__main__':
    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 7)]
    net_path = sys.argv[1] if len(sys.argv) > 1 else\
        "./FMCAD_EXP/models/model_20classes_rnn4_rnn4_fc32_fc32_0050.ckpt"
    points_path = sys.argv[2] if len(sys.argv) > 2 else POINTS_PATH
    find_first_point(net_path, points_path, other_idx_method)