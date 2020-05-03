import pickle
from functools import partial

import numpy as np

from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from rnn_experiment.self_compare.IterationsExperiment import run_all_experiments
from rnn_experiment.self_compare.generate_random_points import POINTS_PATH


def test_single_network():
    points = pickle.load(open(POINTS_PATH, "rb"))[:2]
    t_range = range(4, 6)
    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 4)]
    net = "models/old/model_20classes_rnn4_rnn2_fc16_epochs3.h5"
    gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    results = run_all_experiments([net], points, t_range, other_idx_method, gurobi_ptr, steps_num=10, save_results=True)
    assert len(results) > 0


if __name__ == '__main__':
    # Upper window 9, lower 8
    points = [pickle.load(open(POINTS_PATH, "rb"))[3]]
    t_range = [8]
    other_idx_method = [lambda x: np.argsort(x)[-2]]
    net = "ATVA_EXP/models/epochs100/model_20classes_rnn4_rnn2_fc32_fc32_fc32_fc32_fc32_epochs100.h5"
    gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True, max_steps=20)
    results = run_all_experiments([net], points, t_range, other_idx_method, gurobi_ptr, steps_num=2, save_results=True)
    assert len(results) > 0