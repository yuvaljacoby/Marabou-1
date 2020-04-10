from functools import partial
from itertools import product

import numpy as np
import pytest

from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronIISBased import GurobiMultiLayerIIS
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronRandomImprove import GurobiMultiLayerRandom
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from polyhedron_algorithms.GurobiBased.SingleLayerBase import GurobiSingleLayer

points = [
    np.array([1.0] * 40),
    np.array([0.5] * 40),
    np.array([-1.0] * 40),
    np.array([1.90037058, 2.80762335, 5.661523, -3.3241606, -0.83999373, -4.67291775, -2.3734052, 3.94152213, 1.7820678,
              -0.37256191, 1.07329743, 0.02954765, 0.50143303, 3.98823161, -1.05437203, -1.98067338, 6.12760627,
              -2.5303890, -0.9069811, 4.40535622, -3.30067319, -1.60564116, 0.22373327, 0.43266462, 5.45421917,
              4.11029858, -4.65444165, 0.50871269, 1.40619639, -0.7546163, 3.68131841, 1.18965503, 0.81459484,
              2.36269942, -2.4609835, -1.1422861, -0.28604645, -6.39739288, 3.54854402, -3.2164880]),
    np.array([0.46656992, -0.14269506, 1.73669903, 2.76984167, -3.63039427,
              3.571188, -5.8216382, 3.09517914, -3.81565903, -5.54560341,
              -5.18293851, -0.91985959, 2.60588765, 1.39407387, 4.83666484,
              1.39725658, 0.35908562, -0.53350263, 3.09702831, 2.89868454,
              3.33938659, 1.93290896, -4.42773094, -2.76334927, 0.27125894,
              -1.51579716, -2.87779028, 4.28366679, -0.22930092, -4.56011088,
              1.8762871, -0.41762599, -0.78161998, 0.91724521, -0.2930083,
              4.57093627, 0.41214491, 2.77216615, 0.37158991, 2.39547404]),
    np.array([-2.38736795, -2.06448808, 1.55842541, -5.16959939, -4.12698993,
              6.67856379, -1.16220196, 6.61150044, -0.04729925, 0.94958595,
              -4.92366119, 2.88827515, -0.55605694, 4.16684607, -0.07475666,
              -1.54551184, 3.64153339, -5.06847272, 1.49826572, 1.52659644,
              4.62871035, 2.25001262, 4.14728649, 2.04409215, 0.80052778,
              1.65777114, -5.92517124, 3.02789316, -5.4572159, -2.14427204,
              -1.03729186, -6.18893842, -3.66178804, -1.17103946, -1.03086863,
              2.62874236, 2.87483596, 5.07820105, 1.68202816, -0.98712417]),
    np.array([4.09335522e+00, 8.96169695e-01, 1.07702513e+00, -4.74611218e+00,
              -3.84326866e+00, 5.80201029e-01, 3.90639469e+00, -1.23798776e+00,
              -4.81164004e+00, 2.39150123e+00, 2.77127410e-04, 2.35094486e+00,
              4.59022167e+00, -2.40180867e+00, -2.44018059e-02, -4.06184240e+00,
              -1.57673348e+00, 9.57764218e-01, -2.09173796e+00, -6.19681161e+00,
              3.05176419e+00, -1.74501084e+00, 1.08666606e+00, -7.07330408e-01,
              1.06057652e+00, 2.34367232e+00, -1.40046072e-01, 1.16467488e+00,
              4.07313304e-01, 2.96174215e+00, 4.22257730e+00, -1.97159818e+00,
              1.93427908e-01, 8.78444200e-01, 3.30449797e+00, -2.81077158e+00,
              -1.70496579e+00, 2.78837981e+00, 1.78557462e+00, 4.46912768e-01])
]
paths = ['./models/model_20classes_rnn2_fc32_fc32_fc32_fc32_fc32_epochs50.h5',
         './models/model_20classes_rnn4_fc32_fc32_fc32_fc32_fc32_epochs50.h5',
         # './models/model_20classes_rnn10_fc32_fc32_fc32_fc32_fc32_epochs50.h5'
         ]
multi_layer_paths = ['./FMCAD_EXP/models/model_20classes_rnn4_rnn4_fc32_fc32_0100.ckpt',
                     './FMCAD_EXP/models/model_20classes_rnn4_rnn4_rnn4_fc32_fc32_fc32_0200.ckpt',
                     # './FMCAD_EXP/models/model_20classes_rnn8_rnn4_rnn4_fc32_fc32_fc32_fc32_0150'
                     ]


def test_specific():
    point = np.array([-1.0] * 40)
    net_path = './models/model_20classes_rnn4_fc32_fc32_fc32_fc32_fc32_epochs50.h5'
    n = 3
    gurobi_ptr = partial(GurobiSingleLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    method = lambda x: np.argsort(x)[-2]
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, _, _ = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
    assert res


def test_specific_multilayer():
    point = np.array([1.0] * 40)
    net_path = multi_layer_paths[0]
    n = 3
    gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    method = lambda x: np.argsort(x)[-2]
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, _, _ = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
    assert res


def test_fast_unsat():
    point = np.array([0.8] * 40)
    net_path = './models/model_20classes_rnn2_fc32_fc32_fc32_fc32_fc32_epochs50.h5'
    n = 2
    gurobi_ptr = partial(GurobiSingleLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    idx_max = 0
    other_idx = 16
    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                          gurobi_ptr, n)
    assert res


@pytest.mark.parametrize(['point', 'n', 'net_path'], product(*[points, [2, 5], paths]))
def test_using_gurobi(point, n, net_path):
    method = lambda x: np.argsort(x)[-2]
    gurobi_ptr = partial(GurobiSingleLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    print(idx_max, other_idx)
    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                          gurobi_ptr, n)
    assert res


@pytest.mark.parametrize(['point', 'n', 'net_path'], product(*[points, [2, 4], multi_layer_paths]))
def test_using_multilayer_gurobi(point, n, net_path):
    print(net_path)
    print(n)
    print(point)
    method = lambda x: np.argsort(x)[-2]
    gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                          gurobi_ptr, n)
    assert res


@pytest.mark.parametrize(['point', 'n', 'net_path'], product(*[points[:2], [5], multi_layer_paths[:2]]))
def test_using_multilayer_gurobi_random_improve(point, n, net_path):
    print(net_path)
    print(n)
    print(point)
    method = lambda x: np.argsort(x)[-2]
    gurobi_ptr = partial(GurobiMultiLayerRandom, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True, num_steps=10)
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
    assert res

@pytest.mark.parametrize(['point', 'n', 'net_path'], product(*[points[:2], [5], multi_layer_paths[:2]]))
def test_using_multilayer_gurobi_IIS_improve(point, n, net_path):
    print(net_path)
    print(n)
    print(point)
    method = lambda x: np.argsort(x)[-2]
    gurobi_ptr = partial(GurobiMultiLayerIIS, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True, num_steps=15)
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
    assert res




# def test_example_polyhedron_more_expressive():
#     # return 1
#     point = np.array([-1.0] * 40)
#     net_path = './FMCAD_EXP/models/model_20classes_rnn8_fc32_fc32_fc32_0050.ckpt'
#     n = 4
#     method = lambda x: np.argsort(x)[-2]
#     idx_max, other_idx = get_out_idx(point, n, net_path, method)
#     gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
#                          use_counter_example=True)
#     res, _, _ = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
#     assert not res
#
#     gurobi_ptr = partial(GurobiMultiLayer, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
#                          use_counter_example=True)
#     res, _, _ = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
#     assert res


def test_cond_x_l_zero():
    # This is an example why we have
    #   cond_x_l = min(cond_x_l, 0)
    # line in GurobiBased.py

    test_using_multilayer_gurobi(np.array([1.0] * 40), 4,
                                 './FMCAD_EXP/models/model_20classes_rnn4_rnn4_fc32_fc32_0100.ckpt')

# @pytest.mark.parametrize(['point', 'n', 'net_path'], product(*[points, [2, 5], paths]))
# def test_using_gurobi_SAT(point, n, net_path):
#     method = lambda x: np.argsort(x)[-2]
#     gurobi_ptr = partial(AlphasGurobiBased, use_relu=True, add_alpha_constraint=True, use_counter_example=True)
#     idx_max, other_idx = get_out_idx(point, n, net_path, method)
#     res, queries_stats, alpha_history = adversarial_query(point, 100, idx_max, other_idx, net_path,
#                                                           gurobi_ptr, n)
#     assert not res
