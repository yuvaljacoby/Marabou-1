import numpy as np
from maraboupy.MarabouRNNMultiDim import prove_multidim_property
from maraboupy.keras_to_marabou_rnn import adversarial_query
from rnn_algorithms.RandomAlphasSGD import RandomAlphasSGD
from rnn_algorithms.MaxAlphasSGDInfStart import MaxAlphasSGD
from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
from maraboupy.keras_to_marabou_rnn import RnnMarabouModel, calc_min_max_by_radius, negate_equation
from maraboupy import MarabouCore
from timeit import default_timer as timer
from rns_verify.verify_keras import verify_query as rns_verify_query

import matplotlib.pyplot as plt

MODELS_FOLDER = "/home/yuval/projects/Marabou/models/"
FIGUERS_FOLDER = "/home/yuval/projects/Marabou/figures/"


# def adversarial_query(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str, algorithm_ptr,
#                       n_iterations=10):
#     '''
#     Query marabou with adversarial query
#     :param x: base_vector (input vector that we want to find a ball around it)
#     :param radius: determines the limit of the inputs around the base_vector
#     :param y_idx_max: max index in the output layer
#     :param other_idx: which index to compare max idx
#     :param h5_file_path: path to keras model which we will check on
#     :param algorithm_ptr: TODO
#     :param n_iterations: number of iterations to run
#     :return: True / False
#     '''
#
#     # assert_adversarial_query(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test)
#     rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
#     xlim = calc_min_max_by_radius(x, radius)
#     rnn_model.set_input_bounds(xlim)
#     initial_values = rnn_model.get_rnn_min_max_value_one_iteration(xlim)
#
#     # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
#     adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
#     adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])
#     adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])
#     adv_eq.setScalar(0)
#
#     rnn_output_idxs = rnn_model.rnn_out_idx
#     rnn_start_idxs = [i - 3 for i in rnn_output_idxs]
#
#     # Convert the initial values to the SGDAlphaAlgorithm format
#     rnn_max_values = [val[1] for val in initial_values]
#     rnn_min_values = [val[0] for val in initial_values]
#
#     assert sum([rnn_max_values[i] >= rnn_min_values[i] for i in range(len(rnn_max_values))]) == len(rnn_max_values)
#
#     algorithm = algorithm_ptr((rnn_min_values, rnn_max_values), rnn_start_idxs, rnn_output_idxs)
#     # rnn_model.network.dump()
#     start = timer()
#     assert prove_multidim_property(rnn_model.network, rnn_start_idxs, rnn_output_idxs, [negate_equation(adv_eq)],
#                                    algorithm)
#     end = timer()
#     return end - start

# x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str, algorithm_ptr,
#                       n_iterations=10, steps_num=5000
def run_experiment(in_tensor, radius, idx_max, other_idx, h5_file, max_iterations=100):
    our_results = []
    rnsverify_results = []
    for i in range(2, max_iterations):
        rnsverify_time = rns_verify_query(h5_file, in_tensor, idx_max, other_idx, i, radius)

        start = timer()
        res, _, _ = adversarial_query(in_tensor, radius, idx_max, other_idx, h5_file, IterateAlphasSGD, i)
        our_time = timer() - start
        assert res

        our_results.append(our_time)
        rnsverify_results.append(rnsverify_time)
        # results.append((our_time, rnsverify_time))
        print('iteration: {}, results: {}, {}'.format(i, our_results[-1],
                                                      rnsverify_results[-1]))
    return our_results, rnsverify_results


def plot_results(our_results, rnsverify_results, exp_name):
    assert len(our_results) == len(rnsverify_results)
    x_idx = range(2, len(our_results) + 2)
    plt.plot(x_idx, our_results, 'o', color='r')
    plt.plot(x_idx, rnsverify_results, 'o', color='g')
    plt.legend(['ours', 'rnsverify', ], loc='upper left')
    plt.title(exp_name)
    plt.tight_layout()
    plt.xlabel('number of iterations')
    plt.ylabel('time(sec)')
    plt.savefig((FIGUERS_FOLDER + exp_name + ".pdf").replace(' ', '_'))
    plt.show()


experiemnts = [
    # {'idx_max': 4, 'other_idx': 0, 'in_tensor': [10] * 40,
    #  'radius': 0, 'h5_path': "{}/model_classes5_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    # {'idx_max': 13, 'other_idx': 15,
    #  'in_tensor': np.array([6.3, 9.4, 9.6, 3.1, 8.5, 9.4, 7.2, 8.6, 3.8, 1.4, 0.7, 7.8, 1.9, 8.2, 6.2, 3.6, 8.7, 1.7
    #                            , 2.8, 4.8, 4.3, 5.1, 3.8, 0.8, 2.4, 7.6, 7.3, 0., 3.3, 7.4, 0., 2.1, 0.5, 8., 7.1, 3.9
    #                            , 3., 8.3, 5.6, 1.8]), 'radius': 0.01,
    #  'h5_path': "{}/model_classes20_1rnn4_0_2_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    {'idx_max': 9, 'other_idx': 2,
     'in_tensor': [10] * 40, 'radius': 0.01,
     'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
]

if __name__ == "__main__":
    # idx_max = 4
    # other_idx = 0
    # in_tensor = [10] * 40
    # n_iterations = 20  # 1000?
    # r = 0
    # model_path = 'models/model_classes5_1rnn2_0_64_4.h5'

    for exp in experiemnts:
        our, rns = run_experiment(exp['in_tensor'], 0, exp['idx_max'], exp['other_idx'],
                                  exp['h5_path'], max_iterations=exp['n_iterations'])
        rnn_dim = exp['h5_path'].split('/')[-1].split('_')[2].replace('1rnn', '')
        exp_name = 'verification time as a function of iterations, one rnn cell dimension: {}'.format(rnn_dim)
        plot_results(our, rns, exp_name)
