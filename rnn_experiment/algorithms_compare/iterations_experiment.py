import pickle
import numpy as np
from maraboupy.MarabouRNNMultiDim import prove_multidim_property
from maraboupy.keras_to_marabou_rnn import adversarial_query, get_out_idx
from rnn_algorithms.RandomAlphasSGD import RandomAlphasSGD
from rnn_algorithms.MaxAlphasSGDInfStart import MaxAlphasSGD
from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
from maraboupy.keras_to_marabou_rnn import RnnMarabouModel, calc_min_max_by_radius, negate_equation
from maraboupy import MarabouCore
from timeit import default_timer as timer
import seaborn as sns
from rns_verify.verify_keras import verify_query as rns_verify_query

import matplotlib.pyplot as plt

MODELS_FOLDER = "/home/yuval/projects/Marabou/models/"
FIGUERS_FOLDER = "/home/yuval/projects/Marabou/figures/"


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
    # plt.plot(x_idx, our_results, 'o', color='blue')
    # plt.plot(x_idx, rnsverify_results, 'o', color='orange')
    sns.scatterplot(x_idx, our_results)
    sns.scatterplot(x_idx, rnsverify_results)
    plt.legend(['ours', 'rnsverify', ], loc='upper left')
    plt.title(exp_name)
    plt.tight_layout()
    plt.xlabel('number of iterations')
    plt.ylabel('time(sec)')
    plt.savefig((FIGUERS_FOLDER + exp_name + ".pdf").replace(' ', '_'))
    plt.show()


experiemnts = [
    # {'idx_max': 9, 'other_idx': 2,
    #  'in_tensor': [10] * 40, 'radius': 0.01,
    #  'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    {'idx_max': None, 'other_idx': None,
     'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
       -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
       -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
        5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
        3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
        1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
        1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
       -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
       -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
        2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
     'h5_path': "{}/model_20classes_rnn2_fc32_epochs200.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    {'idx_max': None, 'other_idx': None,
     'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
       -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
       -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
        5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
        3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
        1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
        1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
       -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
       -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
        2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
     'h5_path': "{}/model_20classes_rnn4_fc32_epochs40.h5".format(MODELS_FOLDER), 'n_iterations': 25},
]

if __name__ == "__main__":
    # idx_max = 4
    # other_idx = 0
    # in_tensor = [10] * 40
    # n_iterations = 20  # 1000?
    # r = 0
    # model_path = 'models/model_classes5_1rnn2_0_64_4.h5'

    for exp in experiemnts:
        if exp['idx_max'] is None:
            exp['idx_max'], exp['other_idx'] = get_out_idx(exp['in_tensor'], exp['n_iterations'], exp['h5_path'])
        our, rns = run_experiment(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
                                  exp['h5_path'], max_iterations=exp['n_iterations'])
        rnn_dim = exp['h5_path'].split('/')[-1].split('_')[2].replace('1rnn', '')
        exp_name = 'verification time as a function of iterations, one rnn cell dimension: {}'.format(rnn_dim)

        pickle_dir = "pickles/rns_verify_exp/"
        pickle_path = pickle_dir + "{}_{}.pkl".format(exp['h5_path'].split("/")[-1].split(".")[-2], exp['n_iterations'])
        pickle.dump({'our' : our, 'rns' : rns, 'exp_name' : exp_name}, open(pickle_path, "wb"))

        plot_results(our, rns, exp_name)
