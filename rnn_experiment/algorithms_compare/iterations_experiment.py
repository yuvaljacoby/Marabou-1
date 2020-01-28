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
from tqdm import tqdm
import matplotlib.pyplot as plt

MODELS_FOLDER = "/home/yuval/projects/Marabou/models/"
# FIGUERS_FOLDER = "/home/yuval/projects/Marabou/figures/"
FIGUERS_FOLDER = "/home/yuval/projects/MarabouPapers/rnn/figures/"

from functools import partial
from rnn_algorithms.GurobiBased import AlphasGurobiBased

def run_experiment(in_tensor, radius, idx_max, other_idx, h5_file, max_iterations=100):
    our_results = []
    rnsverify_results = []
    for i in tqdm(range(2, max_iterations)):
        # rnsverify_time = rns_verify_query(h5_file, in_tensor, idx_max, other_idx, i, radius)
        rnsverify_time = -1
        gurobi_ptr = partial(AlphasGurobiBased, random_threshold=20000, use_relu=True, add_alpha_constraint=True,
                             use_counter_example=True)
        try:
            start = timer()
            res, _, _ = adversarial_query(in_tensor, radius, idx_max, other_idx, h5_file, gurobi_ptr, i)
            our_time = timer() - start
        except ValueError:
            res = False
            our_time = -1
        # assert res

        our_results.append(our_time)
        rnsverify_results.append(rnsverify_time)
        # results.append((our_time, rnsverify_time))
        print('iteration: {}, results: {}, {}'.format(i, our_results[-1],
                                                      rnsverify_results[-1]))
    return our_results, rnsverify_results


def plot_results(our_results, rnsverify_results, exp_name):
    assert len(our_results) == len(rnsverify_results)
    x_idx = range(2, len(our_results) + 2)

    plt.figure(figsize=(12.5, 9))
    sns.scatterplot(x_idx, our_results, s=200)
    sns.scatterplot(x_idx, rnsverify_results , s=200)

    plt.legend(['RnnVerify', 'Unrolling'], loc='upper left', fontsize=32)

    plt.xlabel('Number of Iterations ($T_{max}$)', fontsize=36)
    plt.ylabel('Time (seconds)', fontsize=36)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.savefig((FIGUERS_FOLDER + "rns_ours_rnn2_fc0.pdf").replace(' ', '_'), dpi=100)

    # small version:
    # plt.figure(figsize=(14, 11))
    # sns.scatterplot(x_idx, our_results, s=220)
    # sns.scatterplot(x_idx, rnsverify_results , s=220)

    # plt.legend(['RnnVerify', 'Unrolling'], loc='upper left', fontsize=42)

    # plt.xlabel('Number of Iterations ($T_{max}$)', fontsize=48)
    # plt.ylabel('Time (seconds)', fontsize=48)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.savefig((FIGUERS_FOLDER + "rns_ours_rnn2_fc0_03.pdf").replace(' ', '_'), dpi=100)
    # plt.show()


experiemnts = [
    # {'idx_max': 9, 'other_idx': 2,
    #  'in_tensor': [10] * 40, 'radius': 0.01,
    #  'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    # {'idx_max': None, 'other_idx': None,
    #  'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
    #    -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
    #    -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
    #     5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
    #     3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
    #     1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
    #     1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
    #    -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
    #    -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
    #     2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
    #  'h5_path': "{}/model_20classes_rnn2_fc32_epochs200.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    # {'idx_max': None, 'other_idx': None,
    #  'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
    #    -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
    #    -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
    #     5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
    #     3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
    #     1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
    #     1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
    #    -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
    #    -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
    #     2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
    #  'h5_path': "{}/model_20classes_rnn4_fc32_epochs40.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    {'idx_max': 19, 'other_idx': 8,
     'in_tensor':np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
       -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
       -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
        5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
        3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
        1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
        1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
       -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
       -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
        2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
     'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
]

if __name__ == "__main__":
    # idx_max = 4
    # other_idx = 0
    # in_tensor = [10] * 40
    # n_iterations = 20  # 1000?
    # r = 0
    # model_path = 'models/model_classes5_1rnn2_0_64_4.h5'
    # results_path = "pickles/rns_verify_exp/model_classes20_1rnn2_0_64_4_25.pkl"
    # d = pickle.load(open(results_path, "rb"))
    # plot_results(d['our'], d['rns'], d['exp_name'])
    # exit(0)

    for exp in experiemnts:
        if exp['idx_max'] is None:
            exp['idx_max'], exp['other_idx'] = get_out_idx(exp['in_tensor'], exp['n_iterations'], exp['h5_path'])
            print(exp['idx_max'], exp['other_idx'])
        our, rns = run_experiment(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
                                  exp['h5_path'], max_iterations=exp['n_iterations'])
        rnn_dim = exp['h5_path'].split('/')[-1].split('_')[2].replace('1rnn', '')
        exp_name = 'verification time as a function of iterations, one rnn cell dimension: {}'.format(rnn_dim)

        pickle_dir = "pickles/rns_verify_exp/"
        pickle_path = pickle_dir + "ONLYOURS_{}_{}_{}.pkl".format(exp['h5_path'].split("/")[-1].split(".")[-2], exp['n_iterations'],
                                                                  hash(str(exp['in_tensor'])))
        # pickle.dump({'our' : our, 'rns' : rns, 'exp_name' : exp_name}, open(pickle_path, "wb"))

        # plot_results(our, rns, exp_name)
