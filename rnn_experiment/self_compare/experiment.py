import os
import pickle
from functools import partial
from timeit import default_timer as timer

from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from maraboupy.keras_to_marabou_rnn import adversarial_query, get_out_idx
from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
from rnn_algorithms.RandomAlphasSGD import RandomAlphasSGD
from rnn_algorithms.Update_Strategy import Absolute_Step, Relative_Step
from rnn_algorithms.WeightedAlphasSGD import WeightedAlphasSGD
from rnn_experiment.self_compare.draw_self_compare import draw_from_dataframe

BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
#BASE_FOLDER = "/home/yuval/projects/Marabou/"
MODELS_FOLDER = os.path.join(BASE_FOLDER, "models/")
EXPERIMENTS_FOLDER = os.path.join(BASE_FOLDER, "working_arrays/")
IN_SHAPE = (40,)

def classes20_1rnn2_1fc2():
    n_inputs = 40
    y_idx_max = 19
    other_idx = 10
    n_iterations = 10
    # (rnn_min_values, rnn_max_values), rnn_start_idxs, rnn_output_idxs
    algorithm = RandomAlphasSGD

    from timeit import default_timer as timer
    start = timer()
    assert adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_2_4.h5", algorithm, n_iterations)[0]
    rand_time = timer() - start

    algorithm = IterateAlphasSGD
    from timeit import default_timer as timer
    start = timer()
    assert adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_2_4.h5", algorithm, n_iterations)[0]
    iter_time = timer() - start
    return iter_time, rand_time


def run_one_comparison(in_tensor, radius, idx_max, other_idx, h5_file, n_iterations, algorithms_ptrs, steps_num=2500):
    results = {}

    for name, algo_ptr in algorithms_ptrs.items():
        print('starting algo:' + name)
        start = timer()

        res, iterations, alpha_history = adversarial_query(in_tensor, radius, idx_max, other_idx, h5_file, algo_ptr,
                                                           n_iterations, steps_num)
        # res = False
        # iterations = 23
        end = timer()
        if iterations is None:
            return None
        results[name] = {'time': end - start, 'result': res, 'iterations': iterations}
        print("%%%%%%%%% {} %%%%%%%%%".format(end - start))

    # print(results)
    row_result = [results[n]['result'] for n in algorithms_ptrs.keys()] \
                 + [results[n]['iterations'] for n in algorithms_ptrs.keys()] \
                 + [results[n]['time'] for n in algorithms_ptrs.keys()]

    return row_result


experiemnts = [
    {'idx_max': 1, 'other_idx': 4, 'in_tensor': np.array([0.23300637, 0.0577466, 0.88960908, 0.02926062, 0.4322654,
                                                          0.05116153, 0.93342266, 0.3143915, 0.39245229, 0.1144419,
                                                          0.08748452, 0.24332963, 0.34622415, 0.42573235, 0.26952168,
                                                          0.53801347, 0.26718764, 0.24274057, 0.11475819, 0.9423371,
                                                          0.70257952, 0.34443971, 0.08917664, 0.50140514, 0.75890139,
                                                          0.65532994, 0.74165648, 0.46543468, 0.00583174, 0.54016713,
                                                          0.74460554, 0.45771724, 0.59844178, 0.73369685, 0.50576504,
                                                          0.91561612, 0.39746448, 0.14791963, 0.38114261, 0.24696231]),
     'radius': 0, 'h5_path': "{}/model_classes5_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 5},

    {'idx_max': 9, 'other_idx': 2, 'in_tensor': np.array([10] * 40),
     'radius': 0, 'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 1000},
    {'idx_max': 9, 'other_idx': 14, 'in_tensor': np.array([0.43679032, 0.51105192, 0.01603254, 0.45879329, 0.64639347,
                                                           0.39209051, 0.98618169, 0.49293316, 0.70440262, 0.08594672,
                                                           0.17252591, 0.4940284, 0.83947774, 0.55545332, 0.8971317,
                                                           0.72996308, 0.23706766, 0.66869303, 0.74949942, 0.57524252,
                                                           0.94886307, 0.31034989, 0.41785656, 0.5697128, 0.74751913,
                                                           0.48868271, 0.22672374, 0.6350584, 0.88979192, 0.97493685,
                                                           0.96969836, 0.99457045, 0.89433312, 0.19916606, 0.63957592,
                                                           0.02826659, 0.08104817, 0.20176526, 0.1114994, 0.29297289]),
     'radius': 0.01, 'h5_path': "{}/model_classes20_1rnn4_1_32_4.h5".format(MODELS_FOLDER), 'n_iterations': 10},

    {'idx_max': 4, 'other_idx': 0, 'in_tensor': [10] * 40,
     'radius': 0, 'h5_path': "{}/model_classes5_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 100},

    # {'idx_max': 1, 'other_idx': 0, 'in_tensor': np.array([0.19005403, 0.51136299, 0.67302099, 0.59573087, 0.78725824,
    #              0.47257432, 0.65504724, 0.69202802, 0.16531246, 0.84543712,
    #              0.73715671, 0.03674481, 0.39459927, 0.0107714, 0.15337461,
    #              0.44855902, 0.894079, 0.48551109, 0.08504609, 0.74320624,
    #              0.52363974, 0.80471539, 0.06424345, 0.65279486, 0.15554268,
    #              0.63541206, 0.15977761, 0.70137553, 0.34406331, 0.59930546,
    #              0.8740703, 0.89584981, 0.67799938, 0.78253788, 0.33091662,
    #              0.74464927, 0.69366703, 0.96878231, 0.58014617, 0.41094702]),
    #  'radius': 0, 'h5_path': "{}/model_classes20_1rnn8_0_64_100.h5".format(MODELS_FOLDER), 'n_iterations': 5},
    {'idx_max': 13, 'other_idx': 15,
     'in_tensor': np.array([6.3, 9.4, 9.6, 3.1, 8.5, 9.4, 7.2, 8.6, 3.8, 1.4, 0.7, 7.8, 1.9, 8.2, 6.2, 3.6, 8.7, 1.7
                               , 2.8, 4.8, 4.3, 5.1, 3.8, 0.8, 2.4, 7.6, 7.3, 0., 3.3, 7.4, 0., 2.1, 0.5, 8., 7.1, 3.9
                               , 3., 8.3, 5.6, 1.8]), 'radius': 0.01,
     'h5_path': "{}/model_classes20_1rnn4_0_2_4.h5".format(MODELS_FOLDER), 'n_iterations': 500},
    {'idx_max': 13, 'other_idx': 8, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 11, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 11, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 4, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 3, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 5, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 18, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 13, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 1, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 4, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 18, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 10, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 14, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 4, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 0, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 14, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 3, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 0, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 12, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 15, 'radius': 0,
     'in_tensor': np.array([0.5276819, 0.18475515, 0.84049484, 0.05101713, 0.65297865,
                            0.64000277, 0.68154397, 0.17374351, 0.30507248, 0.04626008,
                            0.7759239, 0.78172294, 0.50306415, 0.55160325, 0.5682504,
                            0.44300834, 0.57119383, 0.80649904, 0.61513483, 0.24398878,
                            0.06143529, 0.39079038, 0.44596474, 0.43369353, 0.29322568,
                            0.55481591, 0.99188989, 0.23542668, 0.38524337, 0.08206859,
                            0.49990265, 0.38417799, 0.28538922, 0.99828765, 0.37680467,
                            0.35437023, 0.30889233, 0.28867692, 0.77390004, 0.43394878]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 7, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 5, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 10, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 7, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 11, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 3, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 6, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 8, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 10, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 15, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 16, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 13, 'other_idx': 0, 'radius': 0,
     'in_tensor': np.array([0.23568325, 0.70237988, 0.3166374, 0.74627353, 0.22739224,
                            0.95291164, 0.24576, 0.40445054, 0.36892157, 0.08092092,
                            0.57902572, 0.82626711, 0.56028983, 0.44413096, 0.81031513,
                            0.16866558, 0.93686892, 0.06449081, 0.23722131, 0.74648442,
                            0.85571668, 0.38552926, 0.99267338, 0.32447941, 0.14312457,
                            0.09192649, 0.94586319, 0.47928956, 0.23663665, 0.17057757,
                            0.86041277, 0.08677493, 0.38599816, 0.95202792, 0.25024289,
                            0.44774631, 0.21569389, 0.56249737, 0.24707225, 0.86096177]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_0_2_4.h5'.format(MODELS_FOLDER)},
    {'idx_max': 17, 'other_idx': 1, 'radius': 0,
     'in_tensor': np.array([0.20097051, 0.42479055, 0.34322315, 0.62437296, 0.25862801,
                            0.91247462, 0.63758399, 0.48962824, 0.31047408, 0.67587524,
                            0.49270075, 0.3388583, 0.10273122, 0.90489837, 0.55069923,
                            0.61848446, 0.7047673, 0.81861119, 0.20828558, 0.52244301,
                            0.66099873, 0.22497297, 0.04728098, 0.41952486, 0.30202405,
                            0.11102099, 0.08771732, 0.28779644, 0.58346806, 0.70012583,
                            0.53994143, 0.11465865, 0.26253562, 0.27067036, 0.61152968,
                            0.55825975, 0.59336236, 0.67116022, 0.05547967, 0.07961834]), 'n_iterations': 5,
     'h5_path': '{}/model_classes20_1rnn4_1_32_4.h5'.format(MODELS_FOLDER)},
]


def get_random_input(model_path, mean, var, n_iterations):

    while True:
        in_tensor = np.random.normal(mean, var, IN_SHAPE)
        if any(in_tensor < 0):
            print("resample got negative input")
            continue
        y_idx_max, other_idx = get_out_idx(in_tensor, n_iterations, model_path)
        if y_idx_max is not None and other_idx is not None and y_idx_max != other_idx:
            print(in_tensor)
            return in_tensor, y_idx_max, other_idx

def run_random_experiment(model_name, algorithms_ptrs, num_points=150, mean=10, var=3, radius=0.01, n_iterations=50):
    '''
    runs comperasion between all the given algorithms on num_points each pointed sampled from Normal(mean,var)
    :param model_name: h5 file in MODELS_FOLDER
    :return: DataFrame with results
    '''
    cols = ['exp_name'] + ['{}_result'.format(n) for n in algorithms_ptrs.keys()] +\
           ['{}_queries'.format(n) for n in algorithms_ptrs.keys()] +\
           ['{}_time'.format(n) for n in algorithms_ptrs.keys()]

    df = pd.DataFrame(columns=cols)
    model_path = os.path.join(MODELS_FOLDER, model_name)
    pickle_path = model_name + "_randomexp_" + "_".join(algorithms_ptrs.keys()) + str(datetime.now()).replace('.', '')
    for _ in tqdm(range(num_points)):
        in_tensor, y_idx_max, other_idx = get_random_input(model_path, mean, var, n_iterations)

        row_result = run_one_comparison(in_tensor, radius, y_idx_max, other_idx,
                                        model_path,
                                        n_iterations, algorithms_ptrs, steps_num=3000)
        if row_result is None:
            print("Got out vector with all entries equal")
            continue
        exp_name = model_path.split('.')[0].split('/')[-1] + '_' + str(n_iterations)
        df = df.append({cols[i]: ([exp_name] + row_result)[i] for i in range(len(row_result) + 1)}, ignore_index=True)
        print(df)
        pickle.dump(df, open("results_{}.pkl".format(pickle_path), "wb"))
    return df


def run_experiment_from_pickle(pickle_name, algorithms_ptrs):
    '''
    The search_for_input method is creating a pickle with all the examples, read that and compare algorithms using the
    examples from there
    :param pickle_name: name of file inside the EXPERIMENTS_FOLDER
    :param algorithms_ptrs: pointers to algorithms to run the experiment on
    :return: DataFrame with experiment results
    '''
    pickle_path = os.path.join(EXPERIMENTS_FOLDER, pickle_name)
    experiemnts = pickle.load(open(pickle_path, "rb"))
    model_name = pickle_name.replace(".pkl", "")
    model_path = "{}/{}.h5".format(MODELS_FOLDER, model_name)
    cols = ['exp_name'] + ['{}_result'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_queries'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_time'.format(n) for n in algorithms_ptrs.keys()]
    df = pd.DataFrame(columns=cols)

    for exp in experiemnts:
        row_result = run_one_comparison(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
                                        model_path,
                                        exp['n_iterations'], algorithms_ptrs)
        exp_name = model_path.split('.')[0].split('/')[-1] + '_' + str(exp['n_iterations'])
        df = df.append({cols[i]: ([exp_name] + row_result)[i] for i in range(len(row_result) + 1)}, ignore_index=True)
        print(df)
        pickle_path = model_name + "_".join(algorithms_ptrs.keys())
        pickle.dump(df, open("results_{}.pkl".format(pickle_path), "wb"))

    return df


def get_all_algorithms():
    RandomAlphasSGD_absolute_step = partial(RandomAlphasSGD, update_strategy_ptr=Absolute_Step)
    WeightedAlphasSGD_absolute_step = partial(WeightedAlphasSGD, update_strategy_ptr=Absolute_Step)
    Absolute_Step_Big = partial(Absolute_Step, options=[10 ** i for i in range(-5, 3)])
    RandomAlphasSGD_absolute_step_big = partial(RandomAlphasSGD, update_strategy_ptr=Absolute_Step_Big)
    WeightedAlphasSGD_absolute_step_big = partial(WeightedAlphasSGD, update_strategy_ptr=Absolute_Step_Big)
    WeightedAlphasSGD_relative_step = partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step)
    RandomAlphasSGD_relative_step = partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step)
    # IterateAlphasSGD_relative_step = partial(IterateAlphasSGD, update_strategy_ptr=Relative_Step)

    from collections import OrderedDict
    algorithms_ptrs = OrderedDict({
        'random_relative': RandomAlphasSGD_relative_step,
        'weighted_relative': WeightedAlphasSGD_relative_step,
        'iterate_absolute': RandomAlphasSGD_absolute_step,
        'weighted_absolute': WeightedAlphasSGD_absolute_step,
        'random_big_absolute': RandomAlphasSGD_absolute_step_big,
        'weighted_big_absolute': WeightedAlphasSGD_absolute_step_big,
    })

    return algorithms_ptrs


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)

    algorithms_ptrs = get_all_algorithms()
    # df = run_experiment_from_pickle("model_20classes_rnn4_fc32_epochs40.pkl", algorithms_ptrs)
    df = run_random_experiment("model_20classes_rnn4_fc32_epochs40.h5", algorithms_ptrs, num_points=150)
    draw_from_dataframe(df)

    # cols = ['exp_name'] + ['{}_result'.format(n) for n in algorithms_ptrs.keys()] + ['{}_queries'.format(n) for n in
    #                                                                                  algorithms_ptrs.keys()]
    # df = pd.DataFrame(columns=cols)
    #
    # for exp in experiemnts:
    #     row_result = run_one_comparison(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
    #                                     exp['h5_path'],
    #                                     exp['n_iterations'], algorithms_ptrs)
    #     exp_name = exp['h5_path'].split('.')[0].split('/')[-1] + '_' + str(exp['n_iterations'])
    #     df = df.append({cols[i]: ([exp_name] + row_result)[i] for i in range(len(row_result) + 1)}, ignore_index=True)
    #     print(df)
    #     pickle.dump(df, open("all_results.pkl", "wb"))
