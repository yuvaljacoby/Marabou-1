import torch

from maraboupy import MarabouCore
from maraboupy.MarabouRNNMultiDim import add_rnn_multidim_cells, prove_multidim_property

MODEL_FILE_PATH = 'mnist_example/rnn.pt'
PI = 360
large = 5000


class MnistModel():
    def __init__(self, model_file_path):
        self.state_dict = torch.load(model_file_path)

    def get_rnn_weights(self):
        '''
        rnn_input_weights.shape = [64, 112]
        rnn_bias_weights.shape = [64,]
        rnn_hidden_weights.shape = [64,64]
        :return: rnn_input_weights, rnn_hidden_weights, rnn_bias_weights
        '''
        rnn_input_weights = self.state_dict['rnn.weight_ih_l0'].numpy()
        rnn_hidden_weights = self.state_dict['rnn.weight_hh_l0'].numpy()
        # pytorch saves two biases but it's not more informative then having just one
        rnn_bias = self.state_dict['rnn.bias_ih_l0'].numpy() + self.state_dict['rnn.bias_hh_l0'].numpy()
        return rnn_input_weights, rnn_hidden_weights, rnn_bias

    def get_output_weights(self):
        '''
        output_weights.shape = [16,2]
        output_bias_weights.shape = [2,]
        :return: output_weights, output_bias_weights
        '''
        output_weights = self.state_dict['out.weight'].numpy()
        output_bias_weights = self.state_dict['out.bias'].numpy()
        return output_weights, output_bias_weights


def set_img_bounds(img, network, pertubation_limit=0.05):
    '''

    :param img: ndarray with shape 128,  this is the input image
    :param network: marabou query that will add to it the limit on the input image
    :param pertubation_limit: how much pertubation is allwod in percepntage to the current value
    :return: int how many variables added to the network
    '''
    last_idx = network.getNumberOfVariables()
    network.setNumberOfVariables(last_idx + img.shape[-1])

    for i in range(img.shape[-1]):
        network.setLowerBound(last_idx + i, img[i] * (1 - pertubation_limit))
        network.setUpperBound(last_idx + i, img[i] * (1 + pertubation_limit))

    return img.shape[-1]


def add_rnn_cells(network, rnn_input_weights, rnn_hidden_weights, rnn_bias, num_iterations=10):
    '''
    adding n cells to the network, where n is the size of the hidden weights score matrix
    :param network: marabou query to add the cells
    :param rnn_input_weights: ndarray, shape n on m where m is the input size
    :param rnn_hidden_weights: square matrix, size is n on n
    :param rnn_bias: 1 dim array length n with the bias to the rnn cells
    :return: list of the output indcies for each of the cells
    '''
    assert type(rnn_input_weights) == np.ndarray
    assert type(rnn_hidden_weights) == np.ndarray and len(rnn_hidden_weights.shape) == 2
    assert rnn_hidden_weights.shape[0] == rnn_hidden_weights.shape[1]
    assert rnn_hidden_weights.shape[0] == rnn_input_weights.shape[0]
    assert len(rnn_bias) == rnn_hidden_weights.shape[0]

    var_weight_list = []

    # We assume here that the input variables index is 0 to rnn_input_weights.shape[0]
    # for i in range(rnn_input_weights.shape[0]):  # 64
    #     var_weight_list.append([(i, w) for w in rnn_input_weights[i, :]])
    input_indices = list(range(network.getNumberOfVariables()))
    rnn_output_idx = add_rnn_multidim_cells(network, input_indices, rnn_input_weights, rnn_hidden_weights, rnn_bias,
                                            num_iterations)
    network.dump()
    return rnn_output_idx


def add_output_equations(network, rnn_output_idxs, output_weight, output_bias):
    '''
    build equations for the output
    :param network: network to append equations and variables to
    :param rnn_output_idxs: the output indices of the previous layer
    :param output_weight: Weights to multiply the previous layer
    :param output_bias: The bias of each equation
    :return: list of indices of output classes
    '''
    assert (len(rnn_output_idxs) == output_weight.shape[1])
    assert (output_weight.shape[0] == len(output_bias))
    last_idx = network.getNumberOfVariables()
    output_idxs = []
    network.setNumberOfVariables(last_idx + (output_weight.shape[0] * 2))  # *2 because of the relu
    for i in range(output_weight.shape[0]):
        b_variable_idx = last_idx + (2 * i)
        f_variable_idx = last_idx + 1 + (2 * i)
        output_idxs.append(f_variable_idx)

        network.setLowerBound(b_variable_idx, -large)
        network.setUpperBound(b_variable_idx, large)
        network.setLowerBound(f_variable_idx, 0)
        network.setUpperBound(f_variable_idx, large)
        MarabouCore.addReluConstraint(network, b_variable_idx, f_variable_idx)

        output_eq = MarabouCore.Equation()
        for j in range(output_weight.shape[1]):
            output_eq.addAddend(output_weight[i, j], rnn_output_idxs[j])

        output_eq.addAddend(-1, b_variable_idx)
        output_eq.setScalar(-output_bias[i])
        network.addEquation(output_eq)

    return output_idxs


def build_query(img_example: list, ylim: list, model: MnistModel):
    '''

    :param xlim: list of tuples, each cell is for input variable, and the tuple is (min_val, max_val)
    :param ylim: list of tuples, each cell is for an output variable, and the tuple is (min_val, max_val)
    :param pendulum_model: initialized model with get_rnn_weights and get_output_weights
    :return:
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(0)
    last_idx = set_img_bounds(img_example, network)

    rnn_input_weights, rnn_hidden_weights, rnn_bias = model.get_rnn_weights()
    rnn_output_idx = add_rnn_cells(network, rnn_input_weights, rnn_bias, rnn_hidden_weights)
    output_weights, output_bias_weights = model.get_output_weights()
    add_output_equations(network, rnn_output_idx, output_weights, output_bias_weights)
    return network


def get_max_value(input_bounds, w_in, w_h, b_h, w_out, b_out, n):
    '''
    calculate the max value of a network with one input node, 2d rnn cell, and a number
    :param input_bounds: tuple (min_val, max_val)
    :param w_in: weights on the input to the rnn node array length 2
    :param w_h: hidden matrix, 2x2
    :param b_h: bias for rnn cell array length 2
    :param w_out: weight from the rnn cell to the output array length 2
    :param b_out: bias from the rnn cell to the output length 2
    :param n: number of rounds
    :return: max value of the
    '''
    r0 = 0
    r1 = 1
    for i in range(n):
        r0_new = input_bounds[1] * w_in[0] + r0 * w_h[0,0] + r1 * w_h[0,1] + b_h[0]
        r1_new = input_bounds[1] * w_in[1] + r0 * w_h[1, 0] + r1 * w_h[1, 1] + b_h[1]
        r0 = r0_new
        r1 = r1_new
    y = r0 * w_out[0] + r1 * w_out[1] + b_out
    return y

if __name__ == "__main__":
    import numpy as np

    n = 10
    np.random.seed(0)
    img = np.array([1])  # img = np.array([1, 1])  # np.random.random(2)

    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(0)

    pertubation_limit = 0#.1
    set_img_bounds(img, network, pertubation_limit)
    # network.setNumberOfVariables(2)
    # network.setLowerBound(0, -large)
    # network.setUpperBound(0, large)
    # network.setLowerBound(1, -large)
    # network.setUpperBound(1, large)

    w_in_0 = [1] # w_in_0 = [1, 1]
    w_in_1 = [1] # w_in_1 = [1, 1]

    w_h_0 = [1, 1]
    w_h_1 = [0, 0]
    b_h = [0, 0]
    w_out_0 = [1]
    w_out_1 = [1]


    # w_in_0 = [0.1, 0.2]
    # w_in_1 = [0.3, 0.4]
    # w_h_0 = [0.5, 0.7]
    # w_h_1 = [0.2, 0.3]
    # w_out_0 = [0.2]
    # w_out_1 = [1.2]

    def ReLU(x):
        return max(x, 0)

    x_min = img[0] * (1 - pertubation_limit)
    x_max = img[0] * (1 + pertubation_limit)
    R0_min = ReLU(x_min * w_in_0[0])  #+ img[1] * (1 - pertubation_limit) * w_in_0[1])
    R1_min = ReLU(x_min * w_in_1[0])  #+ img[1] * (1 - pertubation_limit) * w_in_1[1])
    R0_max = ReLU(x_max * w_in_0[0])  #+ img[1] * (1 + pertubation_limit) * w_in_0[1])
    R1_max = ReLU(x_max * w_in_1[0])  #+ img[1] * (1 + pertubation_limit) * w_in_1[1])
    initial_values = [R0_min,R0_max, R1_min, R1_max]

    # initial_values = [4, 6]
    w_in = np.array([w_in_0, w_in_1])
    w_h = np.array([w_h_0, w_h_1])

    rnn_output_idxs = add_rnn_cells(network, w_in, w_h, b_h, n)
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]

    w_out = [1, 0]
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(w_out[0], rnn_output_idxs[0])
    property_eq.setScalar(23)

    from maraboupy.draw_rnn import draw_r_values,calc_rnn_values
    # draw_r_values(*calc_rnn_values(x_max, w_in, w_h, n))
    print("network max value:", get_max_value([x_min, x_max], w_in, w_h, b_h, w_out, 0, n))
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, [property_eq])
    print('property proved')
    #
    # # print("rnn_start_idxs:", rnn_start_idxs)
    # output_idx = add_output_equations(network, rnn_output_idxs, np.array([w_out_0, w_out_1]).T, np.array([0.3]))
    # # network.dump()
    # # print("output idx:", output_idx)
    #
    # # Let's verify that the output node is less then 100, after 10 iterations
    # n = 10
    # property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    # property_eq.addAddend(1, output_idx[0])
    # property_eq.setScalar(100)
    #
    # # network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations,
    # # property_equations, min_alphas = None, max_alphas = None, rnn_dependent = None
    #
    # find_invariant_marabou(network, rnn_start_idxs, [MarabouCore.Equation.LE] * len(rnn_start_idxs), initial_values,  n, [property_eq])

    # model = MnistModel(MODEL_FILE_PATH)
    # build_query(np.random.random((128)), None, model)
