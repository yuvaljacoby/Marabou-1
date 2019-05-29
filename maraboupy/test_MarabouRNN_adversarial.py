from maraboupy.MarabouRNN import *
from maraboupy import MarabouCore


def relu(num):
    return max(0, num)


def adversarial_robustness_sum_results(x):
    '''
    Get a list of inputs and calculate A,B according to the robustness network
    :param x:
    :return:
    '''

    s_i_1_f = 0
    z_i_1_f = 0

    for num in x:
        s_i_f = relu(2 * num + 1 * s_i_1_f)
        z_i_f = relu(1 * num + 1 * z_i_1_f)
        s_i_1_f = s_i_f
        z_i_1_f = z_i_f

    A = 2 * s_i_f  # + z_i_f
    B = 2 * z_i_f  # + s_i_f

    return A, B


def define_sum_adversarial_robustness(xlim, ylim, n_iterations):
    '''
    Defines the zero network in a marabou way
    The zero network is a network with two rnn cells, that always outputs zero
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network, will effect how we create the invariant
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(1)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    # s_i_f = relu(2 * x + s_i-1_f)
    s_cell_iterator = 1  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 2)], 1, n_iterations)
    # z_i_f = relu(x + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    network.setNumberOfVariables(a_idx + 2)  # +2 for A, B

    # A
    network.setLowerBound(a_idx, -large)
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)
    network.setUpperBound(b_idx, large)

    # i = i, we create iterator for each cell, make sure they are the same
    iterator_equation = MarabouCore.Equation()
    iterator_equation.addAddend(1, s_cell_iterator)
    iterator_equation.addAddend(-1, z_cell_iterator)
    iterator_equation.setScalar(0)
    network.addEquation(iterator_equation)

    # A = 2*skf <--> A - 2*skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-2, s_i_f_idx)
    a_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(a_output_eq)

    # B = 2*zkf <--> B - 2*z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-2, z_i_f_idx)
    b_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(b_output_eq)

    # s_i_f <= 2*z_i_f <--> s_i_f - 2*z_i_f <= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, s_i_f_idx)  # s_i f
    invariant_equation.addAddend(-2, z_i_f_idx)  # z_i f
    invariant_equation.setScalar(0)

    # z_i_b <= z_i_f <-- z_i_b - z_i_f <= 0
    # TODO: This is stupid we don't need this
    temp_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    temp_eq.addAddend(1, 7)  # s_i f
    temp_eq.addAddend(-1, 8)  # z_i f
    temp_eq.setScalar(0)
    network.addEquation(temp_eq)

    # A <= 2* B <--> A - 2*B <= 0
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, a_idx)
    property_eq.addAddend(-2, b_idx)
    property_eq.setScalar(0)

    return network, [s_cell_iterator, z_cell_iterator], invariant_equation, [property_eq]


def define_bias_sum_adversiral(xlim, ylim, n_iterations):
    '''
    A = 100 (0*x + 100)
    z_k_f = sum(x_i)
    B = 1 * z_k_f
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network, will effect how we create the invariant
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(2)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    # A
    a_idx = 1
    network.setLowerBound(a_idx, -large)
    network.setUpperBound(a_idx, large)

    # A = 0 * x + 100 <-- > A == 100
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.setScalar(100)
    # output_equation.dump()
    network.addEquation(a_output_eq)

    # s_i_f = relu(x + s_i-1_f)
    s_cell_iterator = 2  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    b_idx = s_i_f_idx + 1

    network.setNumberOfVariables(b_idx + 1)  #for B

    # B
    network.setLowerBound(b_idx, -large)
    network.setUpperBound(b_idx, large)

    # B = 1*zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-1, s_i_f_idx)
    b_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(b_output_eq)

    # b_i <= i <--> b_i - i <= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, b_idx)
    invariant_equation.addAddend(-1, s_cell_iterator)
    invariant_equation.setScalar(0)

    # B <= A <--> B- A <= 0
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, b_idx)
    property_eq.addAddend(-1, a_idx)
    property_eq.setScalar(0)

    return network, [s_cell_iterator], invariant_equation, [property_eq]

def test_sum_adversarial():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert prove_using_invariant(invariant_xlim, None, num_iterations, define_sum_adversarial_robustness)


def test_bias_sum_adversarial_fail():
    num_iterations = 105
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert not prove_using_invariant(invariant_xlim, None, num_iterations, define_bias_sum_adversiral)


def test_bias_sum_adversarial():
    num_iterations = 99
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert prove_using_invariant(invariant_xlim, None, num_iterations, define_bias_sum_adversiral)

if __name__ == "__main__":
    low = -1
    high = 1
    n = 3
    for n in range(1, 5):
        print('low ({}) after: {} iterations:'.format(low, n), adversarial_robustness_sum_results([low] * n))
        print('high ({}) after: {} iterations:'.format(high, n), adversarial_robustness_sum_results([high] * n))

    import numpy as np

    print(adversarial_robustness_sum_results(np.random.permutation([-1] * 100 + [1] * 100)))
