from maraboupy.MarabouRNN import *
from maraboupy import MarabouCore


def define_zero_network(xlim, ylim, n_iterations):
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

    s_cell_iterator = 1  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    y_idx = z_i_f_idx + 1

    network.setNumberOfVariables(y_idx + 1)

    # y
    network.setLowerBound(y_idx, -large)
    network.setUpperBound(y_idx, large)

    # i = i, we create iterator for each cell, make sure they are the same
    iterator_equation = MarabouCore.Equation()
    iterator_equation.addAddend(1, s_cell_iterator)
    iterator_equation.addAddend(-1, z_cell_iterator)
    iterator_equation.setScalar(0)
    network.addEquation(iterator_equation)

    # y = skf - zkf <--> y - skf + zkf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, s_i_f_idx)
    output_equation.addAddend(1, z_i_f_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    network.addEquation(output_equation)

    # s_i f - z_i f <= 0.01
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(-1, z_i_f_idx)  # s_i f
    invariant_equation.addAddend(1, s_i_f_idx)  # s_i f
    invariant_equation.setScalar(ylim / n_iterations)

    # y <= n * 0.01
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim)

    return network, [s_cell_iterator, z_cell_iterator], invariant_equation, [property_eq]


# Simple RNN, only sums the negative inputs
#   0   <= xi  <= 1
#   0   <= sif
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  for each i > 1
#       -xi + s(i-1)f - sib = 0
#  y - skf =0 # Where k == n_iterations
#
#  sif = Relu(sib)
def define_negative_sum_network(xlim, ylim, n_iterations):
    '''
    Defines the negative network in a marabou way
        s_i = ReLu(-1 * x_i + s_i-1)
        y = s_k (where k == n_iterations)
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, -1)], 1, n_iterations)  # rnn_idx == s_i f
    y_idx = rnn_idx + 1

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # s_i f <= i <--> i - s_i f >= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_equation.addAddend(1, rnn_start_idx)  # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(0)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return positive_sum_rnn_query, [rnn_start_idx], invariant_equation, [property_eq]


# Simple RNN, only sums the positive inputs
#   0   <= xi  <= 1
#   0   <= sif
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  for each i > 1
#       xi + s(i-1)f - sib = 0
#  y - skf =0 # Where k == n_iterations
#
#  sif = Relu(sib)
def define_positive_sum_network(xlim, ylim, n_iterations):
    '''
    Defines the positive_sum network in a marabou way
        s_i = ReLu(1 * x_i + 1 * s_i-1)
        y = s_k (where k == n_iterations)
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    # num_params_for_cell = 5

    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1, n_iterations)  # rnn_idx == s_i f
    y_idx = rnn_idx + 1

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # s_i f <= i <--> i - s_i f >= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_equation.addAddend(1, rnn_start_idx)  # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(0)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return positive_sum_rnn_query, [rnn_start_idx], invariant_equation, [property_eq]


def define_last_network(xlim, ylim, n_iterations):
    '''
    Function that define "last_network" which is an RNN network that outputs the last input parameter
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: (network, [rnn output indices], invariant equation, output equation
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(1)

    # x
    query.setLowerBound(0, xlim[0])
    query.setUpperBound(0, xlim[1])

    # rnn, the s_i = 0 * s_i-1 + x * 1
    rnn_idx = add_rnn_cell(query, [(0, 1)], 0, n_iterations)
    y_idx = rnn_idx + 1

    query.setNumberOfVariables(y_idx + 1)
    # y
    query.setLowerBound(y_idx, -large)
    query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    query.addEquation(output_equation)

    # s_i-1 f <= xlim[1]
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, rnn_idx - 2)  # s_i f
    invariant_equation.setScalar(xlim[1])

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return query, [rnn_idx], invariant_equation, [property_eq]


def test_negate_equation_GE():
    # x - y >= 0
    eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    eq.addAddend(1, 1)
    eq.addAddend(-1, 0)  # i
    eq.setScalar(0)

    # x - y <= -epsilon
    not_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    not_eq.addAddend(1, 1)  # s_i b
    not_eq.addAddend(-1, 0)  # i
    not_eq.setScalar(-small)
    actual_not_eq = negate_equation(eq)

    assert actual_not_eq.equivalent(not_eq)
    assert not eq.equivalent(not_eq)


def test_negate_equation_LE():
    # x + y <= 1
    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    eq.addAddend(1, 0)
    eq.addAddend(1, 1)
    eq.setScalar(1)

    # x + y >= 1 + epsilon
    not_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    not_eq.addAddend(1, 0)
    not_eq.addAddend(1, 1)
    not_eq.setScalar(1 + small)
    actual_not_eq = negate_equation(eq)

    assert actual_not_eq.equivalent(not_eq)
    assert not eq.equivalent(not_eq)


def test_create_invariant_equations_sum():
    # i         0
    # s_i-1 f   1
    # s_i b     2
    # s_i f     3

    # s_i f <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 3)  # s_i f
    invariant_equation.addAddend(-1, 0)  # i
    invariant_equation.setScalar(0)

    actual_base_eq, actual_step_eq = create_invariant_equations([0], invariant_equation)

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 1)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= i - 1 <--> (s_i-1 f) - i  <= -1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    hidden_limit_eq.addAddend(-1, 0)  # i
    hidden_limit_eq.addAddend(1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(-1)

    # set i == 1
    base_limit_eq = MarabouCore.Equation()
    base_limit_eq.addAddend(1, 0)
    base_limit_eq.setScalar(1)

    # set s_i-1 f == 0
    hidden_limit_base_eq = MarabouCore.Equation()
    hidden_limit_base_eq.addAddend(1, 1)
    hidden_limit_base_eq.setScalar(0)

    # negate the invariant we want to prove
    # not(s_1 f <= 1) <--> s_1 f  > 1  <--> s_1 f >= 1 + \epsilon
    # or we can do: i == 1 AND (not s_1 f <= i) <--> i == 1 AND s_1 f - i >= \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 3)
    base_output_equation.addAddend(-1, 0)
    base_output_equation.setScalar(small)

    # not (s_i f >= i) <--> s_i f < i <--> s_i f -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 3)  # s_i f
    output_equation.addAddend(-1, 0)  # i
    output_equation.setScalar(small)

    true_base = [base_hidden_limit_eq, base_output_equation, base_limit_eq, hidden_limit_base_eq]
    true_step = [hidden_limit_eq, output_equation]

    # assert False
    assert len(actual_base_eq) == len(true_base)
    assert len(actual_step_eq) == len(true_step)

    for true_eq in true_base:
        found = False
        for eq in actual_base_eq:
            if true_eq.equivalent(eq):
                found = True
                break
        if not found:
            assert False, "didn't find equation for (in base) {}".format(true_eq.dump())

    for true_eq in true_step:
        found = False
        for eq in actual_step_eq:
            if true_eq.equivalent(eq):
                found = True
                continue
        if not found:
            assert False, "didn't find equation for (in step) {}".format(true_eq.dump())


def test_negative_sum_negative():
    num_iterations = 500
    invariant_xlim = (-1.1, -0.9)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network)


def test_negative_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network)


def test_positive_sum_negative():
    num_iterations = 500
    invariant_xlim = (1, 1.1)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network)


def test_positive_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network)


def test_last_network_negative():
    num_iterations = 500
    invariant_xlim = (-1, 2)
    y_lim = (-1, 0)
    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_last_network)


def test_last_network_positive():
    '''
    create wanted property and invariant that holds
    :return:
    '''
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = invariant_xlim
    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_last_network)


def test_zero_network_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = 10 ** -2
    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_zero_network)
