from maraboupy.MarabouRNN import *
from maraboupy import MarabouCore


# <editor-fold desc="negative sum network definition">

# Most simple RNN, only sums the positive inputs
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
#
# Parameters:
#   from - to
#   x1 - x(i-1): input
#   xi - x(3*i): (alternating between sib and sif)
#   xi:          sib
#   x(i+1):      sif
#   x(3i):       output
def define_negative_sum_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the positive_sum network
    :param ylim: ensure the output of the network is not more than ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''
    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_negative_sum_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the positive_sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 1)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= i - 1 <--> -(s_i-1 f) >= - i + 1 <--> i - (s_i-1 f) >= 1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(1, start_param)  # i
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(1)
    # query.addEquation(hidden_limit_eq)

    # negate the invariant we want to prove
    # not(s_1 b <= 1) <--> s_1 b  > 1  <--> s_1 b >= 1 + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.setScalar(1.2 + small)

    # not (s_i b <= i) <--> s_i b > i <--> s_i b -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(-1, start_param)  # i
    output_equation.setScalar(small)

    # s_i b <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.addAddend(-1, start_param)  # i
    invariant_equation.setScalar(0)

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_negative_sum_network(xlim):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    num_params_for_cell = 5

    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(num_params_for_cell)

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    positive_sum_rnn_query.setLowerBound(1, 0)
    positive_sum_rnn_query.setUpperBound(1, large)

    # s_i b
    positive_sum_rnn_query.setLowerBound(2, -large)
    positive_sum_rnn_query.setUpperBound(2, large)

    # s_i f
    positive_sum_rnn_query.setLowerBound(3, 0)
    positive_sum_rnn_query.setUpperBound(3, large)

    # y
    positive_sum_rnn_query.setLowerBound(4, -large)
    positive_sum_rnn_query.setUpperBound(4, large)

    # s_i b = -x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(-1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    # update_eq.dump()
    positive_sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(positive_sum_rnn_query, 2, 3)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 4)
    output_equation.addAddend(-1, 3)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    return positive_sum_rnn_query


# </editor-fold>

# <editor-fold desc="positive sum network definition">

# Most simple RNN, only sums the positive inputs
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
#
# Parameters:
#   from - to
#   x1 - x(i-1): input
#   xi - x(3*i): (alternating between sib and sif)
#   xi:          sib
#   x(i+1):      sif
#   x(3i):       output
def define_positive_sum_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the positive_sum network
    :param ylim: ensure the output of the network is not more than ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''

    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_positive_sum_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the positive_sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_i-1 f) <= i - 1 <--> i - (s_i-1 f) >= 1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.addAddend(1, start_param)  # i
    hidden_limit_eq.setScalar(1)
    # query.addEquation(hidden_limit_eq)

    # s_i f <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 3)  # s_i f
    invariant_equation.addAddend(-1, start_param)  # i
    invariant_equation.setScalar(0)

    # not (s_i f >= i) <--> s_i f < i <--> s_i f -i >= \epsilon
    output_equation = negate_equation(invariant_equation)

    # negate the invariant we want to prove
    # not(s_1 f <= 1) <--> s_1 f  > 1  <--> s_1 f >= 1 + \epsilon
    base_output_equation = negate_equation(invariant_equation)

    # set i == 1
    base_limit_eq = MarabouCore.Equation()
    base_limit_eq.addAddend(1, start_param)
    base_limit_eq.setScalar(1)

    # (s_0 f) = 0 (because i == 1)
    base_hidden_limit_eq = MarabouCore.Equation(hidden_limit_eq)

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation, base_limit_eq]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_positive_sum_network(xlim=(-1, 1)):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    num_params_for_cell = 5

    # Plus one is for the invariant proof, we will add a slack variable
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(num_params_for_cell)  # + extra_params)

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    positive_sum_rnn_query.setLowerBound(1, 0)
    positive_sum_rnn_query.setUpperBound(1, large)

    # s_i b
    positive_sum_rnn_query.setLowerBound(2, -large)
    positive_sum_rnn_query.setUpperBound(2, large)

    # s_i f
    positive_sum_rnn_query.setLowerBound(3, 0)
    positive_sum_rnn_query.setUpperBound(3, large)

    # y
    positive_sum_rnn_query.setLowerBound(4, -large)
    positive_sum_rnn_query.setUpperBound(4, large)

    # s_i b = x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    # update_eq.dump()
    positive_sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(positive_sum_rnn_query, 2, 3)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 4)
    output_equation.addAddend(-1, 3)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    return positive_sum_rnn_query

def define_positive_sum_network2(xlim, ylim, n_iterations):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    # num_params_for_cell = 5

    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1 # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1) #rnn_idx == s_i f
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
    invariant_equation.addAddend(1, rnn_start_idx) # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(0)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])
    property_eq = negate_equation(property_eq)

    return positive_sum_rnn_query, [rnn_start_idx], invariant_equation, [property_eq]
# </editor-fold>

# <editor-fold desc="zero network definition">


def define_zero_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the sum network
    :param ylim: ensure the output of the network is exactly ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''
    return None
    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y <= ylim <--> y >= ylim + \epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_zero_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the zero network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    return None
    xlim = (query.getLowerBound(0), query.getUpperBound(0))
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_0 f) = 0
    base_hidden_limit_eq_s = MarabouCore.Equation()
    base_hidden_limit_eq_s.addAddend(1, 1)
    base_hidden_limit_eq_s.setScalar(0)

    # (z_0 f) = 0
    base_hidden_limit_eq_z = MarabouCore.Equation()
    base_hidden_limit_eq_z.addAddend(1, 4)
    base_hidden_limit_eq_z.setScalar(0)

    # negate the invariant we want to prove
    # not(s_1 b - z_1 b == 0) <--> s_1 b - z_1 b  != 0 <--> s_1 b - z_1 b >= \epsilon OR s_1 b - z_1 b <= \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.addAddend(-1, 5)
    base_output_equation.setScalar(small)
    # TODO: add <= epsilon

    # TODO: adjust the rest of the function also for zero invariant
    # (s_i-1 f) - (z_i-1 f) <=  <--> (s_i-1 f) + (z_i-1 f) - i * xlim[1] <= -xlim[1]
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    hidden_limit_eq.addAddend(-xlim[1], start_param)  # i
    hidden_limit_eq.addAddend(1, 1)  # s_i-1 f
    hidden_limit_eq.addAddend(1, 4)  # z_i-1 f
    hidden_limit_eq.setScalar(-xlim[1])

    # not(s_i b + z_i b <= xlim[1] * i) <--> s_i b + z_i b  > xlim[1] * i <--> s_i b + z_i b - xlim[1] * i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(1, 5)  # z_i b
    output_equation.addAddend(-xlim[1], start_param)  # i
    output_equation.setScalar(-small)
    # TODO: Add also GE from 1 + small and somehow validate also that

    # s_i b + z_i b <= xlim[1] * i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.addAddend(1, 5)  # z_i b
    invariant_equation.addAddend(-xlim[1], start_param)  # i
    invariant_equation.setScalar(0)

    base_invariant_eq = [base_hidden_limit_eq_s, base_hidden_limit_eq_z, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_zero_network(xlim=(-1, 1)):
    '''
    Defines the sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the sum rnn network (without recurent)
    '''
    return None
    num_params_for_cell = 8

    sum_rnn_query = MarabouCore.InputQuery()
    sum_rnn_query.setNumberOfVariables(num_params_for_cell)

    # x
    sum_rnn_query.setLowerBound(0, xlim[0])
    sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    sum_rnn_query.setLowerBound(1, 0)
    sum_rnn_query.setUpperBound(1, large)

    # s_i b
    sum_rnn_query.setLowerBound(2, -large)
    sum_rnn_query.setUpperBound(2, large)

    # s_i f
    sum_rnn_query.setLowerBound(3, 0)
    sum_rnn_query.setUpperBound(3, large)

    # z_i-1 f
    sum_rnn_query.setLowerBound(4, 0)
    sum_rnn_query.setUpperBound(4, large)

    # z_i b
    sum_rnn_query.setLowerBound(5, -large)
    sum_rnn_query.setUpperBound(5, large)

    # z_i f
    sum_rnn_query.setLowerBound(6, 0)
    sum_rnn_query.setUpperBound(6, large)

    # y
    sum_rnn_query.setLowerBound(7, -large)
    sum_rnn_query.setUpperBound(7, large)

    # s_i b = x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(sum_rnn_query, 2, 3)

    # z_i b = -x_i + z_i-1 f
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(-1, 0)
    update_eq.addAddend(1, 4)
    update_eq.addAddend(-1, 5)
    update_eq.setScalar(0)
    sum_rnn_query.addEquation(update_eq)

    # z_i f = ReLu(z_i b)
    MarabouCore.addReluConstraint(sum_rnn_query, 5, 6)

    # - y + skf  + zkf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 3)
    output_equation.addAddend(1, 6)
    output_equation.addAddend(-1, 7)
    output_equation.setScalar(0)
    sum_rnn_query.addEquation(output_equation)

    return sum_rnn_query


# </editor-fold>

# <editor-fold desc="definition of network that returns last input">

# Most simple RNN, only sums the positive inputs
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
#
# Parameters:
#   from - to
#   x1 - x(i-1): input
#   xi - x(3*i): (alternating between sib and sif)
#   xi:          sib
#   x(i+1):      sif
#   x(3i):       output

def define_last_network(xlim, ylim, n_iterations):
    '''
    :param xlim: how to limit the input to the network
    :return: (network, [rnn output indices], invariant equation, output equation
    '''
    num_params_before_rnn = 1

    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(num_params_before_rnn)

    # x
    query.setLowerBound(0, xlim[0])
    query.setUpperBound(0, xlim[1])

    # rnn, the s_i = 0 * s_i-1 + x * 1
    rnn_idx = add_rnn_cell(query, [(0,1)], 0)
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
    property_eq = negate_equation(property_eq)

    return query, [rnn_idx], invariant_equation, [property_eq]


# </editor-fold>



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
    # s_i-1 f   0
    # s_i b     1
    # s_i f     2
    # i         3

    # s_i f <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i f
    invariant_equation.addAddend(-1, 3)  # i
    invariant_equation.setScalar(0)

    actual_base_eq, actual_step_eq = create_invariant_equations(None, 3, [0], invariant_equation)

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 0)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= i - 1 <--> (s_i-1 f) - i  <= -1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    hidden_limit_eq.addAddend(-1, 3)  # i
    hidden_limit_eq.addAddend(1, 0)  # s_i-1 f
    hidden_limit_eq.setScalar(-1)


    # set i == 1
    base_limit_eq = MarabouCore.Equation()
    base_limit_eq.addAddend(1, 3)
    base_limit_eq.setScalar(1)

    # negate the invariant we want to prove
    # not(s_1 f <= 1) <--> s_1 f  > 1  <--> s_1 f >= 1 + \epsilon
    # or we can do: i == 1 AND (not s_1 f <= i) <--> i == 1 AND s_1 f - i >= \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.addAddend(-1, 3)
    base_output_equation.setScalar(small)

    # not (s_i f >= i) <--> s_i f < i <--> s_i f -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i f
    output_equation.addAddend(-1, 3)  # i
    output_equation.setScalar(small)

    # base_output_equation = negate_equation(invariant_equation)
    # base_hidden_limit_eq = MarabouCore.Equation(hidden_limit_eq)


    # output_equation = negate_equation(invariant_equation)
    true_base = [base_hidden_limit_eq, base_output_equation, base_limit_eq]
    true_step = [hidden_limit_eq, output_equation]

    # [eq.dump() for eq in actual_base_eq]
    assert len(actual_base_eq) == len(true_base)
    assert len(actual_step_eq ) == len(true_step)



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

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network,
                                     define_negative_sum_invariant_equations,
                                     define_negative_sum_output_equations)


def test_negative_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network,
                                 define_negative_sum_invariant_equations,
                                 define_negative_sum_output_equations)
    # if pass_run:
    # else:
    #     invariant_xlim = (-1.1, -0.9)
    # print('negative sum result:',


def test_positive_sum_negative():
    num_iterations = 500
    invariant_xlim = (1, 1.1)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network,
                                     define_positive_sum_invariant_equations,
                                     define_positive_sum_output_equations)


def test_positive_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network,
                                 define_positive_sum_invariant_equations,
                                 define_positive_sum_output_equations)
    # if pass_run:
    # else:
    #     invariant_xlim = (-1.1, -0.9)
    # print('negative sum result:',


def test_positive_sum_negative2():
    num_iterations = 500
    invariant_xlim = (1, 1.1)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant_2(invariant_xlim, y_lim, num_iterations, define_positive_sum_network2)


def test_positive_sum_positive2():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations)

    assert prove_using_invariant_2(invariant_xlim, y_lim, num_iterations, define_positive_sum_network2)

def test_last_network_negative():
    num_iterations = 500
    invariant_xlim = (-1, 2)
    y_lim = (-1, 0)
    assert not prove_using_invariant_2(invariant_xlim, y_lim, num_iterations, define_last_network)


def test_last_network_positive():
    '''
    create wanted property and invariant that holds
    :return:
    '''
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = invariant_xlim
    assert prove_using_invariant_2(invariant_xlim, y_lim, num_iterations, define_last_network)


def test_simple_example():
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(4)

    # x
    query.setLowerBound(0, 1)
    query.setUpperBound(0, 1.2)

    # i
    query.setLowerBound(1, 0)
    query.setLowerBound(1, 5)

    # s_i-1 f
    query.setLowerBound(2, 0)
    query.setUpperBound(2, large)

    # s_i b
    query.setLowerBound(3, -large)
    query.setUpperBound(3, large)

    # s_i f
    # query.setLowerBound(4, 0)
    # query.setUpperBound(4, large)

    # s_i f = ReLu(s_i b)
    # MarabouCore.addReluConstraint(query, 3, 4)

    # s_i b = x * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 2)
    update_eq.addAddend(-1, 3)
    update_eq.setScalar(0)
    query.addEquation(update_eq)

    # s_i f <= i <--> i - s_i f >= 0
    # invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    # invariant_equation.addAddend(1, 1)  # i
    # invariant_equation.addAddend(-1, 4)  # s_i f
    # invariant_equation.setScalar(0)

    # base_equations, step_equations = create_invariant_equations(query, [1], invariant_equation)

    # induction_step = negate_equation(invariant_equation)
    # query.addEquation(induction_step)

    # s_i-1 f <= i-1 <--> s_i-1 f - i <= -1 <--> i - s_i-1 f >= 1
    # induction_hypothesis = MarabouCore.Equation(MarabouCore.Equation.GE)
    # induction_hypothesis.addAddend(1, 1)
    # induction_hypothesis.addAddend(-1, 2)
    # induction_hypothesis.setScalar(1)
    # query.addEquation(induction_hypothesis)

    # i == 1
    loop_eq = MarabouCore.Equation()
    loop_eq.addAddend(1,1)
    loop_eq.setScalar(1)
    query.addEquation(loop_eq)

    # s_i-1 f == 0
    # loop_eq2 = MarabouCore.Equation()
    # loop_eq2.addAddend(1, 2)
    # loop_eq2.setScalar(0)
    # query.addEquation(loop_eq2)

    print("Querying for induction base")
    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        assert True
    else:
        print("UNSAT")
        assert False

    # {0: 1.1, 1: 1.0, 2:0.0, 3:1.1, 4:1.1}