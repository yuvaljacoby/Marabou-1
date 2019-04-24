import time

from z3 import *
from maraboupy import MarabouCore

# Most simple RNN, only sums inputs
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


large = 1000.0
small = 10 ** -2


def add_rnn_cell_bounds(inputQuery, n_iterations, i, large):
    '''
    add constraints for rnn hidden vector (unfolded)
    for each hidden vector add b and f constraint
    the constraint are for each hidden vector i:
        constraint i: between -large to large (unbounded)
        constraint i + 1: between 0 to large ReLu result
    :param inputQuery query to add the bounds too
    :param n_iterations: number of hidden vectors (will add two for each constraint)
    :param i: start index
    :param large: big number
    :return: update i
    '''
    for _ in range(n_iterations):
        # sib
        inputQuery.setLowerBound(i, -large)
        inputQuery.setUpperBound(i, large)

        # sif
        inputQuery.setLowerBound(i + 1, 0)
        inputQuery.setUpperBound(i + 1, large)
        i += 2
    return i


def add_hidden_state_equations(inputQuery, variables_first_index, input_weight, hidden_weight, n_iterations):
    '''
    add all hidden state equations:
        input_weight * x1 = s1b
        for each k > 1
            input_weight * xi + hidden_weight * s(k-1)f = sib
        and ReLu's
    :param inputQuery: query to append to
    :param variables_first_index: the first index of the hidden vector variable
    :param input_weight: the weight in the input
    :param hidden_weight: the weight for the hidden vector
    :param n_iterations: number of iterations
    :return:
    '''
    equation1 = MarabouCore.Equation()
    equation1.addAddend(input_weight, 0)
    equation1.addAddend(-1, variables_first_index)
    equation1.setScalar(0)
    inputQuery.addEquation(equation1)

    for k in range(1, n_iterations):
        cur_equation = MarabouCore.Equation()
        cur_equation.addAddend(input_weight, k)  # xk
        cur_equation.addAddend(hidden_weight, variables_first_index + (2 * k) - 1)  # s(k-1)f
        cur_equation.addAddend(-1, variables_first_index + (2 * k))  # skb
        cur_equation.setScalar(0)
        inputQuery.addEquation(cur_equation)

    # ReLu's
    for k in range(variables_first_index, variables_first_index + 2 * n_iterations, 2):
        MarabouCore.addReluConstraint(inputQuery, k, k + 1)


def unfold_sum_rnn(n_iterations, xlim=(-1, 1), ylim=(-1, 1)):
    i = 0  # index for variable number
    inputQuery = MarabouCore.InputQuery()

    num_variables = n_iterations  # the x input
    s_first_index = num_variables
    num_variables += n_iterations * 2  # for each temporal state (2 because of the ReLu)
    y_index = num_variables
    num_variables += 1  # for y

    inputQuery.setNumberOfVariables(num_variables)

    for _ in range(n_iterations):
        inputQuery.setLowerBound(i, xlim[0])
        inputQuery.setUpperBound(i, xlim[1])
        i += 1

    add_rnn_cell_bounds(inputQuery, n_iterations, s_first_index, large)  # add s_i

    # output
    inputQuery.setLowerBound(y_index, ylim[0])
    inputQuery.setUpperBound(y_index, ylim[1])

    add_hidden_state_equations(inputQuery, s_first_index, 1, 1, n_iterations)

    # y - skf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_index)
    output_equation.addAddend(-1, y_index - 1)
    output_equation.setScalar(0)
    inputQuery.addEquation(output_equation)

    vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
    else:
        print("UNSAT")


# def prove_invariant_for_loop(n_iterations, input_weight=1, hidden_weight=1, xlim=(-1, 1)):
#     # check for each s_i if for any s_i-1 might by that s_i > i
#     for i in range(n_iterations):
#         inputQuery = MarabouCore.InputQuery()
#         inputQuery.setNumberOfVariables(3)
#
#         # x
#         inputQuery.setLowerBound(0, xlim[0])
#         inputQuery.setUpperBound(0, xlim[1])
#
#         # s_i-1_f
#         # times i-1 because of the previous loop
#         inputQuery.setLowerBound(1, xlim[0] * (i - 1))
#         inputQuery.setUpperBound(1, xlim[1] * (i - 1))
#
#         # s_i b
#         inputQuery.setLowerBound(2, -large)
#         inputQuery.setUpperBound(2, large)
#
#         # s_i b = s_i-1 f * a2 + x_i * a1
#         hidden_equation = MarabouCore.Equation()
#         hidden_equation.addAddend(input_weight, 0)
#         hidden_equation.addAddend(hidden_weight, 1)
#         hidden_equation.addAddend(-1, 2)
#         hidden_equation.setScalar(0)
#         inputQuery.addEquation(hidden_equation)
#
#         # not (s_i b >= i)
#         output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
#         output_equation.addAddend(1, 2)
#         # output_equation.addAddend(-1, )
#         output_equation.setScalar(i + small)
#         output_equation.dump()
#         inputQuery.addEquation(output_equation)
#
#         # invariant_equation = MarabouCore.Equation()
#         # invariant_equation.addAddend(1,0)
#
#         vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
#         if len(vars1) > 0:
#             print("SAT")
#             print(vars1)
#             return False
#         else:
#             print("UNSAT")
#     return True

def marabou_solve_negate_eq(query):
    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
        return True


def define_sum_rnn_cell(xlim=(-1, 1)):
    '''
    define the rnn cell for sum function i.e.:
    s_i b = x_i * 1 + s_i-1 f * 1
    :param xlim: limit on the input for the cell (in order to define the param)
    :return: tuple (query, num_params used)
    '''

    # s_i b = x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    rnn_cell_query.addEquation(update_eq)


def define_sum_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: the query until now, will append to this
    :return: tuple (list of base equations, list of step equations, #of added param)
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

    # (s_i-1 f) <= i - 1 <--> i - (s_i-1 f) >= 1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(1, start_param)  # i
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(1)
    # query.addEquation(hidden_limit_eq)

    # negate the invariant we want to prove
    # not(s_1 b <= 1) <--> s_1 b  > 1  <--> s_1 b >= 1 + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.setScalar(1 + small)

    # not (s_i b >= i) <--> s_i b < i <--> s_i b -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(-1, start_param)  # i
    # TODO: Check if it has to be 1, something with i being float number and not int
    output_equation.setScalar(small)

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation]
    invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, invariant_eq, 1)


def define_sum_network(xlim=(-1, 1)):
    num_params_for_cell = 5

    # Plus one is for the invariant proof, we will add a slack variable
    rnn_cell_query = MarabouCore.InputQuery()
    rnn_cell_query.setNumberOfVariables(num_params_for_cell)  # + extra_params)

    # x
    rnn_cell_query.setLowerBound(0, xlim[0])
    rnn_cell_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    rnn_cell_query.setLowerBound(1, 0)
    rnn_cell_query.setUpperBound(1, large)

    # s_i b
    rnn_cell_query.setLowerBound(2, -large)
    rnn_cell_query.setUpperBound(2, large)

    # s_i f
    rnn_cell_query.setLowerBound(3, 0)
    rnn_cell_query.setUpperBound(3, large)

    # y
    rnn_cell_query.setLowerBound(4, -large)
    rnn_cell_query.setUpperBound(4, large)

    invariant_num_params = 1
    rnn_cell_query, num_params_for_cell = define_sum_rnn_cell(invariant_num_params, xlim)
    base_invariant_equations, step_invariant_equations, _ = define_sum_invariant_equations(rnn_cell_query)

    return rnn_cell_query, base_invariant_equations, step_invariant_equations


# def prove_induction_base(input_weight=1, hidden_weight=1, xlim=(-1, 1), invariant_lim=1):
#     s_i_max = 0 * hidden_weight + input_weight * xlim[1]
#     # s_i_min = 0 * hidden_weight + input_weight * xlim[1]
#
#     return s_i_max <= invariant_lim
#
# def prove_induction_step(n_iterations, input_weight, hidden_weight, xlim, invariant_lim):
#     # TODO: use invariant_lim, need it in the last equation we add, know we use s_i - i >= 1
#     # TODO: but it needs to be something like s_i - invariant_lim(i) >= 1
#     inputQuery = MarabouCore.InputQuery()
#     inputQuery.setNumberOfVariables(4)
#
#     # x
#     inputQuery.setLowerBound(0, xlim[0])
#     inputQuery.setUpperBound(0, xlim[1])
#
#     # temp
#     inputQuery.setLowerBound(1, 0)
#     inputQuery.setUpperBound(1, large)
#
#     # i, slack variable to help with limiting temp
#     # TODO: do we always need it or it's part of the invariant definition?
#     inputQuery.setLowerBound(2, 0)
#     inputQuery.setUpperBound(2, large)
#
#     # s_i b
#     inputQuery.setLowerBound(3, -large)
#     inputQuery.setUpperBound(3, large)
#
#     # s_i b = x_i * a1 + t * a2
#     update_eq = MarabouCore.Equation()
#     update_eq.addAddend(input_weight, 0)
#     update_eq.addAddend(hidden_weight, 1)
#     update_eq.addAddend(-1, 3)
#     update_eq.setScalar(0)
#     inputQuery.addEquation(update_eq)
#
#     ## TODO: Get from outside, part of the invariant
#     # temp < i <--> i - temp > \epsilon
#     hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
#     hidden_limit_eq.addAddend(-1, 1)
#     hidden_limit_eq.addAddend(1, 2)
#     hidden_limit_eq.setScalar(small)
#     inputQuery.addEquation(hidden_limit_eq)
#
#     ## TODO: Get from outside, part of the invariant
#     # negate the invariant we want to prove
#     # not (s_i b >= i) <--> s_i b -i >= \epsilon
#     output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
#     output_equation.addAddend(1, 3)
#     output_equation.addAddend(-1, 2)
#     # TODO: Check if it has to be 1, something with i being float number and not int
#     output_equation.setScalar(1)
#     # output_equation.dump()
#     inputQuery.addEquation(output_equation)
#
#     vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
#     if len(vars1) > 0:
#         print("SAT")
#         print(vars1)
#         return False
#     else:
#         print("UNSAT")
#         return True


def prove_invariant(f=define_sum_network, xlim=(-1,1)):
    '''
    proving invariant on a given rnn cell
    :param n_iterations: max number of times to run the cell
    :param input_weight: The weight for the input (before the cell)
    :param hidden_weight: The weight inside the cell
    :param xlim: limits on the input
    :param invariant_lim: what to prove (that the output of the network is smaller than some function of i)
    :return: True of the invariant holds, false otherwise
    '''

    rnn_cell_query, base_invariant_equations, step_invariant_equations = f(xlim)


    for eq in base_invariant_equations:
        rnn_cell_query.addEquation(eq)

    if not marabou_solve_negate_eq(rnn_cell_query):
        print("induction base fail")
        return False

    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    rnn_cell_query, base_invariant_equations, step_invariant_equations = f(xlim)
    # There is one equation we want to save in rnn_cell_query, and len(base_invariant_equations) we want to remove)
    # for i in range(len(base_invariant_equations)):
    #     rnn_cell_query.removeEquationsByIndex(1 + i)

    for eq in step_invariant_equations:
        rnn_cell_query.addEquation(eq)
    return marabou_solve_negate_eq(rnn_cell_query)

# def prove_invariant(n_iterations, input_weight=1, hidden_weight=1, xlim=(-1, 1), invariant_lim=lambda i: i):
#     '''
#     proving invariant on a given rnn cell
#     :param n_iterations: max number of times to run the cell
#     :param input_weight: The weight for the input (before the cell)
#     :param hidden_weight: The weight inside the cell
#     :param xlim: limits on the input
#     :param invariant_lim: what to prove (that the output of the network is smaller than some function of i)
#     :return: True of the invariant holds, false otherwise
#     '''
#
#     if not prove_induction_base(input_weight, hidden_weight, xlim, invariant_lim(1)):
#         print("induction base fail")
#         return False
#
#     return prove_induction_step(n_iterations, input_weight, hidden_weight, xlim, invariant_lim)

def define_sum_property(output_lim=11):
    '''
    create a marabou query to validate the wanted property
    :param output_lim: the max output that you wish
    :return: the query
    '''
    inputQuery = MarabouCore.InputQuery()

    num_variables = 3  # y, s_k b, s_k f

    inputQuery.setNumberOfVariables(num_variables)

    # s_k b
    inputQuery.setLowerBound(0, -large)
    inputQuery.setUpperBound(0, large)

    # s_k f
    inputQuery.setLowerBound(1, 0)
    inputQuery.setUpperBound(1, large)

    # output
    inputQuery.setLowerBound(2, -large)
    inputQuery.setUpperBound(2, large)

    # Last ReLu
    MarabouCore.addReluConstraint(inputQuery, 0, 1)

    # y - skf * output_weight = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 2)
    output_equation.addAddend(-1, 1)
    output_equation.setScalar(0)
    inputQuery.addEquation(output_equation)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e. we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 2)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(output_lim + small)
    inputQuery.addEquation(property_eq)

    return inputQuery

def prove_property_z3(invariant_property=10, weight=1, ylim=11):
    '''
    Using z3 to probe the formula
    checking ReLu(sk * w) <= ylim[1] while sk <= sklim
    :param invariant_property: maximum value for sk
    :param weight: the weight between sk and the output
    :param ylim: max output
    :return: True if for every sk <= sklim implies that ReLu(sk * w) <= ylim
    '''

    sk = Real('sk')
    w = Real('w')
    sk_ReLU = If(sk * w >= 0, sk * w, 0)

    s = Solver()
    s.add(w == weight)
    s.add(sk_ReLU <= invariant_property)
    # we negate the condition, insted if for all sk condition we check if there exists sk not condition
    s.add(sk_ReLU * w > ylim)

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def prove_property_marabou(property_f=define_sum_property, invariant_property=10, weight=1, ylim=11):
    '''

    :param property_f:
    :param invariant_property:
    :param weight:
    :param ylim:
    :return:
    '''
    property_query = property_f(ylim)

    # s_k b <= invariant_property from the invariant
    invariant_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_eq.addAddend(1, 0)
    invariant_eq.setScalar(invariant_property)
    # inputQuery.addEquation(invariant_eq)


def prove_using_invariant(n_iterations, input_weight=1, hidden_weight=1, output_weight=1, xlim=(-1, 1), ylim=10,
                          invariant_lim=lambda i: i, use_z3=False):
    if ylim < n_iterations:
        print("you might want to change ylim?\n")
        time.sleep(2)

    # TODO: change func definition to something like this: weights, rnn_cells, invariant, property
    # if not prove_invariant(n_iterations, input_weight, hidden_weight, xlim, invariant_lim):
    if not prove_invariant(define_sum_network, xlim):
        print("invariant doesn't hold")
        return False

    if use_z3:
        return prove_property_z3(invariant_lim(n_iterations), output_weight, ylim)
    else:
        return prove_property_marabou(invariant_lim(n_iterations), output_weight, ylim)


if __name__ == "__main__":
    num_iterations = 500
    pass_run = False
    if pass_run:
        invariant_xlim = (1, 1)
    else:
        invariant_xlim = (1, 1.1)
    print('result:',
          prove_using_invariant(num_iterations, input_weight=1, hidden_weight=1, output_weight=1, xlim=invariant_xlim,
                                ylim=num_iterations, invariant_lim=lambda i: i, use_z3=True))
    # print("invariant: ", prove_invariant(n_iterations, xlim=invariant_xlim))
    # start_invariant = time.time()
    # print("invariant: ", prove_invariant(n_iterations, xlim=invariant_xlim))
    # print("sk < 10 --> y < 11:", prove_property_z3(sklim=10, weight=1, ylim=11))
    # end_invariant = time.time()
    # print("invariant + z3 took:", end_invariant - start_invariant)
    # # print("sk < 10 --> y < 4 :", prove_property_z3(sklim=10, weight=1, ylim=4))
    #
    #
    # xlim = (-1, 1)
    # if make_sat:
    #     ylim = (xlim[0] * n_iterations, xlim[1] * n_iterations)
    # else:
    #     ylim = (xlim[1] * n_iterations + 0.1, xlim[1] * n_iterations + 1)
    #
    # start = time.time()
    # unfold_sum_rnn(n_iterations, xlim, ylim)
    # end = time.time()
    # print("unfold took:", end - start)
    # print("invariant took:", end_invariant - start_invariant)
