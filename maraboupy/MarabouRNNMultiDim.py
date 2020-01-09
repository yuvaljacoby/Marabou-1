import numpy as np

from maraboupy import MarabouCore
from maraboupy.keras_to_marabou_rnn import RnnMarabouModel
large = 5000.0
small = 10 ** -4
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 5


def marabou_solve_negate_eq(query, debug=False, print_vars=False):
    '''
    Run marabou solver
    :param query: query to execute
    :param debug: if True printing all of the query equations
    :return: True if UNSAT (no valid assignment), False otherwise
    '''
    verbose = 0
    if debug:
        query.dump()
    vars1, stats1 = MarabouCore.solve(query, "", 0, verbose)
    if len(vars1) > 0:
        if print_vars:
            print("SAT")
            print(vars1)
        return False
    else:
        # print("UNSAT")
        return True


def negate_equation(eq):
    '''
    negates the equation
    :param eq: equation
    :return: new equation which is exactly (not eq)
    '''
    not_eq = MarabouCore.Equation(eq)
    if eq.getType() == MarabouCore.Equation.GE:
        not_eq.setType(MarabouCore.Equation.LE)
        not_eq.setScalar(eq.getScalar() - small)
    elif eq.getType() == MarabouCore.Equation.LE:
        not_eq.setType(MarabouCore.Equation.GE)
        not_eq.setScalar(eq.getScalar() + small)
    elif eq.setType(MarabouCore.Equation.EQ):
        raise NotImplementedError("can't negate equal equations")
    else:
        raise NotImplementedError("got {} type which is not implemented".format(eq.getType()))
    return not_eq


def add_rnn_multidim_cells(query, input_idx, input_weights, hidden_weights, bias, num_iterations, print_debug=False):
    '''
    Create n rnn cells, where n is hidden_weights.shape[0] == hidden_weights.shape[1] ==  len(bias)
    The added parameters are (same order): i, s_i-1 f, s_i b, s_i f for each of the n added cells (i.e. adding 4*n variables)
    :param query: the network so far (will add to this)
    :param input_idx: list of input id's, length m
    :param input_weights: matrix of input weights, size m x n
    :param hidden_weights: matrix of weights
    :param bias: vector of biases to add to each equation, length should be n, if None use 0 bias
    :param num_iterations: Number of iterations
    :return: list of output cells, the length will be the same n
    '''
    assert type(hidden_weights) == np.ndarray
    assert len(hidden_weights.shape) == 2
    assert hidden_weights.shape[0] == hidden_weights.shape[1]
    assert len(input_idx) == input_weights.shape[1], "{}, {}".format(len(input_idx), input_weights.shape[1])
    assert hidden_weights.shape[0] == input_weights.shape[0]

    n = hidden_weights.shape[0]

    if bias is None:
        bias = [0] * n
    else:
        assert len(bias) == n
    last_idx = query.getNumberOfVariables()
    prev_iteration_idxs = [i + 1 for i in range(last_idx, last_idx + (4 * n), 4)]
    output_idxs = [i + 3 for i in range(last_idx, last_idx + (4 * n), 4)]
    query.setNumberOfVariables(last_idx + (4 * n))  # i, s_i-1 f, s_i b, s_i f

    cell_idx = last_idx
    for i in range(n):
        # i
        query.setLowerBound(cell_idx, 0)
        query.setUpperBound(cell_idx, num_iterations)

        # s_i-1 f
        query.setLowerBound(cell_idx + 1, 0)
        query.setUpperBound(cell_idx + 1, large)

        # s_i b
        query.setLowerBound(cell_idx + 2, -large)
        query.setUpperBound(cell_idx + 2, large)

        # s_i f
        query.setLowerBound(cell_idx + 3, 0)
        query.setUpperBound(cell_idx + 3, large)

        # s_i f = ReLu(s_i b)
        MarabouCore.addReluConstraint(query, cell_idx + 2, cell_idx + 3)

        # s_i-1 f >= i * \sum (x_j_min * w_j)
        # prev_min_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        # prev_min_eq.addAddend(1, last_idx + 1)
        # prev_min_eq.addAddend(1, last_idx + 1)

        # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
        update_eq = MarabouCore.Equation()
        for j in range(len(input_weights[i, :])):
            w = input_weights[i, j]  # round(input_weights[i, j], 2)
            update_eq.addAddend(w, input_idx[j])

        for j, w in enumerate(hidden_weights[i, :]):
            # w = round(w, 2)
            update_eq.addAddend(w, prev_iteration_idxs[j])

        update_eq.addAddend(-1, cell_idx + 2)
        update_eq.setScalar(-bias[i])
        # if print_debug:
        #     update_eq.dump()
        query.addEquation(update_eq)
        cell_idx += 4

    return output_idxs


def add_loop_indices_equations(network, loop_indices):
    '''
    Adds to the network equations that make all loop variabels to be equal
    :param network: marabou quert that the equations will be appended
    :param loop_indices: variables that needs to be equal
    :return: None
    '''
    # Make sure all the iterators are in the same iteration, we create every equation twice
    step_loop_eq = []
    # for idx in loop_indices:
    idx = loop_indices[0]
    for idx2 in loop_indices[1:]:
        if idx < idx2:
            temp_eq = MarabouCore.Equation()
            temp_eq.addAddend(1, idx)
            temp_eq.addAddend(-1, idx2)
            # step_loop_eq.append(temp_eq)
            network.addEquation(temp_eq)


def create_invariant_equations(loop_indices, invariant_eq):
    '''
    create the equations needed to prove using induction from the invariant_eq
    :param loop_indices: List of loop variables (i's), which is the first variable for an RNN cell
    :param invariant_eq: the invariant we want to prove
    :return: [base equations], [step equations]
    '''

    def create_induction_hypothesis_from_invariant_eq():
        '''
        for example our invariant is that s_i f <= i, the induction hypothesis will be s_i-1 f <= i-1
        :return: the induction hypothesis
        '''
        scalar_diff = 0
        hypothesis_eq = []

        cur_temp_eq = MarabouCore.Equation(invariant_eq.getType())
        for addend in invariant_eq.getAddends():
            # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
            if addend.getVariable() in loop_indices:
                scalar_diff = addend.getCoefficient()
            # here we change s_i f to s_i-1 f
            if addend.getVariable() in rnn_output_indices:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
            else:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable())
        cur_temp_eq.setScalar(invariant_eq.getScalar() + scalar_diff)
        hypothesis_eq.append(cur_temp_eq)
        return hypothesis_eq

    rnn_input_indices = [idx + 1 for idx in loop_indices]
    rnn_output_indices = [idx + 3 for idx in loop_indices]

    # equations for induction step
    if isinstance(invariant_eq, list):
        raise Exception

    induction_step = negate_equation(invariant_eq)

    # equations for induction base

    # make sure i == 0 (for induction base)
    loop_equations = []
    for i in loop_indices:
        loop_eq = MarabouCore.Equation()
        loop_eq.addAddend(1, i)
        loop_eq.setScalar(0)
        loop_equations.append(loop_eq)

    # s_i-1 f == 0
    zero_rnn_hidden = []
    for idx in rnn_input_indices:
        base_hypothesis = MarabouCore.Equation()
        base_hypothesis.addAddend(1, idx)
        base_hypothesis.setScalar(0)
        zero_rnn_hidden.append(base_hypothesis)

    step_loop_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    step_loop_eq.addAddend(1, loop_indices[0])
    step_loop_eq.setScalar(1)
    # step_loop_eq.append(step_loop_eq_more_1)

    induction_base_equations = [induction_step] + loop_equations + zero_rnn_hidden

    induction_hypothesis = create_induction_hypothesis_from_invariant_eq()
    induction_step_equations = [induction_step] + [step_loop_eq]
    # induction_step_equations = induction_step + step_loop_eq

    return induction_base_equations, induction_step_equations, induction_hypothesis


def prove_invariant_multi(network, rnn_start_idxs, invariant_equations):
    '''
    Prove invariants where we need to assume multiple assumptions and conclude from them.
    For each of the invariant_equations creating 3 sets: base_equations, hyptosis_equations, step_equations
    First proving on each of the base equations seperatly, Then assuming all the hyptosis equations and proving
    on the step_equations set by set
    At the end of the function the network will be exactly the same as before
    :param network:
    :param rnn_start_idxs:
    :param invariant_equations:
    :return:
    '''
    proved_invariants = [False] * len(invariant_equations)
    base_eq = []
    step_eq = []  # this needs to be a list of lists, each time we work on all equations of a list
    hypothesis_eq = []

    for i in range(len(invariant_equations)):
        cur_base_eq, cur_step_eq, cur_hypothesis_eq = create_invariant_equations(rnn_start_idxs, invariant_equations[i])
        base_eq.append(cur_base_eq)
        step_eq.append(cur_step_eq)
        hypothesis_eq += cur_hypothesis_eq

    # first prove base case for all equations
    for i, ls_eq in enumerate(base_eq):
        for eq in ls_eq:
            network.addEquation(eq)
        marabou_result = marabou_solve_negate_eq(network, print_vars=True)
        # print('induction base query')
        # network.dump()

        for eq in ls_eq:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant {} which is:".format(i))
            # for eq in ls_eq:
            #     eq.dump()
            return False
    # print("proved induction base for all invariants")

    # add all hypothesis equations
    # print("adding hypothesis_eq")
    for eq in hypothesis_eq:
        # eq.dump()
        network.addEquation(eq)

    for i, steq_eq_ls in enumerate(step_eq):
        for eq in steq_eq_ls:
            # eq.dump()
            network.addEquation(eq)

        marabou_result = marabou_solve_negate_eq(network)  # , True, True)
        # print("Querying for induction step: {}".format(marabou_result))
        # network.dump()

        if not marabou_result:
            # for eq in hypothesis_eq:
            #     network.removeEquation(eq)
            # network.dump()
            # print("induction step fail, on invariant:", i)
            proved_invariants[i] = False
        else:
            proved_invariants[i] = True
            # print("induction step work, on invariant:", i)

        for eq in steq_eq_ls:
            network.removeEquation(eq)
    for eq in hypothesis_eq:
        network.removeEquation(eq)

    return proved_invariants


def alphas_to_equations(rnn_start_idxs, rnn_output_idxs, initial_values, inv_type, alphas):
    '''
    Create a list of marabou equations, acording to the template: \alpha*i \le R_i OR \alpha*i \ge R_i
    For parameter look at alpha_to_equation, this is just a syntax sugar to remove the loop from outer functions
    :return: List of marabou equations
    '''
    assert len(rnn_start_idxs) == len(rnn_output_idxs)
    assert len(rnn_start_idxs) == len(initial_values)
    assert len(rnn_start_idxs) == len(alphas)
    invariants = []
    if not isinstance(inv_type, list):
        inv_type = [inv_type] * len(rnn_start_idxs)

    for i in range(len(rnn_start_idxs)):
        invariants.append(
            alpha_to_equation(rnn_start_idxs[i], rnn_output_idxs[i], initial_values[i], alphas[i], inv_type[i]))

    return invariants


def alpha_to_equation(start_idx, output_idx, initial_val, new_alpha, inv_type):
    '''
    Create an invariant equation according to the simple template \alpha*i \le R_i OR \alpha*i \ge R_i
    :param start_idx: index of the rnn iterator (i)
    :param output_idx: index of R_i
    :param initial_val: If inv_type = GE the max value of R_1 if inv_type = LE the min of R_1
    :param new_alpha: alpha to use
    :param inv_type: Marabou.Equation.GE / Marabou.Equation.LE
    :return: marabou equation
    '''
    # Need the invariant from both side because they are all depndent in each other
    invariant_equation = MarabouCore.Equation(inv_type)
    invariant_equation.addAddend(1, output_idx)  # b_i
    if inv_type == MarabouCore.Equation.LE:
        ge_better = -1
    else:
        ge_better = 1

    invariant_equation.addAddend(new_alpha * ge_better, start_idx)  # i
    # TODO: Why isn't it ge_better * initial_val? if it's LE we want:
    # not ( alpha * i + beta \le R ) \iff -alpha * i - beta > R
    invariant_equation.setScalar(initial_val)
    # invariant_equation.dump()
    return invariant_equation


def double_list(ls):
    '''
    create two items from each item in the list
    i.e. if the input is: [1,2,3] the output is: [1,1,2,2,3,3]
    '''
    import copy
    new_ls = []
    for i in range(len(ls)):
        new_ls += [copy.deepcopy(ls[i]), copy.deepcopy(ls[i])]
    return new_ls


def invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs):
    '''
    Creates a function that can verify invariants accodring to the network and rnn indcies
    :param network: Marabou format of a network
    :param rnn_start_idxs: Indcies of the network where RNN cells start
    :param rnn_output_idxs: Output indcies of RNN cells in the network
    :return: A pointer to a function that given a list of equations checks if they stand or not
    '''

    def invariant_oracle(equations_to_verify):
        # assert len(alphas) == len(initial_values)
        # invariant_equations = alphas_to_equations(rnn_start_idxs, rnn_output_idxs, initial_values, alphas)
        return prove_invariant_multi(network, rnn_start_idxs, equations_to_verify)

    return invariant_oracle


def property_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, property_equations):
    def property_oracle(invariant_equations):

        for eq in invariant_equations:
            if eq is not None:
                network.addEquation(eq)

        # TODO: This is only for debug
        # before we prove the property, make sure the invariants does not contradict each other, expect SAT from marabou
        assert not marabou_solve_negate_eq(network, False, False)

        for eq in property_equations:
            if eq is not None:
                network.addEquation(eq)
        res = marabou_solve_negate_eq(network, False, False)
        # network.dump()
        if res:
            pass
        for eq in invariant_equations + property_equations:
            if eq is not None:
                network.removeEquation(eq)
        return res

    return property_oracle


def prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_equations, algorithm,
                            return_alphas=False, number_of_steps=5000):
    add_loop_indices_equations(network, rnn_start_idxs)
    invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs)
    property_oracle = property_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, property_equations)
    equations = algorithm.get_equations()
    res = False
    for i in range(number_of_steps):
        invariant_results = invariant_oracle(equations)
        if all(invariant_results):
            # print('proved an invariant: {}'.format(algorithm.get_alphas()))
            if property_oracle(equations):
                print("proved property after {} iterations, using alphas: {}".format(i, algorithm.get_alphas()))
                res = True
                break
            else:
                # If the property failed no need to pass which invariants passed (of course)
                equations = algorithm.do_step(True, None)
        else:
            # print('fail to prove invariant: {}'.format(algorithm.get_alphas()))
            equations = algorithm.do_step(False, invariant_results)
            # if i % 20 == 0:
            #     print('fail, iteration {} alphas: {}'.format(i, algorithm.get_alphas()))

        #  print progress for debug
        if i % 1000 == 0:
            print('iteration {} sum(alphas): {}'.format(i, sum(algorithm.get_alphas())))
    if i == number_of_steps:
        print("fail to prove property after {} iterations, last alphas: {}".format(i, algorithm.get_alphas()))
    if not return_alphas:
        return res
    else:
        return res, algorithm.get_alphas()
