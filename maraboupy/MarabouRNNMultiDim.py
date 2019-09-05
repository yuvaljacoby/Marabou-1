import numpy as np

from maraboupy import MarabouCore

large = 500000.0
small = 10 ** -2
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 10


def marabou_solve_negate_eq(query, debug=False):
    '''
    Run marabou solver
    :param query: query to execute
    :param debug: if True printing all of the query equations
    :return: True if UNSAT (no valid assignment), False otherwise
    '''
    # if debug:
    #     for eq in query.getEquations():
    #         eq.dump()

    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
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
    assert len(input_idx) == input_weights.shape[1]
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
            update_eq.addAddend(input_weights[i, j], input_idx[j])

        for j, w in enumerate(hidden_weights[i, :]):
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
    for idx in loop_indices:
        for idx2 in loop_indices:
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


def prove_property_marabou(network, invariant_equations, output_equations, iterators_idx, n_iterations):
    '''
    Prove property using marabou (after checking that invariant holds)
    :param network: marabou definition of the network
    :param invariant_equations: equations that the invariant promises
    :param output_equations: equations that we want to check if holds
    :return: True if the property holds, False otherwise
    '''
    added_equations = []
    if iterators_idx:
        for idx in iterators_idx:
            iterator_eq = MarabouCore.Equation()
            iterator_eq.addAddend(1, idx)
            iterator_eq.setScalar(n_iterations)
            added_equations.append(iterator_eq)
            network.addEquation(iterator_eq)

    not_output = []
    for eq in output_equations:
        not_output.append(negate_equation(MarabouCore.Equation(eq)))

    if invariant_equations:
        for eq in invariant_equations:
            added_equations.append(eq)
            network.addEquation(eq)

    for eq in not_output:
        added_equations.append(eq)
        network.addEquation(eq)

    print("prove property on marabou:")
    # network.dump()
    ret_val = marabou_solve_negate_eq(network, True)

    for eq in added_equations:
        network.removeEquation(eq)
    return ret_val


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
    for ls_eq in base_eq:
        for eq in ls_eq:
            network.addEquation(eq)
        marabou_result = marabou_solve_negate_eq(network)
        print('induction base query')
        # network.dump()
        for eq in ls_eq:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant:", i)
            return False
    print("proved induction base for all invariants")

    # add all hypothesis equations
    print("adding hypothesis_eq")
    for eq in hypothesis_eq:
        # eq.dump()
        network.addEquation(eq)

    for i, steq_eq_ls in enumerate(step_eq):
        for eq in steq_eq_ls:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")

        marabou_result = marabou_solve_negate_eq(network)

        if not marabou_result:
            # for eq in hypothesis_eq:
            #     network.removeEquation(eq)
            # network.dump()
            print("induction step fail, on invariant:", i)
            proved_invariants[i] = False
        else:
            proved_invariants[i] = True
            print("induction step work, on invariant:", i)

        for eq in steq_eq_ls:
            network.removeEquation(eq)
    for eq in hypothesis_eq:
        network.removeEquation(eq)

    return proved_invariants


def get_new_invariant(start_idx, output_idx, initial_val, new_alpha, inv_type):
    # Need the invariant from both side because they are all depndent in each other
    invariant_equation = MarabouCore.Equation(inv_type)
    invariant_equation.addAddend(1, output_idx)  # b_i
    if inv_type == MarabouCore.Equation.LE:
        ge_better = -1
    else:
        ge_better = 1
    if new_alpha is None:
        print('a')
    invariant_equation.addAddend(new_alpha * ge_better, start_idx)  # i
    invariant_equation.setScalar(initial_val)

    return invariant_equation


def double_alphas(alphas):
    new_alpha = []
    for i in range(len(alphas)):
        new_alpha += [alphas[i], alphas[i]]
    return new_alpha


def prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas, property_eqs):
    invariant_equations = improve_mutlidim_invariants_binary_search(network, rnn_start_idxs, rnn_output_idxs,
                                                                    initial_values, alphas)
    print("proved alphas:", [alpha.get() for alpha in alphas])
    for eq in invariant_equations + property_eqs:
        network.addEquation(eq)
    res = marabou_solve_negate_eq(network)
    if res:
        network.dump()
    for eq in invariant_equations + property_eqs:
        network.removeEquation(eq)
    return res


def prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, property_equations):
    '''

    :param network: marabou query
    :param rnn_start_idxs: n indcies of the iterator index
    :param rnn_output_idxs: n indcies of the output rnn index
    :param initial_values: list of 2*n items order is: min_0, max_0, min_1, max_1 ... min_n, max_n
    :param property_equations: list of equations that prove the property
    :return:
    '''

    max_alphas = [10] * len(rnn_start_idxs)
    min_alphas = [-10] * len(rnn_start_idxs)
    assert len(max_alphas) == len(rnn_start_idxs)
    assert len(rnn_output_idxs) == len(rnn_start_idxs)
    assert len(initial_values) // 2 == len(max_alphas)

    add_loop_indices_equations(network, rnn_start_idxs)
    # Create 2*alpha because we create LE and GE invariant for each
    max_alphas = double_alphas(max_alphas)
    min_alphas = double_alphas(min_alphas)
    rnn_start_idxs = double_alphas(rnn_start_idxs)
    rnn_output_idxs = double_alphas(rnn_output_idxs)

    proved_property = False
    c = 0
    alphas = [AlphaSearch() for _ in range(len(rnn_start_idxs))]
    if prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas,
                                     property_equations):
        return True
    else:
        while c <= 10 and not proved_property:
            c += 1
            for i in range(len(alphas)):
                if alphas[i].update_property_fail():
                    # for j in range(len(alphas)):
                    #     if j != i:
                    #         alphas[j].reset_search()
                    if prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas,
                                                     property_equations):
                        print("property proved, alphas range:",
                              "\n\t".join(["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
                        # print("min_alphas:", "\n\t".join(["alpha_{}: {}".format(i, a) for i, a in enumerate(min_alphas)]))
                        return True
                    else:
                        print("fail to prove property {}, network:".format(c))

    print("failed to prove property")
    return False


# def improve_mutlidim_invariants(network, rnn_start_idxs, rnn_output_idxs, initial_values, max_alpha):
#     # initial_values = double_alphas(initial_values)
#     c = 0
#     new_alphas = max_alpha #[0] * len(rnn_start_idxs)
#     proved_alphas = [None] * len(rnn_start_idxs)
#     prove_inv_res = [False] * len(rnn_start_idxs)
#     # new_alphas = [None] * len(prove_inv_res)
#     while c < 20 and not all(prove_inv_res):
#         print('c:', c)
#         c += 1
#         invariant_equations = [None] * len(rnn_start_idxs)
#
#         # for i in range(len(rnn_start_idxs)):
#         #     if not prove_inv_res[i]:
#         #         new_alphas[i] = (max_alphas[i] + min_alphas[i]) / 2
#             # invariant_equations = []
#         for k in range(0, len(rnn_start_idxs), 2):
#             for j, inv_type in enumerate([MarabouCore.Equation.GE, MarabouCore.Equation.LE]):
#                 i = k + j
#                 if not invariant_equations[i]:
#                     invariant_equations[i] = get_new_invariant(rnn_start_idxs[i], rnn_output_idxs[i],
#                                                                initial_values[i], new_alphas[i], inv_type)
#
#         prove_inv_res = prove_invariant_multi(network, rnn_start_idxs, invariant_equations)
#         # print("results:", prove_inv_res)
#         for i, res in enumerate(prove_inv_res):
#             if res:
#                 proved_alphas[i] = new_alphas[i]
#                 # reduce alpha for the next iteration
#                 # if abs(new_alphas[i]) <= 0.25:
#                 #     new_alphas[i] = -1
#                 # elif proved_alphas[i] > 0:
#                 #     new_alphas[i] = new_alphas[i] / 2
#                 # else:
#                 #     new_alphas[i] = new_alphas[i] * 2
#             else:
#                 if abs(new_alphas[i]) <= 0.25:
#                     new_alphas[i] = 1
#                 elif new_alphas[i] > 0:
#                     new_alphas[i] = new_alphas[i] * 2
#                 else:
#                     new_alphas[i] = new_alphas[i] / 2
#                 # min_alphas[i] = new_alphas[i]
#                 invariant_equations[i] = None
#
#     if all(prove_inv_res):
#         return invariant_equations, proved_alphas
#     else:
#         return None, None, None

class AlphaSearch:
    def __init__(self):
        self.alpha = None
        self.old_alpha = None
        self.large = 10
        self.reset_search()

    def proved_alpha(self):
        '''
        Update with the last used alpha that is proved to work
        :param used_alpha:
        :return:
        '''
        temp_alpha = self.get()
        if self.alpha is not None and temp_alpha > self.alpha:
            self.alpha = temp_alpha

    def reset_search(self):
        self.max_val = self.large
        self.min_val = -self.large
        self.old_alpha = self.get()

    def update_invariant_fail(self):
        '''
        Next time the will get a larger alpha
        :param used_alpha:
        :return:
        '''
        self.min_val = self.get()
        self.alpha = None

    def update_property_fail(self):
        '''
        Update the using this alpha a property failed, next time will get smaller alpha
        :param used_alpha:
        :return: Wheather we can still improve this alpha or not
        '''
        self.max_val = self.get()
        return self.max_val - self.min_val > TOLERANCE_VALUE

    def get(self):
        # if self.old_alpha is not None:
        #     old_alpha = self.old_alpha
        #     self.old_alpha = None
        #     return old_alpha
        return (self.max_val + self.min_val) / 2


def improve_mutlidim_invariants_binary_search(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas):
    c = 0
    prove_inv_res = [False, False, False, False]
    # Running until proving all invariants once, at each iteration we make the invariants that didn't pass a bit larger
    while c <= 20 and not all(prove_inv_res):
        c += 1
        invariant_equations = [None] * len(alphas)

        for k in range(0, len(alphas), 2):
            for j, inv_type in enumerate([MarabouCore.Equation.GE, MarabouCore.Equation.LE]):
                i = k + j
                if not invariant_equations[i]:
                    invariant_equations[i] = get_new_invariant(rnn_start_idxs[i], rnn_output_idxs[i],
                                                               initial_values[i], alphas[i].get(), inv_type)

        prove_inv_res = prove_invariant_multi(network, rnn_start_idxs, invariant_equations)
        for i, res in enumerate(prove_inv_res):
            if res:
                alphas[i].proved_alpha()
            else:
                alphas[i].update_invariant_fail()
                invariant_equations[i] = None

    if all(prove_inv_res):
        return invariant_equations
    else:
        raise Exception


def find_invariant_marabou_multidim(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations,
                                    property_equations,
                                    rnn_dependent=None):
    '''
    Function to automatically find invariants that hold and prove the property
    The order of the rnn indices matter (!), we try to prove invariants on them sequentially,
    i.e. if rnn_x is dependent on the output of rnn_y then index(rnn_x) > index(rnn_y)
    :param network: Description of the network in Marabou style
    :param rnn_start_idxs: list of lists, each list if a set of rnns we need to prove together, then each cell is index
     with the iterator variable for each rnn cell
    :param rnn_invariant_type: List of MarabouCore.Equation.GE/LE, for each RNN cell
    :param n_iterations: for how long we will run the network (how many inputs will there be)
    :param rnn_dependent: list of lists (or none), for each cell which rnn are dependent on him. we need this to
            recompute the search space after finding a better invariant
    :return:
    '''

    assert len(rnn_start_idxs) == len(rnn_invariant_type)
    for t in rnn_invariant_type:
        assert t == MarabouCore.Equation.GE or t == MarabouCore.Equation.LE
    if not rnn_dependent:
        rnn_dependent = [None] * len(rnn_start_idxs)
    assert len(rnn_dependent) == len(rnn_invariant_type)

    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    invariant_equation = None
    assert invariant_equation is None

    min_alphas = [[-large] * len(rnn_start_idxs[i]) for i in
                  range(len(rnn_start_idxs))]  # min alpha that we try but invariant does not hold
    max_alphas = [[large] * len(rnn_start_idxs[i]) for i in
                  range(len(rnn_start_idxs))]  # max alpha that we try but property does not hold

    alphas = []
    for i, inv_type in enumerate(rnn_invariant_type):
        if inv_type == MarabouCore.Equation.GE:
            alphas.append(min_alphas[i])
        else:
            alphas.append(max_alphas[i])

    still_improve = [True] * len(rnn_start_idxs)

    # Keep track on the invariants we know that hold for each cell
    invariant_that_hold = [[None] * len(rnn_start_idxs[i]) for i in range(len(rnn_start_idxs))]

    while any(still_improve):

        for i in range(len(rnn_start_idxs)):
            if still_improve[i]:

                if invariant_that_hold[i]:
                    for inv in invariant_that_hold[i]:
                        if inv:
                            network.removeEquation(inv)

                counter = 0
                cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
                proven_invariant_equation = None

                if cur_alpha is not None:
                    alphas[i] = cur_alpha
                    invariant_that_hold[i] = proven_invariant_equation
                    # Found a better invariant need to change the search space for all the rest
                    if rnn_dependent[i]:
                        print("found invariant for: {}, zeroing: {}".format(i, rnn_dependent[i]))
                        for j in rnn_dependent[i]:
                            max_alphas[j] = large
                            min_alphas[j] = -large
                            still_improve[i] = True

                if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
                    still_improve[i] = False

                # This invariant hold, all other rnn cells can use this fact
                network.addEquation(invariant_that_hold[i])

        if prove_property_marabou(network, None, property_equations, None, n_iterations):
            print("Invariant and property holds. invariants:\n\t" + "\n\t".join(
                ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            return True
        else:
            print("Property does not fold for alphas:\n\t" + "\n\t".join(
                ["{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    # network.dump()
    print("Finish trying to find sutiable invariant, the last invariants we found are\n\t" + "\n\t".join(
        ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    print("last search area\n\t" + "\n\t".join(
        ["{}: {} TO {}".format(i, min_a, max_a) for i, (min_a, max_a) in enumerate(zip(min_alphas, max_alphas))]))
    return False
