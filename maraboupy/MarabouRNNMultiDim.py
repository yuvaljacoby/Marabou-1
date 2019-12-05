import numpy as np

from maraboupy import MarabouCore

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
            w = input_weights[i, j] #round(input_weights[i, j], 2)
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
        # network.dump()
        marabou_result = marabou_solve_negate_eq(network, print_vars=True)
        # print('induction base query')
        # if not marabou_result:
        #     network.dump()
        for eq in ls_eq:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant {} which is:".format(i))
            for eq in ls_eq:
                eq.dump()
            return False
    print("proved induction base for all invariants")

    # add all hypothesis equations
    # print("adding hypothesis_eq")
    for eq in hypothesis_eq:
        # eq.dump()
        network.addEquation(eq)

    for i, steq_eq_ls in enumerate(step_eq):
        for eq in steq_eq_ls:
            # eq.dump()
            network.addEquation(eq)

        # print("Querying for induction step")

        marabou_result = marabou_solve_negate_eq(network)

        if not marabou_result:
            # for eq in hypothesis_eq:
            #     network.removeEquation(eq)
            # network.dump()
            print("induction step fail, on invariant:", i)
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
    invariant_equation.setScalar(initial_val)
    invariant_equation.dump()
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
        res = marabou_solve_negate_eq(network, False, True)
        if res:
            # network.dump()
            pass
        for eq in invariant_equations + property_equations:
            if eq is not None:
                network.removeEquation(eq)
        return res

    return property_oracle


def prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_equations, algorithm):
    add_loop_indices_equations(network, rnn_start_idxs)

    invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs)
    property_oracle = property_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, property_equations)
    # Get first batch of inductive alphas
    alphas = algorithm.getAlphasThatProveProperty(invariant_oracle, property_oracle)
    if alphas:
        print('property proved using alphas:', [a.get() for a in alphas])
        return True
    else:
        print('failed to prove property, last used alphas are:', [a.get() for a in algorithm.alphas])
        return False


class AlphaSearchSGD:
    def __init__(self):
        self.alpha = 0
        self.old_alpha = None
        self.large = 10
        self.next_step = None
        # self.reset_search()

    def proved_alpha(self):
        '''
        Update with the last used alpha that is proved to work
        :param used_alpha:
        :return:
        '''
        pass
        # temp_alpha = self.get()
        # if self.alpha is not None and temp_alpha > self.alpha:
        #     self.alpha = temp_alpha

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

        direction = 1
        self.step(direction)

    def update_property_fail(self):
        '''
        Update the using this alpha a property failed, next time will get smaller alpha
        :param used_alpha:
        :return: Wheather we can still improve this alpha or not
        '''
        direction = -1
        # self.max_val = self.get()
        self.step(direction)
        # print("property fail use smaller alpha, new alpha:", self.alpha)
        return self.alpha < self.large

    def step(self, direction):
        sign = lambda x: 1 if x >= 0 else -1
        self.prev_alpha = self.alpha
        if abs(self.alpha) > 0.2:
            self.alpha = self.alpha + (
                    direction * self.alpha * 0.3 * sign(self.alpha))  # do step size 0.3 to the next direction
        else:
            self.alpha = 0.5 * direction
        return self.alpha

    def get(self):
        return self.alpha
        # if self.old_alpha is not None:
        #     old_alpha = self.old_alpha
        #     self.old_alpha = None
        #     return old_alpha

        # return (self.max_val + self.min_val) / 2


class SGDAlphaAlgorithm:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs):
        '''

        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''

        self.rnn_start_idxs = double_list(rnn_start_idxs)
        self.rnn_output_idxs = double_list(rnn_output_idxs)
        min_values = initial_values[0]
        max_values = initial_values[1]
        self.initial_values = [item for i in range(len(min_values)) for item in [min_values[i], max_values[i]]]
        # times two to have for each invariant ge and le
        self.alphas = [AlphaSearchSGD() for _ in range(len(self.initial_values))]
        self.inv_type = [MarabouCore.Equation.GE if i % 2 == 0 else MarabouCore.Equation.LE for i in
                         range(len(self.alphas))]

        self.inductive_steps = 50
        self.invariant_equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self._update_invariant_equation(i)

        # the property steps are much slower, for each step we do all inductive steps at the worst case
        self.property_steps = 10
        # self.alphas_le = [AlphaSearchSGD() for _ in range(len(initial_values[0]))]
        assert len(self.rnn_output_idxs) == len(self.rnn_start_idxs)
        assert len(self.rnn_output_idxs) == len(self.alphas)
        assert len(self.rnn_output_idxs) == len(self.initial_values)
        assert len(self.rnn_output_idxs) == len(self.invariant_equations)

    def _update_invariant_equation(self, i):
        self.invariant_equations[i] = alpha_to_equation(self.rnn_start_idxs[i], self.rnn_output_idxs[i],
                                                        self.initial_values[i], self.alphas[i].get(), self.inv_type[i])


    def getAlphasThatProveProperty(self, invariant_oracle, property_oracle):
        '''
        Look for alphas that prove the property using the "SGD" algorithm.
        i.e. we first check if basic alphas work, if not
        for each alpha:
            check that it's inductive (if not fix)
            check if property holds
        we do this loop property_steps times
        :param invariant_oracle: function pointer, input is list of marabou equations, output is whether this is a valid invariant
        :param property_oracle: function pointer, input is list of marabou equations, output is whether the property holds using this invariants
        :return: list of alphas if proved, None otherwise
        '''
        # TODO: How can I do this I did not prove the invariant holds...
        # First check if need if the property holds
        # if property_oracle(self.invariant_equations):
        #     return self.alphas
        # Run maximum property_steps
        counter = 0
        while counter < self.property_steps:
            counter += 1
            # Do a step to improve each alpha, after that make sure they are inductive and try to prove the propety
            for i, alpha in enumerate(self.alphas):
                alpha.update_property_fail()
                self._update_invariant_equation(i)
                # Get inductive Alphas updates the equations
                if self.getInductiveAlphas(invariant_oracle):
                    print("proved an invariant:", [a.get() for a in self.alphas])
                    if property_oracle(self.invariant_equations):
                        return self.alphas
                else:
                    print("fail to prove inductive alphas cur alphas::", [a.get() for a in self.alphas])
                    return None
        return None

    def _proveInductiveAlphasOnce(self, invariant_oracle, invariant_equations_idx):
        prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])

        for i, res in enumerate(prove_inv_res):
            # i is an index in the current invariant_equations which is a subset of the entire invariants
            idx = invariant_equations_idx[i]
            if res:
                self.alphas[idx].proved_alpha()
            else:
                # Doing a step
                self.alphas[idx].update_invariant_fail()
                self._update_invariant_equation(idx)
                print("after invariant fail, new_alphas:", [a.get() for a in self.alphas])
        return all(prove_inv_res)

    def getInductiveAlphas(self, invariant_oracle):
        '''
        Do a step to get better alphas, use the oracle to verify that they are valid invariant
        :param invariant_oracle: function pointer, input is marabou equations values output is whether this is a valid invariant
        :return: list of alphas if found better invariant, otherwise None
        '''
        ge_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.GE]
        le_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.LE]
        assert sorted(ge_invariant_eq_idx + le_invariant_eq_idx) == list(range(len(self.inv_type)))
        all_ge_proved = False
        all_le_proved = False
        counter = 0
        while counter <= self.inductive_steps:
            counter += 1
            if not all_ge_proved:
                all_ge_proved = self._proveInductiveAlphasOnce(invariant_oracle, ge_invariant_eq_idx)
            if not all_le_proved:
                all_le_proved = self._proveInductiveAlphasOnce(invariant_oracle, le_invariant_eq_idx)

            if all_ge_proved and all_le_proved:
                # Make sure the invariants do not contrdicte each other (we have a lower bound and an upper bound on each cell)
                for i in range(0, len(self.alphas), 2):
                    # The invariant order is GE, LE, GE, LE ....
                    # But when we create an LE equation we multiply alpha by -1
                    assert self.alphas[i].alpha >= -1 * self.alphas[i + 1].alpha, [a.alpha for a in self.alphas]

                return True

        print("didn't find invariant")
        return False

# def improve_mutlidim_invariants_binary_search(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas,
#                                               inv_type):
#     c = 0
#     # prove_inv_res = [False, False, False, False]
#     prove_inv_res = [False, False]
#     # Running until proving all invariants once, at each iteration we make the invariants that didn't pass a bit larger
#     while c <= 20 and not all(prove_inv_res):
#         c += 1
#         invariant_equations = [None] * len(alphas)
#
#         for i in range(0, len(alphas)):
#             if not invariant_equations[i]:
#                 invariant_equations[i] = alpha_to_equation(rnn_start_idxs[i], rnn_output_idxs[i],
#                                                            initial_values[i], alphas[i].get(), inv_type)
#
#         prove_inv_res = prove_invariant_multi(network, rnn_start_idxs, invariant_equations)
#         for i, res in enumerate(prove_inv_res):
#             if res:
#                 alphas[i].proved_alpha()
#             else:
#                 # Doing a step
#                 alphas[i].update_invariant_fail()
#                 print("after invariant fail, new_alphas:", [a.get() for a in alphas])
#                 invariant_equations[i] = None
#
#     if all(prove_inv_res):
#         print("proved an invariant:", [a.get() for a in alphas])
#         return invariant_equations
#     else:
#         raise Exception

# def prove_multidim_property2(network, rnn_start_idxs, rnn_output_idxs, initial_values, property_equations):
#     '''
#
#     :param network: marabou query
#     :param rnn_start_idxs: n indcies of the iterator index
#     :param rnn_output_idxs: n indcies of the output rnn index
#     :param initial_values: list of 2*n items order is: min_0, max_0, min_1, max_1 ... min_n, max_n
#     :param property_equations: list of equations that prove the property
#     :return:
#     '''
#     if isinstance(initial_values[0], np.ndarray):
#         initial_values_ls = [[initial_values[0][i], initial_values[1][i]] for i in range(len(initial_values[0]))]
#         initial_values = [item for sublist in initial_values_ls for item in sublist]
#     # max_alphas = [10] * len(rnn_start_idxs)
#     # min_alphas = [-10] * len(rnn_start_idxs)
#     assert len(rnn_start_idxs) == len(rnn_start_idxs)
#     assert len(rnn_output_idxs) == len(rnn_start_idxs)
#     # assert len(initial_values) // 2 == len(max_alphas)
#
#     add_loop_indices_equations(network, rnn_start_idxs)
#     # Create 2*alpha because we create LE and GE invariant for each
#
#     # max_alphas = double_alphas(max_alphas)
#     # min_alphas = double_alphas(min_alphas)
#     # rnn_start_idxs = double_alphas(rnn_start_idxs)
#     # rnn_output_idxs = double_alphas(rnn_output_idxs)
#
#     proved_property = False
#     c = 0
#     alphas = [AlphaSearchSGD() for _ in range(len(rnn_start_idxs) * 2)]
#     if prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas,
#                                      property_equations):
#         return True
#     else:
#         while c <= 10 and not proved_property:
#             c += 1
#             for i in range(len(alphas)):
#                 # print("alphas:", )
#                 if alphas[i].update_property_fail():
#                     print("after property fail, alphas:", [a.get() for a in alphas])
#                     if prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas,
#                                                      property_equations):
#                         print("property proved, alphas range:",
#                               "\n\t".join(["alpha_{}: {}".format(i, a.get()) for i, a in enumerate(alphas)]))
#                         # print("min_alphas:", "\n\t".join(["alpha_{}: {}".format(i, a) for i, a in enumerate(min_alphas)]))
#                         return True
#                     else:
#                         print("fail to prove property, for the {} time, network:".format(c))
#                         # network.dump()
#
#     print("failed to prove property")
#     return False
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
#                     invariant_equations[i] = alpha_to_equation(rnn_start_idxs[i], rnn_output_idxs[i],
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
# def prove_invatiants_and_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, alphas, property_eqs):
#     assert len(alphas) == len(initial_values)
#     initial_values_ge = [initial_values[i] for i in range(0, len(initial_values), 2)]
#     initial_values_le = [initial_values[i] for i in range(1, len(initial_values), 2)]
#     alphas_ge = [alphas[i] for i in range(0, len(alphas), 2)]
#     alphas_le = [alphas[i] for i in range(1, len(alphas), 2)]
#     # alphas_ge = double_alphas([alphas[0]])  # [alphas[i] for i in range(0, len(alphas), 2)]
#     # alphas_le = double_alphas([alphas[1]])  # [alphas[i] for i in range(1, len(alphas), 2)]
#     # print("starting GE constraints")
#     invariant_equations_ge = improve_mutlidim_invariants_binary_search(network, rnn_start_idxs, rnn_output_idxs,
#                                                                        initial_values_ge, alphas_ge,
#                                                                        MarabouCore.Equation.GE)
#
#     # print("starting LE constraints")
#     invariant_equations_le = improve_mutlidim_invariants_binary_search(network, rnn_start_idxs, rnn_output_idxs,
#                                                                        initial_values_le, alphas_le,
#                                                                        MarabouCore.Equation.LE)
#
#     invariant_equations = invariant_equations_le + invariant_equations_ge
#     # print("proved alphas GE:", [alpha.get() for alpha in alphas_ge])
#     # print("proved alphas LE:", [alpha.get() for alpha in alphas_le])
#
#     if 0:
#         print("property_eqs:")
#         for eq in property_eqs:
#             eq.dump()
#         print("\n")
#     for eq in invariant_equations + property_eqs:
#         network.addEquation(eq)
#     res = marabou_solve_negate_eq(network, False, True)
#     if res:
#         network.dump()
#     for eq in invariant_equations + property_eqs:
#         network.removeEquation(eq)
#     return res
