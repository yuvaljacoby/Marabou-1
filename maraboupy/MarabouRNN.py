from z3 import Solver, Array, BitVec, BitVecSort, RealSort, ForAll, sat, BV2Int, BitVecVal

from maraboupy import MarabouCore

large = 500000.0
small = 10 ** -2
TOLERANCE_VALUE = 0.01


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

    vars1, stats1 = MarabouCore.solve(query, "", 0, 0)
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


def add_rnn_cell(query, input_weights, hidden_weight, num_iterations, print_debug=False):
    '''
    Create rnn cell --> add 4 parameters to the query and the equations that describe the cell
    The added parameters are (same order): i, s_i-1 f, s_i b, s_i f
    :param query: the network so far (will add to this)
    :param input_weights: list of tuples, each tuple (variable_idx, weight)
    :param hidden_weight: the weight inside the cell
    :param num_iterations: Number of iterations the cell runs
    :return: the index of the last parameter (which is the output of the cell)
    '''

    last_idx = query.getNumberOfVariables()
    query.setNumberOfVariables(last_idx + 4)  # i, s_i-1 f, s_i b, s_i f

    # i
    # TODO: when doing this we make the number of iterations to be n_iterations + 1
    query.setLowerBound(last_idx, 0)
    query.setUpperBound(last_idx, num_iterations)

    # s_i-1 f
    query.setLowerBound(last_idx + 1, 0)
    query.setUpperBound(last_idx + 1, large)

    # s_i b
    query.setLowerBound(last_idx + 2, -large)
    query.setUpperBound(last_idx + 2, large)

    # s_i f
    query.setLowerBound(last_idx + 3, 0)
    query.setUpperBound(last_idx + 3, large)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(query, last_idx + 2, last_idx + 3)

    # s_i-1 f >= i * \sum (x_j_min * w_j)
    # prev_min_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    # prev_min_eq.addAddend(1, last_idx + 1)
    # prev_min_eq.addAddend(1, last_idx + 1)

    # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
    update_eq = MarabouCore.Equation()
    for var_idx, weight in input_weights:
        update_eq.addAddend(weight, var_idx)
    update_eq.addAddend(hidden_weight, last_idx + 1)
    update_eq.addAddend(-1, last_idx + 2)
    update_eq.setScalar(0)
    # if print_debug:
    #     update_eq.dump()
    query.addEquation(update_eq)

    return last_idx + 3


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
        for eq in invariant_eq:
            cur_temp_eq = MarabouCore.Equation(eq.getType())
            for addend in eq.getAddends():
                # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
                if addend.getVariable() in loop_indices:
                    scalar_diff = addend.getCoefficient()
                # here we change s_i f to s_i-1 f
                if addend.getVariable() in rnn_output_indices:
                    cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
                else:
                    cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable())
            cur_temp_eq.setScalar(eq.getScalar() + scalar_diff)
            hypothesis_eq.append(cur_temp_eq)
        return hypothesis_eq

    rnn_input_indices = [idx + 1 for idx in loop_indices]
    rnn_output_indices = [idx + 3 for idx in loop_indices]

    # equations for induction step
    if isinstance(invariant_eq, list):
        induction_step = []
        for eq in invariant_eq:
            induction_step.append(negate_equation(eq))
    else:
        induction_step = [negate_equation(invariant_eq)]

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

    # Make sure all the iterators are in the same iteration, we create every equation twice
    step_loop_eq = []
    for idx in loop_indices:
        for idx2 in loop_indices:
            if idx != idx2:
                temp_eq = MarabouCore.Equation()
                temp_eq.addAddend(1, idx)
                temp_eq.addAddend(-1, idx2)
                step_loop_eq.append(temp_eq)

    step_loop_eq_more_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
    step_loop_eq_more_1.addAddend(1, loop_indices[0])
    step_loop_eq_more_1.setScalar(1)
    step_loop_eq.append(step_loop_eq_more_1)

    induction_base_equations = induction_step + loop_equations + zero_rnn_hidden

    induction_hypothesis = create_induction_hypothesis_from_invariant_eq()
    induction_step_equations = induction_hypothesis + induction_step + step_loop_eq
    # induction_step_equations = induction_step + step_loop_eq

    return induction_base_equations, induction_step_equations


def prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations):
    '''
    Using z3 to probe the formula
    checking ReLu(sk * w) <= ylim[1] while sk <= sklim
    :param invariant_property: maximum value for sk
    :param weight: the weight between sk and the output
    :param ylim: max output
    :return: True if for every sk <= sklim implies that ReLu(sk * w) <= ylim
    '''
    from math import log2, ceil
    num_bytes = ceil(log2(n_iterations)) + 1
    print('n_iterations', n_iterations, '\nnum_bytes', num_bytes)
    assert n_iterations <= 2 ** num_bytes  # if the bit vec is 32 z3 takes to long
    a_invariants = Array('a_invariants', BitVecSort(num_bytes), RealSort())
    b_invariants = Array('b_invariants', BitVecSort(num_bytes), RealSort())
    i = BitVec('i', num_bytes)
    n = BitVec('n', num_bytes)

    s = Solver()
    s.add(a_invariants[0] == min_a)
    s.add(b_invariants[0] == max_b)
    s.add(n == BitVecVal(n_iterations, num_bytes))

    # The invariant
    s.add(ForAll(i, a_invariants[i] >= a_invariants[0] + BV2Int(i) * a_pace))
    s.add(ForAll(i, b_invariants[i] <= b_invariants[0] + BV2Int(i) * b_pace))
    # s.add(ForAll(i, a_invariants[i] >= a_invariants[0] + BV2Int(i * BitVecVal(a_pace, num_bytes))))
    # s.add(ForAll(i, b_invariants[i] <= b_invariants[0] + BV2Int(i * BitVecVal(b_pace, num_bytes))))

    # NOT the property to prove
    s.add(a_invariants[n] < b_invariants[n])

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        print('proved adversarial property using z3')
        return True


def prove_property_marabou(network, invariant_equations, output_equations, iterators_idx, n_iterations):
    '''
    Prove property using marabou (after checking that invariant holds)
    :param network: marabou definition of the network
    :param invariant_equations: equations that the invariant promises
    :param output_equations: equations that we want to check if holds
    :return: True if the property holds, False otherwise
    '''
    # TODO: Find the problem with the first index in adversarial_robustness
    for idx in iterators_idx[1:]:
        iterator_eq = MarabouCore.Equation()
        iterator_eq.addAddend(1, idx)
        iterator_eq.setScalar(n_iterations)
        network.addEquation(iterator_eq)

    not_output = []
    for eq in output_equations:
        not_output.append(negate_equation(MarabouCore.Equation(eq)))

    print("invariant_equations")
    for eq in invariant_equations:
        # eq.dump()
        network.addEquation(eq)

    print("output_equations")
    for eq in not_output:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for output")
    return marabou_solve_negate_eq(network, True)


def simplify_network_using_invariants(network_define_f, xlim, ylim, n_iterations):
    network, rnn_start_idxs, invariant_equation, *_ = network_define_f(xlim, ylim, n_iterations)

    for idx in rnn_start_idxs:
        for idx2 in rnn_start_idxs:
            if idx != idx2:
                temp_eq = MarabouCore.Equation()
                temp_eq.addAddend(1, idx)
                temp_eq.addAddend(-1, idx2)
                network.addEquation(temp_eq)

    if not isinstance(invariant_equation, list):
        invariant_equation = [invariant_equation]

    for i in range(len(invariant_equation)):
        if not prove_invariant(network, [rnn_start_idxs[i]], invariant_equation[i]):
            print("Fail on invariant: ", i)
            return False
        else:
            # Add the invariant hypothesis for the next proving
            network.addEquation(invariant_equation[i])

    return True


def prove_invariant(network, rnn_start_idxs, invariant_equation):
    '''
    proving invariant network using induction (proving for the first iteration, and concluding that after iteration k
        the property holds assuming k-1 holds)
    does not change the network, removing all the equations that are being added
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param xlim: limits on the input
    :param xlim: limits on the output (for
    :param n_iterations: max number of times to run the cell
    :return: True if the invariant holds, false otherwise
    '''

    base_equations, step_equations = create_invariant_equations(rnn_start_idxs, [invariant_equation])

    for eq in base_equations:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for induction base")
    # network.dump()
    if not marabou_solve_negate_eq(network):
        print("\n")
        # network.dump()
        print("\ninduction base fail")
        return False

    for eq in base_equations:
        network.removeEquation(eq)

    for eq in step_equations:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for induction step")
    # network.dump()
    if not marabou_solve_negate_eq(network):
        # network.dump()
        print("induction step fail")
        return False

    for eq in step_equations:
        network.removeEquation(eq)
    return True


def prove_invariant2(network, rnn_start_idxs, invariant_equations):
    '''
    proving invariant network using induction (proving for the first iteration, and concluding that after iteration k
       the property holds assuming k-1 holds)
    Not changing the network, i.e. every equation we add we also remove
    :param network: description of the NN in marabou style
    :param invariant_equations: List of Marabou equations that describe the current invariant
    :return: True if the invariant holds, false otherwise
    '''

    for i in range(len(invariant_equations)):
        base_equations, step_equations = create_invariant_equations(rnn_start_idxs, [invariant_equations[i]])

        for eq in base_equations:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction base")
        network.dump()

        marabou_result = marabou_solve_negate_eq(network)
        for eq in base_equations:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant:", i)
            return False

        for eq in base_equations:
            network.removeEquation(eq)

        for eq in step_equations:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")
        network.dump()

        marabou_result = marabou_solve_negate_eq(network)
        for eq in step_equations:
            network.removeEquation(eq)

        if not marabou_result:
            # network.dump()
            print("induction step fail, on invariant:", i)
            return False

    return True


def find_stronger_le_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                               initial_values):
    return find_stronger_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                                   initial_values, MarabouCore.Equation.LE)


def find_stronger_ge_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                               initial_values):
    return find_stronger_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                                   initial_values, MarabouCore.Equation.GE)


def find_stronger_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                            initial_values, eq_type=MarabouCore.Equation.GE):
    cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
    while max_alphas[i] - min_alphas[i] > TOLERANCE_VALUE:
        invariant_equation = MarabouCore.Equation(eq_type)
        invariant_equation.addAddend(1, rnn_output_idxs[i])  # b_i
        if eq_type == MarabouCore.Equation.LE:
            ge_better = -1
        else:
            ge_better = 1
        invariant_equation.addAddend(cur_alpha * ge_better, rnn_start_idxs[i])  # i
        invariant_equation.setScalar(initial_values[i])
        prove_inv_res = prove_invariant2(network, rnn_start_idxs, [invariant_equation])
        if prove_inv_res:
            print("For alpha_{} {} invariant holds".format(i, cur_alpha))
            max_alphas[i] = cur_alpha
            return min_alphas, max_alphas, cur_alpha, invariant_equation
        else:
            print("For alpha_{} {} invariant does not hold".format(i, cur_alpha))
            # Invariant does not hold
            min_alphas[i] = cur_alpha
        cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant

        print(cur_alpha)
        # cur_alpha = temp
    return min_alphas, max_alphas, None, None


def find_invariant(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations, min_alphas=None,
                   max_alphas=None, rnn_dependent=None):
    '''
    Function to automatically find invariants that hold and prove the property
    The order of the rnn indices matter (!), we try to prove invariants on them sequentially,
    i.e. if rnn_x is dependent on the output of rnn_y then index(rnn_x) > index(rnn_y)
    :param network: Description of the network in Marabou style
    :param rnn_start_idxs: list of indcies with the iterator variable for each rnn cell
    :param rnn_invariant_type: List of MarabouCore.Equation.GE/LE, for each RNN cell
    :param initial_values: (min_a, max_b)
    :param n_iterations: for how long we will run the network (how many inputs will there be)
    :param min_alphas:
    :param max_alphas:
    :param rnn_dependent: list of lists (or none), for each cell which rnn are dependent on him. we need this to
            recompute the search space after finiding a better invariant
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

    initial_diff = initial_values[0] - initial_values[1]
    assert initial_diff >= 0

    # TODO: Find suitable range for the invariant to be in
    if min_alphas is None:
        min_alphas = [-large] * len(rnn_start_idxs)  # min alpha that we try but invariant does not hold
    if max_alphas is None:
        max_alphas = [large] * len(rnn_start_idxs)  # max alpha that we try but property does not hold

    # For A small alpha yields stronger invariant, while B is the opposite
    alphas = []
    for i, inv_type in enumerate(rnn_invariant_type):
        if inv_type == MarabouCore.Equation.GE:
            alphas.append(min_alphas[i])
        else:
            alphas.append(max_alphas[i])

    still_improve = [True] * len(rnn_start_idxs)

    # Keep track on the invariants we now that hold for each cell
    invariant_that_hold = [None] * len(rnn_start_idxs)
    while any(still_improve):
        # i = 0
        for i in range(len(rnn_start_idxs)):
            if still_improve[i]:
                if invariant_that_hold[i]:
                    network.removeEquation(invariant_that_hold[i])

                min_alphas, max_alphas, temp_alpha, cur_inv_eq = find_stronger_invariant(network, max_alphas,
                                                                                         min_alphas,
                                                                                         i,
                                                                                         rnn_output_idxs,
                                                                                         rnn_start_idxs,
                                                                                         initial_values,
                                                                                         rnn_invariant_type[i])

                if temp_alpha is not None:
                    alphas[i] = temp_alpha
                    invariant_that_hold[i] = cur_inv_eq
                    # Found a better invariant need to change the search space for all the rest
                    if rnn_dependent[i]:
                        for j in rnn_dependent[i]:
                            max_alphas[j] = large
                            min_alphas[j] = -large

                if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
                    still_improve[i] = False

                # This invariant hold, all other rnn cells can use this fact
                network.addEquation(invariant_that_hold[i])

        # TODO: Need to change this, no sense to take the last two alphas, probably need to prove an invariant on A and B also and not only the RNN's
        if prove_adversarial_property_z3(-alphas[-2], alphas[-1], initial_values[-2], initial_values[-1], n_iterations):
            print("Invariant and property holds. invariants:\n\t" + "\n\t".join(
                ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            # print("For alpha_{} {}, alpha_{} {} invariant and property holds".format(0, alphas[0], 1, alphas[1]))
            return True
        else:
            print("Property does not fold for alphas:\n\t" + "\n\t".join(
                ["{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            # print("For alpha_{} {}, alpha_{} {} property does not hold".format(0, alphas[0], 1, alphas[1]))
            # print("For alpha_{} {} invariant holds, property does not".format(i, alphas[i]))
            # The invariant holds but the property does not
            # TODO: We should not change both, change only one if fail change the second if still fail change both
            # max_alphas[-1] = large
            # min_alphas[-1] = -large
            # max_alphas[1] = alphas[1]
        # else:
        #     print("For alpha {} invariant does not hold".format(cur_alpha))
        #     # Invariant does not hold
        #     min_alphas[i] = cur_alpha
        # cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant
        # print(cur_alpha)
        # cur_alpha = temp

    print("Finish trying to find sutiable invariant, the last invariants we found are\n\t" + "\n\t".join(
        ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    print("last search area\n\t" + "\n\t".join(
        ["{}: {} TO {}".format(i, min_a, max_a) for i, (min_a, max_a) in enumerate(zip(min_alphas, max_alphas))]))
    # print("last search area\n\t0: {} TO {}\n\t1: {} TO {}".format(min_alphas[0], max_alphas[0], min_alphas[1],
    #                                                               max_alphas[1]))
    return False


def prove_using_invariant(xlim, ylim, n_iterations, network_define_f, use_z3=False):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: number of times to "run" the rnn cell
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param use_z3:
    :return: True if the invariant holds and we can conclude the property from it, False otherwise
    '''
    if not simplify_network_using_invariants(network_define_f, xlim, ylim, n_iterations):
        print("invariant doesn't hold")
        return False
    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network, iterators_idx, invariant_equation, output_eq, *_ = network_define_f(xlim, ylim, n_iterations)
        # inv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)

        if not isinstance(invariant_equation, list):
            invariant_equation = [invariant_equation]
        return prove_property_marabou(network, invariant_equation, output_eq, iterators_idx, n_iterations)


def prove_adversarial_using_invariant(xlim, n_iterations, network_define_f):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: number of times to "run" the rnn cell
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param use_z3:
    :return: True if the invariant holds and we can conclude the property from it, False otherwise
    '''

    # Use partial define because in the meantime network_define_f gets also ylim which in this case we don't need
    # The partial allows us to use a generic prove_invariant for both cases
    partial_define = lambda xlim, ylim, n_iterations: network_define_f(xlim, n_iterations)

    if not simplify_network_using_invariants(partial_define, xlim, None, n_iterations):
        print("invariant doesn't hold")
        return False

    _, _, _, (min_a, max_b), (a_pace, b_pace), *_ = network_define_f(xlim, n_iterations)
    return prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)
