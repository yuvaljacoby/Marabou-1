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
    if debug:
        for eq in query.getEquations():
            eq.dump()

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
    prev_min_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    prev_min_eq.addAddend(1, last_idx + 1)
    prev_min_eq.addAddend(1, last_idx + 1)

    # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
    update_eq = MarabouCore.Equation()
    for var_idx, weight in input_weights:
        update_eq.addAddend(weight, var_idx)
    update_eq.addAddend(hidden_weight, last_idx + 1)
    update_eq.addAddend(-1, last_idx + 2)
    update_eq.setScalar(0)
    if print_debug:
        update_eq.dump()
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
        eq.dump()
        network.addEquation(eq)

    print("output_equations")
    for eq in not_output:
        eq.dump()
        network.addEquation(eq)

    print("Querying for output")
    return marabou_solve_negate_eq(network, True)


def prove_invariant(network_define_f, xlim, ylim, n_iterations):
    '''
    proving invariant network using induction (proving for the first iteration, and concluding that after iteration k
        the property holds assuming k-1 holds)
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param xlim: limits on the input
    :param xlim: limits on the output (for
    :param n_iterations: max number of times to run the cell
    :return: True if the invariant holds, false otherwise
    '''
    network, rnn_start_idxs, invariant_equation, *_ = network_define_f(xlim, ylim, n_iterations)

    if not isinstance(invariant_equation, list):
        invariant_equation = [invariant_equation]

    for i in range(len(invariant_equation)):
        network, rnn_start_idxs, invariant_equation, *_ = network_define_f(xlim, ylim, n_iterations)

        if not isinstance(invariant_equation, list):
            invariant_equation = [invariant_equation]

        base_equations, step_equations = create_invariant_equations(rnn_start_idxs, [invariant_equation[i]])
        # exit(1)

        for eq in base_equations:
            eq.dump()
            network.addEquation(eq)

        print("Querying for induction base")
        if not marabou_solve_negate_eq(network):
            print("induction base fail, on invariant:", i)
            return False
        # exit(1)
        # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
        network, *_ = network_define_f(xlim, ylim, n_iterations)

        for eq in step_equations:
            eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")
        if not marabou_solve_negate_eq(network):
            print("induction step fail, on invariant:", i)
            return False

    return True


def prove_invariant2(network_define_f, invariant_equations, xlim, n_iterations):
    '''
       proving invariant network using induction (proving for the first iteration, and concluding that after iteration k
           the property holds assuming k-1 holds)
       :param network_define_f: function that returns the marabou network, invariant, output property
       :param invariant_equations: List of Marabou equations that describe the current invariant
       :param xlim: limits on the input
       :param n_iterations: max number of times to run the cell
       :return: True if the invariant holds, false otherwise
       '''
    network, rnn_start_idxs, *_ = network_define_f(xlim, None, n_iterations)

    for i in range(len(invariant_equations)):
        base_equations, step_equations = create_invariant_equations(rnn_start_idxs, [invariant_equations[i]])

        for eq in base_equations:
            eq.dump()
            network.addEquation(eq)

        print("Querying for induction base")
        if not marabou_solve_negate_eq(network):
            print("induction base fail, on invariant:", i)
            return False
        # exit(1)
        # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
        network, *_ = network_define_f(xlim, None, n_iterations)

        for eq in step_equations:
            eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")
        if not marabou_solve_negate_eq(network):
            print("induction step fail, on invariant:", i)
            return False

    return True


def find_stronger_invariant(network_define_f, xlim, n_iterations, output_idx, initial_value, i_idx, cur_alpha,
                            search_range,
                            larger_better=True, tolerance=TOLERANCE_VALUE):
    '''
    Improve the invariant with a stronger one, get alpha
    :param network_define_f: function the describes the network
    :param output_idx: Index we want to prove on
    :param initial_value: What is the initial value of this output idx
    :param i_idx: index of the iterator counter
    :param cur_alpha: current value that we try to improve
    :param search_range: where to search alpha
    :param larger_better: Should we try to get alpha larger or smaller
    :param tolerance: how small changes effect alpha, i.e. if the change in the alpha is smaller then this we stop
    :return:
    '''
    proved_invariant = False
    improvement = True
    prev_alpha = cur_alpha

    while improvement:
        if larger_better:
            invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
            invariant_equation.addAddend(cur_alpha, i_idx)  # i
        else:
            invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
            invariant_equation.addAddend(-cur_alpha, i_idx)  # i

        invariant_equation.addAddend(1, output_idx)  # a_i
        invariant_equation.setScalar(initial_value)

        if prove_invariant2(network_define_f, [invariant_equation], xlim, n_iterations):
            print('proved invariant, alpha:', cur_alpha)
            prev_alpha = cur_alpha
            if larger_better:
                search_range = (cur_alpha, search_range[1])
                cur_alpha += (search_range[1] - cur_alpha) / 2.0  # Make alpha larger
                assert search_range[1] > cur_alpha
                assert cur_alpha > prev_alpha
            else:
                search_range = (search_range[0], cur_alpha)
                cur_alpha -= (cur_alpha - search_range[0]) / 2  # Make alpha smaller
                assert cur_alpha < prev_alpha

            improvement = True
            proved_invariant = True  # It's enough to get here once to know we proved the invariant
            if abs(prev_alpha - cur_alpha) < tolerance:
                # otherwise this will go on forever
                improvement = False
        else:
            # Finished going in this direction for this invariant
            if larger_better:
                search_range = (search_range[0], cur_alpha)
            else:
                search_range = (cur_alpha, search_range[1])
            cur_alpha = prev_alpha
            improvement = False

    if not proved_invariant:
        cur_alpha = None

    return cur_alpha, search_range


def find_stronger_le_invariant(network_define_f, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                               initial_values, xlim, n_iterations):
    cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
    while max_alphas[i] - min_alphas[i] > TOLERANCE_VALUE:
        invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
        invariant_equation.addAddend(1, rnn_output_idxs[i])  # b_i
        invariant_equation.addAddend(-cur_alpha, rnn_start_idxs[i])  # i
        invariant_equation.setScalar(initial_values[i])
        if prove_invariant2(network_define_f, [invariant_equation], xlim, n_iterations):
            print("For alpha_{} {} invariant holds".format(i, cur_alpha))
            max_alphas[i] = cur_alpha
            return min_alphas, max_alphas, cur_alpha
        else:
            print("For alpha_{} {} invariant does not hold".format(i, cur_alpha))
            # Invariant does not hold
            min_alphas[i] = cur_alpha
        cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant
        print(cur_alpha)
        # cur_alpha = temp
    return min_alphas, max_alphas, None


def find_stronger_ge_invariant(network_define_f, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                               initial_values, xlim, n_iterations):
    cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
    while max_alphas[i] - min_alphas[i] > TOLERANCE_VALUE:
        invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
        invariant_equation.addAddend(1, rnn_output_idxs[i])  # b_i
        invariant_equation.addAddend(cur_alpha, rnn_start_idxs[i])  # i
        invariant_equation.setScalar(initial_values[i])
        if prove_invariant2(network_define_f, [invariant_equation], xlim, n_iterations):
            print("For alpha_{} {} invariant holds".format(i, cur_alpha))
            max_alphas[i] = cur_alpha
            return min_alphas, max_alphas, cur_alpha
        else:
            print("For alpha_{} {} invariant does not hold".format(i, cur_alpha))
            # Invariant does not hold
            min_alphas[i] = cur_alpha
        cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant
        print(cur_alpha)
        # cur_alpha = temp
    return min_alphas, max_alphas, None


def find_invariant2(network_define_f, xlim, ylim, n_iterations, min_alphas=None, max_alphas=None):
    network, rnn_start_idxs, invariant_equation, initial_values, *_ = network_define_f(xlim, ylim, n_iterations)
    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    invariant_equation = None
    assert invariant_equation is None

    initial_diff = initial_values[0] - initial_values[1]
    assert initial_diff > 0

    # The invariant holds for sure with zero
    large = initial_diff
    if min_alphas is None:
        min_alphas = [0] * len(rnn_start_idxs)  # min alpha that we try but invariant does not hold
        max_alphas = [(40)] * len(rnn_start_idxs)  # max alpha that we try but property does not hold

    alphas = [min_alphas[0], max_alphas[1]]  # For A small alpha yields stronger invariant, while B is the opposite

    i = 0
    # alphas[i], (min_alphas[i], max_alphas[i]) = find_stronger_invariant(network_define_f, xlim, n_iterations, rnn_output_idxs[i],
    #                                                                     initial_values[i], rnn_start_idxs[i], alphas[i],
    #                                                                     (min_alphas[i], max_alphas[i]), True)
    alphas[i] = 0
    # min_alphas[i] = 0
    # max_alphas[i] = 0

    i = 1
    alphas[i], (min_alphas[i], max_alphas[i]) = find_stronger_invariant(network_define_f, xlim, n_iterations,
                                                                        rnn_output_idxs[i],
                                                                        initial_values[i], rnn_start_idxs[i], alphas[i],
                                                                        (min_alphas[i], max_alphas[i]), False)

    if None in alphas:
        print("Couldn't find invariant for one of the rnn cells, invariant found:", alphas)
        return False

    # i =1
    # improvement = True
    # prev_alpha = alphas[i]
    #
    # while improvement:
    #     # We only make invariant smaller
    #     invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    #     invariant_equation.addAddend(1, rnn_output_idxs[i])
    #     invariant_equation.addAddend(-alphas[i], rnn_start_idxs[i])  # i
    #     invariant_equation.setScalar(initial_values[i])
    #
    #     if prove_invariant2(network_define_f, [invariant_equation], xlim, n_iterations):
    #         print('proved invariant, alpha:', alphas[i])
    #         max_alphas[i] = alphas[i]
    #         prev_alpha = alphas[i]
    #         alphas[i] = (alphas[i] - min_alphas[i]) / 2  # Make alpha smaller
    #         improvement = True
    #         if abs(prev_alpha - alphas[i]) < TOLERANCE_VALUE:
    #             # otherwise this will go on forever
    #             improvement = False
    #     else:
    #         # Finished going in this direction for this invariant
    #         alphas[i] = prev_alpha
    #         improvement = False

    # return False
    if prove_adversarial_property_z3(-alphas[0], alphas[1], initial_values[0], initial_values[1], n_iterations):
        # invariant hold and property hold
        print('Proved adversarial property, alphas:', alphas[0], ",", alphas[1])
        return True
    else:
        # # Property does not hold for the invariants
        # for i, output_idx in enumerate(rnn_output_idxs):
        #     improvement = True
        #     prev_alpha = alphas[i]
        #     while improvement:
        #         if prove_adversarial_property_z3(-alphas[0], alphas[1], initial_values[0], initial_values[1],
        #                                          n_iterations):
        #             # Finished going in this direction for this invariant
        #             alphas[i] = prev_alpha
        #             improvement = False
        #         else:
        #             min_alphas[i] = alphas[i]
        #             alphas[i] = (max_alphas[i] - alphas[i]) / 2  # make alpha larger
        #             improvement = True
        i = 0
        # alphas[i], (min_alphas[i], max_alphas[i]) = find_weaker_invariant(network_define_f, xlim, n_iterations,
        #                                                                     rnn_output_idxs[i],
        #                                                                     initial_values[i], rnn_start_idxs[i],
        #                                                                     alphas[i],
        #                                                                     (min_alphas[i], max_alphas[i]), True)
        alphas[i] = 0

        i = 1
        alphas[i], (min_alphas[i], max_alphas[i]) = find_weaker_invariant(network_define_f, xlim, n_iterations,
                                                                          rnn_output_idxs[i],
                                                                          initial_values[i], rnn_start_idxs[i],
                                                                          alphas[i],
                                                                          (min_alphas[i], max_alphas[i]), False)

    if max_alphas[0] - min_alphas[0] < 0.1 and max_alphas[1] - min_alphas[1] < 0.1:
        print('finish to search')
        return False
    else:
        return find_invariant(network_define_f, xlim, ylim, n_iterations, min_alphas, max_alphas)


def find_invariant(network_define_f, xlim, ylim, n_iterations, min_alphas=None, max_alphas=None):
    network, rnn_start_idxs, invariant_equation, initial_values, *_ = network_define_f(xlim, ylim, n_iterations)
    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    invariant_equation = None
    assert invariant_equation is None

    initial_diff = initial_values[0] - initial_values[1]
    assert initial_diff > 0

    # The invariant holds for sure with zero
    large = initial_diff
    if min_alphas is None:
        min_alphas = [-100] * 2  # len(rnn_start_idxs) # min alpha that we try but invariant does not hold
        max_alphas = [100] * 2  # len(rnn_start_idxs)  # max alpha that we try but property does not hold

    alphas = [min_alphas[0], max_alphas[1]]  # For A small alpha yields stronger invariant, while B is the opposite

    # i = 0
    # alphas[i] = 0
    # max_alphas[i] = 0
    # min_alphas[i] = 0
    # cur_alpha = min_alphas[i]
    # while max_alphas[i] - min_alphas[i] > 0.1:
    #     invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    #     invariant_equation.addAddend(1, rnn_output_idxs[0])  # b_i
    #     invariant_equation.addAddend(cur_alpha, rnn_start_idxs[0])  # i
    #     invariant_equation.setScalar(initial_values[i])
    #     if prove_invariant2(network_define_f, [invariant_equation], xlim, n_iterations):
    #         alphas[i] = cur_alpha
    #         if prove_adversarial_property_z3(-alphas[0], alphas[1], initial_values[0], initial_values[1], n_iterations):
    #             print("For alpha {} invariant and property holds".format(cur_alpha))
    #             return True
    #         else:
    #             print("For alpha {} invariant holds, property does not".format(cur_alpha))
    #             # The invariant holds but the property does not
    #             max_alphas[i] = cur_alpha
    #             # cur_alpha = cur_alpha / 2
    #     else:
    #         print("For alpha {} invariant does not hold".format(cur_alpha))
    #         # Invariant does not hold
    #         min_alphas[i] = cur_alpha
    #     cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant
    #     print(cur_alpha)

    still_improve = [True, True]
    # cur_alpha = max_alphas[i]
    while any(still_improve):

        i = 0
        if still_improve[i]:
            min_alphas, max_alphas, temp_alpha = find_stronger_ge_invariant(network_define_f, max_alphas, min_alphas, i,
                                                                            rnn_output_idxs, rnn_start_idxs,
                                                                            initial_values,
                                                                            xlim, n_iterations)
            if temp_alpha is not None:
                alphas[i] = temp_alpha
            if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
                still_improve[i] = False

        i = 1
        if still_improve[i]:
            min_alphas, max_alphas, temp_alpha = find_stronger_le_invariant(network_define_f, max_alphas, min_alphas, i,
                                                                            rnn_output_idxs, rnn_start_idxs,
                                                                            initial_values,
                                                                            xlim, n_iterations)

            if temp_alpha is not None:
                alphas[i] = temp_alpha
            if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
                still_improve[i] = False

        # 1.1, 12.5, 6, 2, 3
        if prove_adversarial_property_z3(-alphas[0], alphas[1], initial_values[0], initial_values[1], n_iterations):
            print("For alpha_{} {}, alpha_{} {} invariant and property holds".format(0, alphas[0], 1, alphas[1]))
            return True
        else:
            print("For alpha_{} {}, alpha_{} {} property does not hold".format(0, alphas[0], 1, alphas[1]))
            # print("For alpha_{} {} invariant holds, property does not".format(i, alphas[i]))
            # The invariant holds but the property does not
            # TODO: We should not change both, change only one if fail change the second if still fail change both
            max_alphas[0] = alphas[0]
            max_alphas[1] = alphas[1]
            # cur_alpha = cur_alpha / 2
        # else:
        #     print("For alpha {} invariant does not hold".format(cur_alpha))
        #     # Invariant does not hold
        #     min_alphas[i] = cur_alpha
        # cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant
        # print(cur_alpha)
        # cur_alpha = temp

    print(
        "Finish trying to find sutiable invariant, the last invariants we found are\n\talpha_0: {}\n\talpha_1: {}".format(
            alphas[0], alphas[1]))
    print("last search area\n\t0: {} TO {}\n\t1: {} TO {}".format(min_alphas[0], max_alphas[0], min_alphas[1],
                                                                  max_alphas[1]))
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
    if not prove_invariant(network_define_f, xlim, ylim, n_iterations):
        print("invariant doesn't hold")
        return False
    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network, iterators_idx, invariant_equation, output_eq, *_ = network_define_f(xlim, ylim, n_iterations)
        # inv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)

        return prove_property_marabou(network, [invariant_equation], output_eq, iterators_idx, n_iterations)


def prove_adversarial_property(xlim, n_iterations, network_define_f):
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

    a_pace, b_pace = find_invariant(partial_define, xlim, None, n_iterations)
    if a_pace is None or b_pace is None:
        print("invariant doesn't hold")
        return False

    _, _, _, (min_a, max_b), *_ = network_define_f(xlim, n_iterations)
    return prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)


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

    if not prove_invariant(partial_define, xlim, None, n_iterations):
        print("invariant doesn't hold")
        return False

    _, _, _, (min_a, max_b), (a_pace, b_pace), *_ = network_define_f(xlim, n_iterations)
    return prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)
