from z3 import Int, Solver, Array, IntSort, RealSort, ForAll, sat

from maraboupy import MarabouCore

large = 50000.0
small = 10 ** -3


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
        hypothesis_eq = MarabouCore.Equation(invariant_eq.getType())
        for addend in invariant_eq.getAddends():
            # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
            if addend.getVariable() in loop_indices:
                scalar_diff = addend.getCoefficient()
            # here we change s_i f to s_i-1 f
            if addend.getVariable() in rnn_output_indices:
                hypothesis_eq.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
            else:
                hypothesis_eq.addAddend(addend.getCoefficient(), addend.getVariable())
        hypothesis_eq.setScalar(invariant_eq.getScalar() + scalar_diff)
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

    # make sure i == 1 (for induction base)
    loop_equations = []
    for i in loop_indices:
        loop_eq = MarabouCore.Equation()
        loop_eq.addAddend(1, i)
        loop_eq.setScalar(1)
        loop_equations.append(loop_eq)

    # s_i-1 f == 0
    zero_rnn_hidden = []
    # for idx in rnn_input_indices:
    #     base_hypothesis = MarabouCore.Equation()
    #     base_hypothesis.addAddend(1, idx)
    #     base_hypothesis.setScalar(0)
    #     zero_rnn_hidden.append(base_hypothesis)

    # Make sure all the iterators are in the same iteration, we create every equation twice
    step_loop_eq = []
    for idx in loop_indices:
        for idx2 in loop_indices:
            if idx != idx2:
                temp_eq = MarabouCore.Equation()
                temp_eq.addAddend(1, idx)
                temp_eq.addAddend(-1, idx2)
                step_loop_eq.append(temp_eq)

    induction_base_equations = induction_step + loop_equations + zero_rnn_hidden

    # induction_hypothesis = create_induction_hypothesis_from_invariant_eq()
    # induction_step_equations = [induction_hypothesis, induction_step]
    induction_step_equations = induction_step + step_loop_eq

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
    # TODO: Change the method to get also alpha, then we don't limit to linear decaying in the diff
    a_invariants = Array('a_invariants', IntSort(), RealSort())
    b_invariants = Array('b_invariants', IntSort(), RealSort())
    i = Int('i')

    s = Solver()
    # s.add(a_invariants[0] == min_a)
    # s.add(b_invariants[0] == max_b)
    s.add(i <= n_iterations)

    # The invariant
    s.add(ForAll([i], a_invariants[i] >= min_a + a_pace * i))
    s.add(ForAll([i], b_invariants[i] <= max_b + b_pace * i))
    # NOT the property to prove
    s.add(a_invariants[n_iterations] < b_invariants[n_iterations])

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

    # iterator_eq = MarabouCore.Equation()
    # iterator_eq.addAddend(1, 2)
    # iterator_eq.setScalar(n_iterations)
    # network.addEquation(iterator_eq)

    # iterator_eq = MarabouCore.Equation()
    # iterator_eq.addAddend(1, 6)
    # iterator_eq.setScalar(n_iterations)
    # network.addEquation(iterator_eq)
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
    network, rnn_start_idxs, invariant_equation, _, more_base_equations, _  = \
        network_define_f(xlim, ylim, n_iterations)

    base_equations, step_equations = create_invariant_equations(rnn_start_idxs, invariant_equation)
    # exit(1)

    for eq in base_equations:
        eq.dump()
        network.addEquation(eq)
    for eq in more_base_equations:
        eq.dump()
        network.addEquation(eq)

    print("Querying for induction base")
    if not marabou_solve_negate_eq(network):
        print("induction base fail")
        return False
    # exit(1)
    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    network, _, _, _, _, _ = network_define_f(xlim, ylim, n_iterations)

    for eq in step_equations:
        eq.dump()
        network.addEquation(eq)


    print("Querying for induction step")
    return marabou_solve_negate_eq(network)


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
        network, iterators_idx, invariant_equation, output_eq, _ = network_define_f(xlim, ylim, n_iterations)
        # inv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)

        return prove_property_marabou(network, [invariant_equation], output_eq, iterators_idx, n_iterations)


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

    _, _, _, (min_a, max_b), _,  (a_pace, b_pace) = network_define_f(xlim, n_iterations)
    # a_invariant = invariants[0]
    # b_invariant = invariants[1]
    # a_pace = None
    # b_pace = None
    return prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)
