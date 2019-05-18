from maraboupy import MarabouCore
from z3 import *

large = 1000.0
small = 10 ** -3


def marabou_solve_negate_eq(query, debug=False):
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


def add_rnn_cell(query, input_weights, hidden_weight, num_iterations):
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

    # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
    update_eq = MarabouCore.Equation()
    for var_idx, weight in input_weights:
        update_eq.addAddend(weight, var_idx)
    update_eq.addAddend(hidden_weight, last_idx + 1)
    update_eq.addAddend(-1, last_idx + 2)
    update_eq.setScalar(0)
    query.addEquation(update_eq)

    return last_idx + 3


def create_invariant_equations(loop_indices, invariant_eq):
    '''
    create the equations needed to prove using induction from the invariant_eq
    i.e. list of base equations and step equations
    :param loop_indices: The index of the loop variable, if doesn't exists yet than give None here and will add it to the query
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
    induction_step = negate_equation(invariant_eq)
    induction_hypothesis = create_induction_hypothesis_from_invariant_eq()

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
    for idx in rnn_input_indices:
        base_hypothesis = MarabouCore.Equation()
        base_hypothesis.addAddend(1, idx)
        base_hypothesis.setScalar(0)
        zero_rnn_hidden.append(base_hypothesis)

    induction_base_equations = [induction_step, induction_hypothesis] + loop_equations + zero_rnn_hidden
    induction_step_equations = [induction_hypothesis, induction_step]

    return induction_base_equations, induction_step_equations


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


def prove_property_marabou(network, invariant_equations, output_equations):
    '''
    Prove property using marabou (after checking that invariant holds)
    :param network: marabou definition of the network
    :param invariant_equations: equations that the invariant promises
    :param output_equations: equations that we want to check if holds
    :return: True / False
    '''
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

    network, rnn_start_idxs, invariant_equation, _ = network_define_f(xlim, ylim, n_iterations)

    base_equations, step_equations = create_invariant_equations(rnn_start_idxs, invariant_equation)

    for eq in base_equations:
        network.addEquation(eq)

    print("Querying for induction base")
    if not marabou_solve_negate_eq(network):
        print("induction base fail")
        return False

    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    network, _, _, _ = network_define_f(xlim, ylim, n_iterations)

    for eq in step_equations:
        network.addEquation(eq)

    print("Querying for induction step")
    return marabou_solve_negate_eq(network)


