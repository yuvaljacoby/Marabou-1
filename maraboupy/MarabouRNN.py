from maraboupy import MarabouCore
from z3 import *

large = 1000.0
small = 10 ** -2


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


def add_rnn_cell(query, input_weights, hidden_weight):
    '''
    Create rnn cell --> add 3 parameters to the query and the equations that describe the cell
    The output of the cell will be the last parameter that we add
    :param query: the network so far (will add to this)
    :param input_weights: list of tuples, each tuple (variable_idx, weight)
    :param hidden_weight: the weight inside the cell
    :return: the index of the last parameter (which is the output of the cell)
    '''

    last_idx = query.getNumberOfVariables()
    query.setNumberOfVariables(last_idx + 4)  # s_i-1 f, s_i b, s_i f, i

    # i
    query.setLowerBound(last_idx, 0)
    query.setLowerBound(last_idx, large)

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
    # update_eq.dump()


    return last_idx + 3


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


def prove_invariant(xlim, network_define_f, invarinet_define_f):
    '''
    proving invariant on a given rnn cell
    :param n_iterations: max number of times to run the cell
    :param input_weight: The weight for the input (before the cell)
    :param hidden_weight: The weight inside the cell
    :param xlim: limits on the input
    :param invariant_lim: what to prove (that the output of the network is smaller than some function of i)
    :return: True of the invariant holds, false otherwise
    '''

    network = network_define_f(xlim)
    base_invariant_equations, step_invariant_equations, invariant_eq = invarinet_define_f(network)

    for eq in base_invariant_equations:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for induction base")
    if not marabou_solve_negate_eq(network, True):
        print("induction base fail")
        return False
    # exit(1)
    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    network = network_define_f(xlim)
    base_invariant_equations, step_invariant_equations, invariant_eq = invarinet_define_f(network)

    for eq in step_invariant_equations:
        # eq.dump()
        network.addEquation(eq)
    print("Querying for induction step")
    return marabou_solve_negate_eq(network)


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
    print("invariant_equations")
    for eq in invariant_equations:
        eq.dump()
        network.addEquation(eq)

    print("output_equations")
    for eq in output_equations:
        eq.dump()
        network.addEquation(eq)

    print("Querying for output")
    return marabou_solve_negate_eq(network)


def prove_using_invariant(xlim, ylim, n_iterations, network_define_f, invariant_define_f, output_define_f,
                          use_z3=False):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: numebr of times to "run" the rnn cell
    :param network_define_f: pointer to function that defines the network (marabou style), gets xlim return marabou query
    :param invariant_define_f: pointer to function that defines the invariant equations, gets a network returns ([base eq, step eq, equations that hold if ivnariant holds])
    :param output_define_f: pointer to function that defines the output equations, gets, network, ylim, n_iterations return [eq to validate outputs]
    :param use_z3:
    :return:
    '''
    if not prove_invariant(xlim, network_define_f, invariant_define_f):
        print("invariant doesn't hold")
        return False

    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network = network_define_f(xlim)
        _, _, invariant_eq = invariant_define_f(network)
        # TODO: find a better way to remove equations that were added in invariant_define_f
        network = network_define_f(xlim)
        return prove_property_marabou(network, invariant_eq, output_define_f(network, ylim, n_iterations))


def create_invariant_equations(query, loop_indices, invariant_eq):
    '''
    create the equations needed to prove using induction from the invariant_eq
    i.e. list of base equations and step equations
    :param query: The network definition, we need this only if loop_index == None
    :param loop_indices: The index of the loop variable, if doesn't exists yet than give None here and will add it to the query
    :param rnn_end_indices: for each rnn cell in the network this is the output index
    :param invariant_eq: the invariant we want to prove
    :return: [base equations], [step equations]
    '''
    scalar_diff = 0
    # rnn_output_indices = [idx + 2 for idx in rnn_first_indices]
    rnn_input_indices = [idx + 1 for idx in loop_indices]
    rnn_output_indices = [idx + 3 for idx in loop_indices]

    induction_step = negate_equation(invariant_eq)
    # induction_step = MarabouCore.Equation(invariant_eq)

    # print('induction_step')
    # induction_step.dump()

    induction_hypothesis = MarabouCore.Equation(invariant_eq.getType())
    for addend in invariant_eq.getAddends():
        # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
        if addend.getVariable() in loop_indices:
            scalar_diff = addend.getCoefficient()
        # here we change s_i f to s_i-1 f
        if addend.getVariable() in rnn_output_indices:
            induction_hypothesis.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
        else:
            induction_hypothesis.addAddend(addend.getCoefficient(), addend.getVariable())
    induction_hypothesis.setScalar(invariant_eq.getScalar() + scalar_diff)
    print('induction_hypothesis')
    induction_hypothesis.dump()

    # make sure i == 1
    loop_equations = []
    for i in loop_indices:
        loop_eq = MarabouCore.Equation()
        loop_eq.addAddend(1, i)
        loop_eq.setScalar(1)
        # print('loop_eq')
        # loop_eq.dump()
        loop_equations.append(loop_eq)

    zero_rnn_hidden = []
    for idx in rnn_input_indices:
        base_hypothesis = MarabouCore.Equation()
        base_hypothesis.addAddend(1, idx)  # s_i-1 f == s_0 f
        base_hypothesis.setScalar(0)
        # print('base_hypothesis')
        # base_hypothesis.dump()
        zero_rnn_hidden.append(base_hypothesis)

    return [induction_step, induction_hypothesis] + loop_equations + zero_rnn_hidden, [induction_hypothesis,
                                                                                       induction_step]


def prove_invariant_2(network_define_f, xlim, ylim, n_iterations):
    '''
    proving invariant on a given rnn cell
    :param n_iterations: max number of times to run the cell
    :param input_weight: The weight for the input (before the cell)
    :param hidden_weight: The weight inside the cell
    :param xlim: limits on the input
    :param invariant_lim: what to prove (that the output of the network is smaller than some function of i)
    :return: True of the invariant holds, false otherwise
    '''

    network, rnn_start_idxs, invariant_equation, _ = network_define_f(xlim, ylim, n_iterations)
    # print("invariant_equation:")
    # invariant_equation.dump()
    base_equations, step_equations = create_invariant_equations(network, rnn_start_idxs, invariant_equation)

    for eq in base_equations:
        network.addEquation(eq)

    print("Querying for induction base")
    if not marabou_solve_negate_eq(network, True):
        print("induction base fail")
        return False

    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    network, _, _, _ = network_define_f(xlim, ylim, n_iterations)

    for eq in step_equations:
        network.addEquation(eq)

    print("Querying for induction step")
    return marabou_solve_negate_eq(network)


def prove_using_invariant_2(xlim, ylim, n_iterations, network_define_f, use_z3=False):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: numebr of times to "run" the rnn cell
    :param network_define_f: pointer to function that defines the network (marabou style), gets xlim return marabou query
    :param invariant_define_f: pointer to function that defines the invariant equations, gets a network returns ([base eq, step eq, equations that hold if ivnariant holds])
    :param output_define_f: pointer to function that defines the output equations, gets, network, ylim, n_iterations return [eq to validate outputs]
    :param use_z3:
    :return:
    '''
    if not prove_invariant_2(network_define_f, xlim, ylim, n_iterations):
        print("invariant doesn't hold")
        return False
    # exit(1)
    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network, _, invariant_equation, output_eq = network_define_f(xlim, ylim, n_iterations)
        return prove_property_marabou(network, [invariant_equation], output_eq)
