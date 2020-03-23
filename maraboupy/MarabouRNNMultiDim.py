from timeit import default_timer as timer

import numpy as np

from maraboupy import MarabouCore
from maraboupy.MarabouRnnModel import RnnMarabouModel
from maraboupy.MarabouRnnModel import MARABOU_TIMEOUT

large = 5000.0
small = 10 ** -4
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 5


def marabou_solve_negate_eq(query, debug=False, print_vars=False, return_vars=False):
    '''
    Run marabou solver
    :param query: query to execute
    :param debug: if True printing all of the query equations
    :return: True if UNSAT (no valid assignment), False otherwise
    '''
    verbose = 0
    if debug:
        query.dump()

    # print("{}: start query".format(str(datetime.now()).split(".")[0]), flush=True)
    vars1, stats1 = MarabouCore.solve(query, "", MARABOU_TIMEOUT, verbose)
    # print("{}: finish query".format(str(datetime.now()).split(".")[0]), flush=True)
    if stats1.hasTimedOut():
        print("Marabou has timed out")
        raise TimeoutError()
    if len(vars1) > 0:
        if print_vars:
            print("SAT")
            print(vars1[29])
        res = False
    else:
        # print("UNSAT")
        res = True

    if return_vars:
        # if len(vars1) > 0:
        #     print(vars1)
        return res, vars1
    else:
        return res


def negate_equation(eq):
    '''
    negates the equation
    :param eq: equation
    :return: new equation which is exactly (not eq)
    '''
    assert eq is not None
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
    if isinstance(loop_indices, list):
        loop_indices = [i for ls in loop_indices for i in ls]
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

    if isinstance(loop_indices[0], list):
        rnn_input_indices = [idx + 1 for ls in loop_indices for idx in ls]
        rnn_output_indices = [idx + 3 for ls in loop_indices for idx in ls]
    else:
        rnn_input_indices = [idx + 1 for idx in loop_indices]
        rnn_output_indices = [idx + 3 for idx in loop_indices]

    # equations for induction step
    if isinstance(invariant_eq, list):
        raise Exception

    induction_step = negate_equation(invariant_eq)

    # equations for induction base

    # make sure i == 0 (for induction base)
    loop_equations = []
    if isinstance(loop_indices[0], list):
        loop_indices = [i for ls in loop_indices for i in ls]
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

    induction_hypothesis = create_induction_hypothesis_from_invariant_eq() + [step_loop_eq]
    induction_step_equations = [induction_step]
    # induction_step_equations = induction_step + step_loop_eq

    return induction_base_equations, induction_step_equations, induction_hypothesis


def prove_invariant_multi(network, rnn_start_idxs, invariant_equations, return_vars=False):
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
    vars = []

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

    hypothesis_fail = False
    # TODO: DEBUG
    # marabou_result, cur_vars = marabou_solve_negate_eq(network, print_vars=False, return_vars=True)
    # if marabou_result:
    #     # UNSAT Conflict in the hypothesis
    #     assert False
    #     proved_invariants = [False] * len(proved_invariants)
    #     hypothesis_fail = True

    if not hypothesis_fail:
        for i, steq_eq_ls in enumerate(step_eq):
            for eq in steq_eq_ls:
                # eq.dump()
                network.addEquation(eq)

            marabou_result, cur_vars = marabou_solve_negate_eq(network, print_vars=True, return_vars=True)
            vars.append(cur_vars)
            # print("Querying for induction step: {}".format(marabou_result))
            # network.dump()

            if not marabou_result:
                # for eq in hypothesis_eq:
                #     network.removeEquation(eq)
                network.dump()
                # print("induction step fail, on invariant:", i)
                proved_invariants[i] = False
            else:
                proved_invariants[i] = True
                # print("induction step work, on invariant:", i)

            for eq in steq_eq_ls:
                network.removeEquation(eq)
    for eq in hypothesis_eq:
        network.removeEquation(eq)

    if return_vars:
        return proved_invariants, vars
    else:
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
        # TODO: I don't like this either
        ge_better = 1
        # ge_better = -1

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


def invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=False):
    '''
    Creates a function that can verify invariants accodring to the network and rnn indcies
    :param network: Marabou format of a network
    :param rnn_start_idxs: Indcies of the network where RNN cells start
    :param rnn_output_idxs: Output indcies of RNN cells in the network
    :return: A pointer to a function that given a list of equations checks if they stand or not
    '''

    def invariant_oracle(equations_to_verify):
        # if isinstance(rnn_start_idxs, list):
        #     invariant_results = []
        #     for start_idxs in rnn_start_idxs:
        #         invariant_results.append(prove_invariant_multi(network, start_idxs, equations_to_verify, return_vars=return_vars))
        #     return invariant_results
        # else:
        return prove_invariant_multi(network, rnn_start_idxs, equations_to_verify, return_vars=return_vars)

    return invariant_oracle


def property_oracle_generator(network, property_equations):
    def property_oracle(invariant_equations):

        for eq in invariant_equations:
            if eq is not None:
                network.addEquation(eq)

        # TODO: This is only for debug
        # before we prove the property, make sure the invariants does not contradict each other, expect SAT from marabou
        # network.dump()
        # assert not marabou_solve_negate_eq(network, False, False)

        for eq in property_equations:
            if eq is not None:
                network.addEquation(eq)
        res = marabou_solve_negate_eq(network, False, print_vars=True)
        # network.dump()
        if res:
            pass
        for eq in invariant_equations + property_equations:
            if eq is not None:
                network.removeEquation(eq)
        return res

    return property_oracle


def prove_multidim_property(rnnModel: RnnMarabouModel, property_equations, algorithm, return_alphas=False,
                            number_of_steps=5000, debug=False, return_queries_stats=False):
    network = rnnModel.network
    rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=None)
    add_loop_indices_equations(network, rnn_start_idxs)
    # invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=True)
    property_oracle = property_oracle_generator(network, property_equations)

    res = False
    unsat=  False
    invariant_times = []
    property_times = []
    step_times = []
    for i in range(number_of_steps):
        start_invariant = timer()
        invariant_results = []
        proved_equations = []

        for l in range(rnnModel.num_rnn_layers):
            start_step = timer()
            if hasattr(algorithm, 'support_multi_layer') and algorithm.support_multi_layer:
                rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=l)
                equations = algorithm.get_equations(layer_idx=l)
            else:
                rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(rnn_layer=None)
                equations = algorithm.get_equations()
            if equations is None:
                unsat = True
                break
            end_step = timer()
            step_times.append(end_step - start_step)
            invariant_oracle = invariant_oracle_generator(network, rnn_start_idxs, rnn_output_idxs, return_vars=True)
            layer_invariant_results, sat_vars = invariant_oracle(equations)
            invariant_results += layer_invariant_results
            if all(layer_invariant_results):
                if hasattr(algorithm, 'proved_invariant'):
                    algorithm.proved_invariant(l, equations)
                # When we prove layer l+1 we need to proved equations on layer l
                proved_equations += equations
                for eq in proved_equations:
                    network.addEquation(eq)
            else:
                # TODO: DELETE
                # Failed to prove on of the layers, we can  break the loop
                break
        end_invariant = timer()
        if unsat:
            res = False
            break
        for eq in proved_equations:
            network.removeEquation(eq)
        # print(invariant_results)
        invariant_times.append(end_invariant - start_invariant)
        if all(invariant_results):
            # print('proved an invariant: {}'.format(algorithm.get_alphas()))
            start_property = timer()
            prop_res = property_oracle(proved_equations)
            end_property = timer()
            property_times.append(end_property - start_property)
            if prop_res:
                print("proved property after {} iterations, using alphas: {}".format(i, algorithm.get_alphas()))
                res = True
                break
            else:
                start_step = timer()
                # If the property failed no need to pass which invariants passed (of course)
                if hasattr(algorithm, 'return_vars') and algorithm.return_vars:
                    algorithm.do_step(True, None, sat_vars)
                else:
                    algorithm.do_step(True, None)
                end_step = timer()
                step_times.append(end_step - start_step)
        else:
            start_step = timer()
            if hasattr(algorithm, 'return_vars') and algorithm.return_vars:
                # Invariant failed in gurobi based search, does not suppose to happen
                assert False, invariant_results
                # algorithm.do_step(False, invariant_results, sat_vars)
            else:
                algorithm.do_step(False, invariant_results)
            end_step = timer()
            step_times.append(end_step - start_step)

        #  print progress for debug
        if debug:
            if i > 0 and i % 300 == 0:
                print('iteration {} sum(alphas): {}, alphas: {}'.format(i, sum(algorithm.get_alphas()),
                                                                        algorithm.get_alphas()))

    if debug:
        if len(property_times) != len(invariant_times):
            print('What happed?')
        if len(property_times) > 0:
            print("did {} invariant queries that took on avg: {}, and {} property, that took: {} on avg".format(
                len(invariant_times), sum(invariant_times) / len(invariant_times), len(property_times),
                                      sum(property_times) / len(property_times)))
        else:
            avg_inv_time = sum(invariant_times) / len(invariant_times) if len(invariant_times) > 0 else 0
            print("did {} invariant queries that took on avg: {}, and {} property".format(
                len(invariant_times), avg_inv_time, len(property_times)))
    queries_stats = {}
    if return_queries_stats:
        safe_percentile = lambda func, x: func(x) if len(x) > 0 else 0
        queries_stats['property_times'] = {'avg': safe_percentile(np.mean, property_times),
                                           'median': safe_percentile(np.median, property_times), 'raw': property_times}
        queries_stats['invariant_times'] = {'avg': safe_percentile(np.mean, invariant_times),
                                            'median': safe_percentile(np.median, invariant_times),
                                            'raw': invariant_times}
        queries_stats['step_times'] = {'avg': safe_percentile(np.mean, step_times),
                                       'median': safe_percentile(np.median, step_times), 'raw': step_times}
        queries_stats['step_queries'] = len(step_times)
        queries_stats['property_queries'] = len(property_times)
        queries_stats['invariant_queries'] = len(invariant_times)
        queries_stats['number_of_updates'] = i + 1  # one based counting
    if not return_alphas:
        if not return_queries_stats:
            return res
        if return_queries_stats:
            return res, queries_stats
    else:
        if not return_queries_stats:
            return res, algorithm.get_alphas()
        if return_queries_stats:
            return res, algorithm.get_alphas(), queries_stats
