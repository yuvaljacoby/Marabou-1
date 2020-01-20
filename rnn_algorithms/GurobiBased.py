import random

from gurobipy import *

from maraboupy import MarabouCore
from maraboupy.MarabouRNNMultiDim import alpha_to_equation
from rnn_algorithms.Update_Strategy import Absolute_Step

sign = lambda x: 1 if x >= 0 else -1

SMALL = 0  # 10 ** -5
LARGE = 10 ** 5

# def relative_step(threshold=0.2, relative_step_size=0.3, init_after_threshold=0.5):
#     '''
#     Create a function that is doing a relative step in the form:
#     if alpha <= threshold:
#         return init_after_threshold * direction
#     else:
#         return alphas + step
#     where step is: direction * alpha * relative_step_size * sign(alpha)
#     :return: function pointer that given alpha and direction returning the new alpha
#     '''
#
#     def do_relative_step(alpha, direction):
#         if abs(alpha) > threshold:
#             return alpha + (
#                     direction * alpha * relative_step_size * sign(alpha))  # do step size 0.3 to the next direction
#         else:
#             return init_after_threshold * direction
#
#     return do_relative_step
#
#
# def absolute_step(step_size=0.1):
#     '''
#      Create a function that is doing an absolute step in the form:
#      alpha + (step_size * direction)
#      :return: function pointer that given alpha and direction returning the new alpha
#      '''
#
#     def do_relative_step(alpha, direction):
#         return alpha + (direction * step_size)
#
#     return do_relative_step


class RandomAlphaSGDOneSideBound:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, inv_type, xlim, w_in, w_h, b, n_iterations,
                 constraint_type='max', update_strategy=Absolute_Step()):
        '''
        :param initial_values: list of values (min / max)
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''
        self.rnn_start_idxs = rnn_start_idxs
        self.rnn_output_idxs = rnn_output_idxs
        self.initial_values = initial_values
        self.inv_type = inv_type
        self.next_idx_step = 0
        self.prev_alpha = None
        self.prev_idx = None
        self.update_strategy = update_strategy
        self.w_in = w_in
        self.w_h = w_h
        self.b = b
        self.xlim = xlim
        self.constraint_type = constraint_type
        self.n_iterations = n_iterations
        alpha_initial_value = build_gurobi_query(xlim, self.w_in, self.w_h, self.b, self.n_iterations,
                                                 constraint_type=self.constraint_type)
        if alpha_initial_value is None:
            alpha_initial_value = 0
        # initial_min = build_gurobi_query(xlim[0], w_in, w_h, b, 50, constraint_type='min')

        self.first_call = True
        assert inv_type in [MarabouCore.Equation.LE, MarabouCore.Equation.GE]

        if not hasattr(alpha_initial_value, "__len__"):
            self.alphas = [alpha_initial_value] * len(initial_values)
        else:
            assert len(alpha_initial_value) == len(initial_values)
            self.alphas = alpha_initial_value

        self.equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self.update_equation(i)

    def update_next_idx_step(self):
        self.next_idx_step = random.randint(0, len(self.alphas) - 1)
        return self.next_idx_step

    def do_step(self, strengthen=True):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''
        self.first_call = False
        if strengthen:
            new_min_sum = sum(self.alphas) * 0.5
            direction = -1
        else:
            new_min_sum = sum(self.alphas) * 2
            direction = 1

        # i = self.next_idx_step
        # self.prev_alpha = self.alphas[self.next_idx_step]
        # self.prev_idx = self.next_idx_step
        # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        new_alphas = build_gurobi_query(self.xlim, self.w_in, self.w_h, self.b, self.n_iterations,
                                         constraint_type=self.constraint_type, alphas_sum=new_min_sum)
        if new_alphas == None:
        #     No fesabile solution, maybe to much over approximation, imporve at random
            i = self.next_idx_step
            self.update_next_idx_step()
            self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        else:
            self.alphas = new_alphas

        for i in range(len(self.alphas)):
            self.update_equation(i)

        i = self.next_idx_step
        self.update_next_idx_step()
        # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        return self.equations

    def update_equation(self, idx):
        self.equations[idx] = alpha_to_equation(self.rnn_start_idxs[idx], self.rnn_output_idxs[idx],
                                                self.initial_values[idx], self.alphas[idx], self.inv_type)

    def revert_last_step(self):
        if self.prev_idx is not None:
            self.alphas[self.prev_idx] = self.prev_alpha
            self.update_equation(self.prev_idx)

    def get_equations(self):
        return self.equations

    def get_alphas(self):
        return self.alphas


def build_gurobi_query(prev_layer, w_in, w_h, b, tmax, constraint_type='max', alphas_sum=0):
    # TODO: Need to extract the counter example
    gmodel = Model("test")
    # add the alphas variables
    alphas_vars = []
    in_vars = []
    obj = LinExpr()
    for i in range(w_h.shape[0]):
        alphas_vars.append(gmodel.addVar(lb=SMALL, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha{}".format(i)))
        obj += alphas_vars[-1]
    for i in range(len(prev_layer)):
        in_vars.append(gmodel.addVar(lb=prev_layer[i][0], ub=prev_layer[i][1], vtype=GRB.CONTINUOUS, name="in{}".format(i)))

    if constraint_type == 'max':
        gmodel.setObjective(obj, GRB.MINIMIZE)
    else:
        gmodel.setObjective(obj, GRB.MAXIMIZE)
    gmodel.addConstr(obj >= alphas_sum, "minimum_alpha_sum")

    # Add the constraints
    for t in range(1, tmax):
        for i in range(w_h.shape[0]):
            cond = LinExpr()
            for j in range(w_h.shape[0]):
                cond += alphas_vars[j] * (t - 1) * w_h[j,i]

            for j in range(len(in_vars)):
                cond += in_vars[j] * w_in[j,i]
            cond += b[i]
            if constraint_type == 'max':
                gmodel.addConstr(alphas_vars[i] * t >= cond, "alpha{}_t{}".format(i, t))
            else:
                gmodel.addConstr(alphas_vars[i] * t <= cond, "alpha{}_t{}".format(i, t))
            # gmodel.addConstr(alphas_vars[i] >= SMALL, "alpha{}_positive".format(i, t))

    gmodel.optimize()
    # gmodel.write("temp.lp")

    if gmodel.status == GRB.CUTOFF or gmodel.status == GRB.INFEASIBLE:
        print("INFEASIBLE sum_alpahs = {} constraint_type={}".format(alphas_sum, constraint_type))
        return None
    print("FEASIBLE sum_alpahs = {} constraint_type={}".format(alphas_sum, constraint_type))
    for v in gmodel.getVars():
        print(v.varName, v.x)

    return [a.x for a in alphas_vars]


class AlphasGurobiBased:
    def __init__(self, rnnModel, xlim, update_strategy_ptr=Absolute_Step):
        '''
        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''

        self.next_is_max = True

        w_in, w_h, b = rnnModel.get_weights()[0]

        self.update_strategy = update_strategy_ptr()
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs()
        initial_values = rnnModel.get_rnn_min_max_value_one_iteration(xlim)

        # The initial values are opposite to the intuition, for LE we use max_value
        self.min_invariants = RandomAlphaSGDOneSideBound(initial_values[1], rnn_start_idxs, rnn_output_idxs,
                                                         MarabouCore.Equation.LE,xlim, w_in, w_h, b,
                                                         rnnModel.n_iterations, 'max',
                                                         self.update_strategy)
        self.max_invariants = RandomAlphaSGDOneSideBound(initial_values[0], rnn_start_idxs, rnn_output_idxs,
                                                         MarabouCore.Equation.GE,xlim, w_in, w_h, b,
                                                         rnnModel.n_iterations, 'min',
                                                         self.update_strategy)
        self.last_fail = None
        self.alpha_history = []

    def name(self):
        return 'iterate_sgd_init{}_step{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)

    def do_step(self, strengthen=True, invariants_results=[]):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''

        if invariants_results != [] and invariants_results is not None:
            min_invariants_results = invariants_results[len(self.min_invariants.alphas):]
            max_invariants_results = invariants_results[:len(self.min_invariants.alphas)]
            # If we all invariants from above or bottom are done do step in the other
            # if all(min_invariants_results):
            #     self.next_is_max = True
            # elif all(max_invariants_results):
            #     self.next_is_max = False

        # TODO: If this condition is true it means the last step we did was not good, and we can decide what to do next
        #  (for example revert, and once passing all directions do a big step)
        if self.last_fail == strengthen:
            pass
        self.last_fail = strengthen

        if self.next_is_max:
            res = self.min_invariants.get_equations() + self.max_invariants.do_step(strengthen)
        else:
            res = self.min_invariants.do_step(strengthen) + self.max_invariants.get_equations()

        self.next_is_max = random.randint(0, 1)
        self.alpha_history.append(self.get_alphas())
        return res

    def revert_last_step(self):
        '''
        If last step did not work, call this to revert it (and then we still have a valid invariant)
        '''
        if self.next_is_max:
            self.min_invariants.revert_last_step()
        else:
            self.max_invariants.revert_last_step()

    def get_equations(self):
        return self.max_invariants.get_equations() + self.min_invariants.get_equations()

    def get_alphas(self):
        return self.max_invariants.get_alphas() + self.min_invariants.get_alphas()
