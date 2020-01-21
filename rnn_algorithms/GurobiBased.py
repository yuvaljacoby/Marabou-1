import random

import numpy as np
from gurobipy import *

from maraboupy import MarabouCore
from maraboupy.MarabouRNNMultiDim import alpha_to_equation
from rnn_algorithms.Update_Strategy import Absolute_Step

sign = lambda x: 1 if x >= 0 else -1

SMALL = 0  # 10 ** -5
LARGE = 10 ** 5


# class RandomAlphaSGDOneSideBound:
#     def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, inv_type, xlim, w_in, w_h, b, n_iterations,
#                  constraint_type='max', update_strategy=Absolute_Step()):
#         '''
#         :param initial_values: list of values (min / max)
#         :param rnn_start_idxs:
#         :param rnn_output_idxs:
#         '''
#         self.rnn_start_idxs = rnn_start_idxs
#         self.rnn_output_idxs = rnn_output_idxs
#         self.initial_values = initial_values
#         self.inv_type = inv_type
#         self.next_idx_step = 0
#         self.prev_alpha = None
#         self.prev_idx = None
#         self.update_strategy = update_strategy
#         self.w_in = w_in
#         self.w_h = w_h
#         self.b = b
#         self.xlim = xlim
#         self.constraint_type = constraint_type
#         self.n_iterations = n_iterations
#         alpha_initial_value = build_gurobi_query(xlim, self.w_in, self.w_h, self.b, self.n_iterations,
#                                                  constraint_type=self.constraint_type)
#         if alpha_initial_value is None:
#             alpha_initial_value = 0
#         # initial_min = build_gurobi_query(xlim[0], w_in, w_h, b, 50, constraint_type='min')
#
#         self.first_call = True
#         assert inv_type in [MarabouCore.Equation.LE, MarabouCore.Equation.GE]
#
#         if not hasattr(alpha_initial_value, "__len__"):
#             self.alphas = [alpha_initial_value] * len(initial_values)
#         else:
#             assert len(alpha_initial_value) == len(initial_values)
#             self.alphas = alpha_initial_value
#
#         self.equations = [None] * len(self.alphas)
#         for i in range(len(self.alphas)):
#             self.update_equation(i)
#
#     def update_next_idx_step(self):
#         self.next_idx_step = random.randint(0, len(self.alphas) - 1)
#         return self.next_idx_step
#
#     def do_step(self, strengthen=True):
#         '''
#         do a step in the one of the alphas
#         :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
#         :return list of invariant equations (that still needs to be proven)
#         '''
#         self.first_call = False
#         if strengthen:
#             new_min_sum = sum(self.alphas) * 0.5
#             direction = -1
#         else:
#             new_min_sum = sum(self.alphas) * 2
#             direction = 1
#
#         # i = self.next_idx_step
#         # self.prev_alpha = self.alphas[self.next_idx_step]
#         # self.prev_idx = self.next_idx_step
#         # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
#         new_alphas = build_gurobi_query(self.xlim, self.w_in, self.w_h, self.b, self.n_iterations,
#                                         constraint_type=self.constraint_type, alphas_sum=new_min_sum)
#         if new_alphas == None:
#             #     No fesabile solution, maybe to much over approximation, imporve at random
#             i = self.next_idx_step
#             self.update_next_idx_step()
#             self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
#         else:
#             self.alphas = new_alphas
#
#         for i in range(len(self.alphas)):
#             self.update_equation(i)
#
#         i = self.next_idx_step
#         self.update_next_idx_step()
#         # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
#         return self.equations
#
#     def update_equation(self, idx):
#         self.equations[idx] = alpha_to_equation(self.rnn_start_idxs[idx], self.rnn_output_idxs[idx],
#                                                 self.initial_values[idx], self.alphas[idx], self.inv_type)
#
#     def revert_last_step(self):
#         if self.prev_idx is not None:
#             self.alphas[self.prev_idx] = self.prev_alpha
#             self.update_equation(self.prev_idx)
#
#     def get_equations(self):
#         return self.equations
#
#     def get_alphas(self):
#         return self.alphas


class AlphasGurobiBased:
    def __init__(self, rnnModel, xlim, update_strategy_ptr=Absolute_Step):
        '''
        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''

        self.return_vars = True
        self.next_is_max = True
        self.xlim = xlim
        self.w_in, self.w_h, self.b = rnnModel.get_weights()[0]
        self.n_iterations = rnnModel.n_iterations

        self.update_strategy = update_strategy_ptr()
        self.same_step_counter = 0
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs()
        self.rnn_output_idxs = rnn_output_idxs
        self.rnn_start_idxs = rnn_start_idxs + rnn_start_idxs
        self.rnn_output_idxs_double = rnn_output_idxs + rnn_output_idxs
        initial_values = rnnModel.get_rnn_min_max_value_one_iteration(xlim)

        self.initial_values = initial_values  # [1] + initial_values[0]
        self.inv_type = [MarabouCore.Equation.LE] * len(initial_values[1]) + [MarabouCore.Equation.GE] * len(
            initial_values[0])
        self.alphas = self.build_gurobi_query()
        if self.alphas is None:
            self.alphas = [0] * len(self.inv_type)
        assert len(self.alphas) == len(self.inv_type)
        assert len(self.alphas) == (len(self.initial_values[0]) + len(self.initial_values[1]))
        assert len(self.alphas) == len(self.rnn_output_idxs_double)

        self.equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self.update_equation(i)

        # The initial values are opposite to the intuition, for LE we use max_value
        # self.min_invariants = RandomAlphaSGDOneSideBound(initial_values[1], rnn_start_idxs, rnn_output_idxs,
        #                                                  MarabouCore.Equation.LE,xlim, w_in, w_h, b,
        #                                                  rnnModel.n_iterations, 'max',
        #                                                  self.update_strategy)
        # self.max_invariants = RandomAlphaSGDOneSideBound(initial_values[0], rnn_start_idxs, rnn_output_idxs,
        #                                                  MarabouCore.Equation.GE,xlim, w_in, w_h, b,
        #                                                  rnnModel.n_iterations, 'min',
        #                                                  self.update_strategy)
        self.last_fail = None
        self.alpha_history = []

    def build_gurobi_query(self, alphas_sum=None, counter_examples=([], []), hyptoesis_ce=([], [])):
        # TODO: Need to extract the counter example
        gmodel = Model("test")
        # add the alphas variables
        alphas_u = []
        alphas_l = []
        in_vars = []
        obj = LinExpr()
        for i in range(self.w_h.shape[0]):
            alphas_l.append(gmodel.addVar(lb=-LARGE, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_l_{}".format(i)))
            obj += -2 * alphas_l[-1]

        for i in range(self.w_h.shape[0]):
            alphas_u.append(gmodel.addVar(lb=-LARGE, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_u_{}".format(i)))
            obj += alphas_u[-1]
            gmodel.addConstr(alphas_u[i] >= alphas_l[i], "alpha_l{}<alpha_u{}".format(i, i))

        for i in range(len(self.xlim)):
            in_vars.append(
                gmodel.addVar(lb=self.xlim[i][0], ub=self.xlim[i][1], vtype=GRB.CONTINUOUS, name="in{}".format(i)))

        gmodel.setObjective(obj, GRB.MINIMIZE)
        if alphas_sum:
            gmodel.addConstr(obj >= alphas_sum, "minimum_alpha_sum")

        # Add the constraints
        upper_bounds = []
        delta_u = []
        lower_bounds = []
        delta_l = []


        for i in range(self.w_h.shape[0]):
            for t in range(1, self.n_iterations):
                # Conditions for the over approximation of the memory cell at every time point
                cond_u = LinExpr()
                cond_l = LinExpr()
                cond_x = LinExpr()
                # Result of the relu of the over approximation of the memory cell
                upper_bounds.append(gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="ub{}_t_{}".format(i, t)))
                lower_bounds.append(gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="lb{}_t_{}".format(i, t)))
                delta_u.append(gmodel.addVar(vtype=GRB.BINARY))
                delta_l.append(gmodel.addVar(vtype=GRB.BINARY))
                for j in range(self.w_h.shape[0]):
                    # For the min invariants we want max value, and the other way around
                    # if self.w_h[j, i] > 0:
                    #     cond_l += (alphas_l[j] + self.initial_values[1][i]) * (t - 1) * self.w_h[j, i]
                    #     cond_u += (alphas_u[j] + self.initial_values[0][i]) * (t - 1) * self.w_h[j, i]
                    # else:
                    #     cond_l += (alphas_u[j] + self.initial_values[0][i]) * (t - 1) * self.w_h[j, i]
                    #     cond_u += (alphas_l[j] + self.initial_values[1][i]) * (t - 1) * self.w_h[j, i]
                    if self.w_h[j, i] > 0:
                        cond_l += (alphas_l[j] + self.initial_values[0][j]) * (t - 1) * self.w_h[j, i]
                        cond_u += (alphas_u[j] + self.initial_values[1][j]) * (t - 1) * self.w_h[j, i]
                    else:
                        cond_l += (alphas_u[j] + self.initial_values[1][j]) * (t - 1) * self.w_h[j, i]
                        cond_u += (alphas_l[j] + self.initial_values[0][j]) * (t - 1) * self.w_h[j, i]

                for j in range(len(in_vars)):
                    cond_x += in_vars[j] * self.w_in[j, i]

                cond_x += self.b[i]
                cond_u += cond_x
                cond_l += cond_x

                gmodel.addConstr(upper_bounds[-1] >= cond_u)
                gmodel.addConstr(upper_bounds[-1] <= cond_u + LARGE * delta_u[-1])
                gmodel.addConstr(upper_bounds[-1] <= LARGE * (1 - delta_u[-1]))

                gmodel.addConstr(lower_bounds[-1] >= cond_l)
                gmodel.addConstr(lower_bounds[-1] <= cond_l + LARGE * delta_l[-1])
                gmodel.addConstr(lower_bounds[-1] <= LARGE * (1 - delta_l[-1]))

                gmodel.addConstr(alphas_u[i] * t >= upper_bounds[-1], "alpha_u{}_t{}".format(i, t))
                gmodel.addConstr(alphas_l[i] * t <= lower_bounds[-1], "alpha_l{}_t{}".format(i, t))

                # gmodel.addConstr(alphas_vars[i] >= SMALL, "alpha{}_positive".format(i, t))

        outputs, time = counter_examples
        for i in range(len(time)):
            for j in range(len(alphas_u)):
                gmodel.addConstr(alphas_u[j] * time[i] >= outputs[i][j])
                gmodel.addConstr(alphas_l[j] * time[i] <= outputs[i][j])


        # hy_outputs, _ = hyptoesis_ce
        # for i in range(len(time)):
        #     for j in range(len(alphas_u)):
        #         gmodel.addConstr(alphas_u[j] * (time[i] - 1) >= hy_outputs[i][j])
        #         gmodel.addConstr(alphas_l[j] * (time[i] - 1) <= hy_outputs[i][j])
        gmodel.setParam('OutputFlag', False)
        gmodel.optimize()
        gmodel.write("temp.lp")

        if gmodel.status == GRB.CUTOFF or gmodel.status == GRB.INFEASIBLE:
            print("INFEASIBLE sum_alpahs = {} constraint_type={}".format(alphas_sum, ''))
            exit(0)
            return None
        print("FEASIBLE sum_alpahs = {}".format(alphas_sum))

        # for v in gmodel.getVars():
        for v in alphas_l + alphas_u:
            print(v.varName, v.x)

        return [a.x for a in alphas_l] + [a.x for a in alphas_u]

    def update_equation(self, idx):

        # Change the formal of initial values, first those for LE equations next to GE
        initial_values = self.initial_values[1] + self.initial_values[0]
        self.equations[idx] = alpha_to_equation(self.rnn_start_idxs[idx], self.rnn_output_idxs_double[idx],
                                                initial_values[idx], self.alphas[idx], self.inv_type[idx])

    def name(self):
        return 'gurobi_based_{}_{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)

    def extract_equation_from_counter_example(self, counter_examples=[{}]):
        '''

        :param counter_examples: Array of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of rnn_output values (as number of alpha_u), times the assingment for t  (len(times) == len(outputs)
        '''
        outputs = []
        times = []
        for counter_example in counter_examples:
            if counter_example == {}:
                continue
            # We need to extract the time, and the values of all output indcies
            # Next create alpha_u >= time * output \land alpha_l \le time * output
            outputs.append([counter_example[i] for i in self.rnn_output_idxs])
            assert counter_example[self.rnn_start_idxs[0]] == counter_example[self.rnn_start_idxs[1]]
            times.append(counter_example[self.rnn_start_idxs[0]])
        return outputs, times

    def extract_hyptoesis_from_counter_example(self, counter_examples=[{}]):
        '''

        :param counter_examples: Array of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of memory cell values (as number of alpha_u), times the assingment for t  (len(times) == len(outputs)
        '''
        outputs = []
        times = []
        for counter_example in counter_examples:
            if counter_example == {}:
                continue
            # We need to extract the time, and the values of all output indcies
            # Next create alpha_u >= time * output \land alpha_l \le time * output
            outputs.append([counter_example[i - 1] for i in self.rnn_output_idxs])
            assert counter_example[self.rnn_start_idxs[0]] == counter_example[self.rnn_start_idxs[1]]
            times.append(counter_example[self.rnn_start_idxs[0]])
        return outputs, times

    def do_step(self, strengthen=True, invariants_results=[], sat_vars=None):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''

        if invariants_results != [] and invariants_results is not None:
            pass
            # If we all invariants from above or bottom are done do step in the other
            # if all(min_invariants_results):
            #     self.next_is_max = True
            # elif all(max_invariants_results):
            #     self.next_is_max = False

        # TODO: If this condition is true it means the last step we did was not good, and we can decide what to do next
        #  (for example revert, and once passing all directions do a big step)
        if self.last_fail == strengthen:
            self.same_step_counter += 1
        else:
            self.same_step_counter = 0

        self.last_fail = strengthen

        if strengthen:
            new_min_sum = sum(self.alphas) * np.random.normal(0.5, 0.1)
            direction = -1
        else:
            new_min_sum = sum(self.alphas) * 2
            direction = 1

            # i = self.next_idx_step
            # self.prev_alpha = self.alphas[self.next_idx_step]
            # self.prev_idx = self.next_idx_step
            # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        i = random.randint(0, len(self.alphas) - 1)
        if self.same_step_counter > 10:
            # do a random (big)  step try to move else where
            print("random step")
            new_alphas = self.alphas
            # new_alphas[i] = new_alphas[i] + (direction * new_alphas[i] * 0.1 * sign(new_alphas[i]))
            new_alphas[i] = new_alphas[i] + (direction * 10)
            # self.same_step_counter = math.floor(self.same_step_counter / 2)
        else:
            counter_examples = self.extract_equation_from_counter_example(sat_vars)
            hyptoesis_ce = self.extract_hyptoesis_from_counter_example(sat_vars)
            new_alphas = self.build_gurobi_query(alphas_sum=new_min_sum, counter_examples=counter_examples, hyptoesis_ce=hyptoesis_ce)

        if new_alphas is None:
            #     No fesabile solution, maybe to much over approximation, imporve at random
            # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
            self.alphas[i] = self.alphas[i] + (direction * 10)
        else:
            self.alphas = new_alphas

        for i in range(len(self.alphas)):
            self.update_equation(i)
        # print(self.alphas)
        self.alpha_history.append(self.get_alphas())
        # self.update_next_idx_step()
        # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        return self.equations

        # if self.next_is_max:
        #     res = self.min_invariants.get_equations() + self.max_invariants.do_step(strengthen)
        # else:
        #     res = self.min_invariants.do_step(strengthen) + self.max_invariants.get_equations()

        # self.next_is_max = random.randint(0, 1)

        return res

    def revert_last_step(self):
        '''
        If last step did not work, call this to revert it (and then we still have a valid invariant)
        '''
        return

    def get_equations(self):
        return self.equations

    def get_alphas(self):
        return self.alphas
