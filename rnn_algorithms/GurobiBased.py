import random

random.seed(10)
import numpy as np
from gurobipy import *

from maraboupy import MarabouCore
from maraboupy.MarabouRNNMultiDim import alpha_to_equation
from rnn_algorithms.Update_Strategy import Absolute_Step

sign = lambda x: 1 if x >= 0 else -1

SMALL = 10 ** -4
LARGE = 10 ** 5

RANDOM_THRESHOLD = 200  #
PRINT_GUROBI = False



class AlphasGurobiBasedMultiLayer:
    # Alpha Search Algorithm for multilayer recurrent, assume the recurrent layers are one following the other
    # we need this assumptions in proved_invariant method, if we don't have it we need to extract bounds in another way
    # not sure it is even possible to create multi recurrent layer NOT in a row
    def __init__(self, rnnModel, xlim, update_strategy_ptr=Absolute_Step, random_threshold=RANDOM_THRESHOLD,
                 use_relu=True, use_counter_example=False, add_alpha_constraint=False):
        '''

        :param rnnModel: multi layer rnn model (MarabouRnnModel class)
        :param xlim: limit on the input layer
        :param update_strategy_ptr: not used
        :param random_threshold: not used
        :param use_relu: if to encode relus in gurobi
        :param use_counter_example: if property is not implied wheter to use counter_example to  create better alphas
        :param add_alpha_constraint: if detecting a loop (proved invariant and then proving the same invariant) wheter to add a demand that every alpha will change in epsilon
        '''
        self.return_vars = True
        self.support_multi_layer = True
        self.num_layers = len(rnnModel.rnn_out_idx)
        self.alphas_algorithm_per_layer = []
        self.alpha_history = None
        for i in range(self.num_layers):
            if i == 0:
                prev_layer_lim = xlim
            else:
                prev_layer_lim = None # [(-LARGE, LARGE) for _ in range(len(xlim))]
            self.alphas_algorithm_per_layer.append(
                AlphasGurobiBased(rnnModel, prev_layer_lim, update_strategy_ptr, random_threshold, use_relu,
                                  use_counter_example, add_alpha_constraint, layer_idx=i))

    def proved_invariant(self, layer_idx=0, equations=None):
        # Proved invariant on layer_idx --> we can update bounds for layer_idx +1
        alphas = self.alphas_algorithm_per_layer[layer_idx].get_alphas()
        betas = self.alphas_algorithm_per_layer[layer_idx].initial_values
        assert len(alphas) % 2 == 0
        alphas_l = alphas[:len(alphas) // 2]
        alphas_u = alphas[len(alphas) // 2:]
        if len(self.alphas_algorithm_per_layer) > layer_idx + 1:
            self.alphas_algorithm_per_layer[layer_idx + 1].update_xlim(alphas_l, alphas_u, betas)

    def do_step(self, strengthen=True, invariants_results=[], sat_vars=None, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].do_step(strengthen, invariants_results, sat_vars)

    def get_equations(self, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].get_equations()

    def get_alphas(self, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].get_alphas()


class AlphasGurobiBased:
    def __init__(self, rnnModel, xlim, update_strategy_ptr=Absolute_Step, random_threshold=RANDOM_THRESHOLD,
                 use_relu=True, use_counter_example=False, add_alpha_constraint=False, layer_idx=0):
        '''

        :param rnnModel:
        :param xlim:
        :param update_strategy_ptr:
        :param random_threshold: If more then this thresholds steps get the same result doing random step
        '''
        self.w_in, self.w_h, self.b = rnnModel.get_weights()[layer_idx]
        self.rnnModel = rnnModel
        self.dim = self.w_h.shape[0]
        self.add_alpha_constraint = add_alpha_constraint
        self.use_counter_example = use_counter_example
        self.use_relu = use_relu
        self.return_vars = True
        self.is_infesiable = False
        self.xlim = xlim
        self.xlim = xlim
        self.initial_values = None
        self.random_threshold = random_threshold
        self.n_iterations = rnnModel.n_iterations
        self.layer_idx = layer_idx
        self.prev_layer_beta = [None] * self.dim

        # initalize alphas to -infty +infty
        self.alphas = [-LARGE] * self.dim + [LARGE] * self.dim
        self.update_strategy = update_strategy_ptr()
        self.same_step_counter = 0
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(layer_idx)
        self.rnn_output_idxs = rnn_output_idxs
        self.rnn_start_idxs = rnn_start_idxs + rnn_start_idxs
        self.rnn_output_idxs_double = rnn_output_idxs + rnn_output_idxs
        self.is_time_limit = False

        # self.inv_type = [MarabouCore.Equation.LE] * len(initial_values[1]) + [MarabouCore.Equation.GE] * len(initial_values[0])
        self.inv_type = [MarabouCore.Equation.GE] * self.dim + [MarabouCore.Equation.LE] * self.dim
        # if self.alphas is None:
        #     self.alphas = [0] * len(self.inv_type)
        assert len(self.alphas) == (2 * self.dim)
        assert len(self.alphas) == len(self.inv_type)
        assert len(self.alphas) == len(self.rnn_output_idxs_double)

        self.equations = [None] * self.dim * 2
        if xlim is not None:
            self.update_xlim([x[0] for x in xlim], [x[1] for x in xlim])

        self.last_fail = None
        self.alpha_history = []

    def update_xlim(self, lower_bound, upper_bound, beta=None):
        '''
        Update of the xlim, if it is from an invariant then lower_bound and upper_bound are time dependent and beta is not
        (i.e. lower_bound * t + beta <= V <= upper_bound*t + beta where V is the neuron value
        :param lower_bound: length self.dim of lower bounds
        :param upper_bound: length self.dim of upper bounds
        :param beta: length self.dim of scalars
        :return: 
        '''
        assert len(lower_bound) == len(upper_bound)

        xlim = []
        for l, u in zip(lower_bound, upper_bound):
            xlim.append((l, u))
        # assert len(xlim) == len(self.xlim)
        if beta is not None:
            self.is_time_limit = True
            assert len(beta[0]) == len(lower_bound)
            assert len(beta[1]) == len(lower_bound)
            self.prev_layer_beta = beta
            # for i in range(len(xlim)):
            #     # Need to do the same for beta[0], in cases I saw it is always zero
            #     if xlim[i][1] + beta[1][i] > xlim[i][1]:
            #         xlim[i] = (xlim[i][0], xlim[i][1])

        self.xlim = xlim

        initial_values = self.rnnModel.get_rnn_min_max_value_one_iteration(xlim, layer_idx=self.layer_idx, prev_layer_beta=beta)
        initial_values = ([0] * len(initial_values[0]), initial_values[1])
        self.initial_values = initial_values  # [1] + initial_values[0]
        self.alphas = self.do_gurobi_step(strengthen=True)

        for i in range(len(self.alphas)):
            self.update_equation(i)


    def do_random_step(self, strengthen):
        if strengthen:
            direction = -1
        else:
            direction = 1

        # self.update_strategy.counter = self.same_step_counter
        i = np.random.randint(0, len(self.alphas))

        # self.prev_alpha = self.alphas[self.next_idx_step]
        # self.prev_idx = self.next_idx_step
        self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction, self.same_step_counter)

        self.update_equation(i)
        # self.update_next_idx_step()
        return self.equations

    def do_gurobi_step(self, strengthen, alphas_sum=None, counter_examples=([], []), hyptoesis_ce=([], []),
                       loop_detected=False, previous_alphas=None):
        # Note that if we someday remove constraints and not only add we need to change this
        if self.is_infesiable:
            return None

        gmodel = Model("test")
        # add the alphas variables
        alphas_u = []
        alphas_l = []
        obj = LinExpr()
        for i in range(self.w_h.shape[0]):
            alphas_l.append(gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_l_{}".format(i)))
            obj += -alphas_l[-1]
            # gmodel.addConstr(alphas_l[i] >= 0, "alpha_l{}>=0".format(i, i))

        for i in range(self.w_h.shape[0]):
            alphas_u.append(gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_u_{}".format(i)))
            obj += alphas_u[-1]
            gmodel.addConstr(alphas_u[i] >= SMALL + alphas_l[i], "alpha_l{}<alpha_u{}".format(i, i))

        gmodel.setObjective(obj, GRB.MINIMIZE)
        # if alphas_sum:
        #     gmodel.addConstr(obj >= alphas_sum, "minimum_alpha_sum")

        # Add the constraints
        cond_u_f = []
        delta_u = []
        cond_l_f = []
        delta_l = []


        for i in range(self.w_h.shape[0]):
            # TODO: Should t start from zero or 1? start from zero for now
            for t in range(self.n_iterations):
                # Conditions for the over approximation of the memory cell at every time point
                cond_u = LinExpr()
                cond_l = LinExpr()
                cond_x_u = 0 #LinExpr()
                cond_x_l = 0 #LinExpr()
                for j in range(self.w_h.shape[0]):
                    if self.w_h[i, j] > 0:
                        cond_l += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]
                        cond_u += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
                    else:
                        cond_l += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
                        cond_u += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]

                for j in range(len(self.xlim)):
                    if self.prev_layer_beta[0] is not None:
                        # Not first RNN layer, previous layer bound on the memory unit is in  alpha*time + beta
                        # In this case we know xlim > 0

                        if self.w_in[j, i] >= 0:
                            cond_x_u += (self.xlim[j][1] * (t+1) + self.prev_layer_beta[1][j]) * self.w_in[j, i]
                            cond_x_l += (self.xlim[j][0] * (t+1) + self.prev_layer_beta[0][j]) * self.w_in[j, i]

                        else:
                            cond_x_u += (self.xlim[j][0] * (t+1) + self.prev_layer_beta[0][j]) * self.w_in[j, i]
                            cond_x_l += (self.xlim[j][1] * (t+1) + self.prev_layer_beta[1][j]) * self.w_in[j, i]
                    else:
                        # if self.xlim[j][1] < 0 or self.xlim[j][0] < 0:
                        #     print(self.xlim[j][1], self.xlim[j][0])
                        v1 = self.xlim[j][1] * self.w_in[j, i]
                        v2 = self.xlim[j][0] * self.w_in[j, i]
                        if v1 > v2:
                            cond_x_u += v1
                            cond_x_l += v2
                        else:
                            cond_x_u += v2
                            cond_x_l += v1

                    # TODO: I don't like this
                    cond_x_l = min(cond_x_l, 0)


                cond_u += cond_x_u + self.b[i]  # + SMALL
                cond_l += cond_x_l + self.b[i]  # - SMALL

                if self.use_relu:
                    # Result of the relu of the over approximation of the memory cell
                    cond_u_f.append(
                        gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="ub{}_t_{}".format(i, t)))
                    cond_l_f.append(
                        gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="lb{}_t_{}".format(i, t)))
                    delta_u.append(gmodel.addVar(vtype=GRB.BINARY))
                    delta_l.append(gmodel.addVar(vtype=GRB.BINARY))

                    gmodel.addConstr(cond_u_f[-1] >= cond_u, "cond_u_relu0_i{}_t{}".format(i, t))
                    gmodel.addConstr(cond_u_f[-1] <= cond_u + LARGE * delta_u[-1], "cond_u_relu1_i{}_t{}".format(i, t))
                    gmodel.addConstr(cond_u_f[-1] <= LARGE * (1 - delta_u[-1]), "cond_u_relu2_i{}_t{}".format(i, t))

                    gmodel.addConstr(cond_l_f[-1] >= cond_l, "cond_l_relu0_i{}_t{}".format(i, t))
                    gmodel.addConstr(cond_l_f[-1] <= cond_l + LARGE * delta_l[-1], "cond_l_relu1_i{}_t{}".format(i, t))
                    gmodel.addConstr(cond_l_f[-1] <= LARGE * (1 - delta_l[-1]), "cond_l_relu2_i{}_t{}".format(i, t))

                    gmodel.addConstr((alphas_u[i]) * (t + 1) >= cond_u_f[-1], "alpha_u{}_t{}".format(i, t))
                    gmodel.addConstr((alphas_l[i]) * (t + 1) <= cond_l_f[-1], "alpha_l{}_t{}".format(i, t))
                else:
                    gmodel.addConstr(alphas_u[i] * (t + 1) >= cond_u, "alpha_u{}_t{}".format(i, t))
                    gmodel.addConstr(alphas_l[i] * (t + 1) <= cond_l, "alpha_l{}_t{}".format(i, t))

        if self.use_counter_example:
            if not strengthen:
                # Invariant failed, does not suppose to happen
                assert False
                    # if t is not None:
                    #     if i < len(time) / 2:
                    #         gmodel.addConstr(alphas_l[i] * t <= outputs[i], "ce_alpha_l")
                    #     else:
                    #         idx = i - len(alphas_u)
                    #         gmodel.addConstr(alphas_u[idx] * t >= outputs[i], 'ce_alpha_u')
            if strengthen and previous_alphas is not None:
                # We proved invariant but the property is not implied
                # do a big step in one of the invariants
                idx = np.random.randint(0, len(previous_alphas))
                while previous_alphas[idx] == 0:
                    idx = np.random.randint(0, len(previous_alphas))

                if idx < len(previous_alphas) / 2:
                    gmodel.addConstr(alphas_l[idx] <= previous_alphas[idx] * 2, "ce_output_alpha_l")
                else:
                    gmodel.addConstr(alphas_u[idx - len(alphas_u)] >= previous_alphas[idx] * 2, 'ce_output_alpha_u')
                # for i, a in enumerate(previous_alphas):
                #     # First half of previous_alphas is a_l, second a_u
                #     if i < len(previous_alphas) / 2:
                #         gmodel.addConstr(alphas_l[i] <= a, "ce_output_alpha_l")
                #     else:
                #         gmodel.addConstr(alphas_u[i - len(alphas_u)] >= a,  'ce_output_alpha_u')

        if self.add_alpha_constraint and loop_detected:
            for j in range(len(alphas_u)):
                gmodel.addConstr(alphas_u[j] >= self.alphas[j + len(alphas_u)] + SMALL, 'loop_constraint_u')
                if self.alphas[j] > SMALL:
                    gmodel.addConstr(alphas_l[j] <= self.alphas[j] - SMALL, 'loop_constraint_l')

        if not PRINT_GUROBI:
            gmodel.setParam('OutputFlag', False)

        gmodel.optimize()
        # gmodel.write("temp.lp")

        if gmodel.status == GRB.CUTOFF:
            print("CUTOFF")
            raise ValueError("CUTOFF problem")
        if gmodel.status == GRB.INFEASIBLE:
            # print("INFEASIBLE sum_alpahs = {} constraint_type={}".format(alphas_sum, ''))
            self.is_infesiable = True
            raise ValueError("INFEASIBLE problem")

        # print("FEASIBLE sum_alpahs = {}".format(alphas_sum))

        # for v in alphas_l:
        #     print(v.varName, 1 * v.x)
        # for v in alphas_u:
        #     print(v.varName, v.x)

        # for v in x_add_l:
        #     print(v.varName, 1 * v.x)
        # for v in x_add_u:
        #     print(v.varName, v.x)

        return [1 * a.x for a in alphas_l] + [a.x for a in alphas_u]

    def update_equation(self, idx):

        # Change the formal of initial values, first those for LE equations next to GE
        initial_values = self.initial_values[0] + self.initial_values[1]
        self.equations[idx] = alpha_to_equation(self.rnn_start_idxs[idx], self.rnn_output_idxs_double[idx],
                                                initial_values[idx], self.alphas[idx], self.inv_type[idx])

    def name(self):
        return 'gurobi_based_{}_{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)

    def extract_equation_from_counter_example(self, counter_examples=[{}]):
        '''

        :param counter_examples: Array of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of rnn_output values (as number of alpha_u), times the assingment
        for t  (len(times) == len(outputs)
        '''
        outputs = []
        times = []
        for i, counter_example in enumerate(counter_examples):
            if counter_example == {}:
                # We late user the index to update the correct invariant, so need to keep same length
                outputs.append(None)
                times.append(None)
                continue
            # We need to extract the time, and the values of all output indcies
            # Next create alpha_u >= time * output \land alpha_l \le time * output
            # outputs.append([counter_example[i] for i in self.rnn_output_idxs])
            outputs.append(counter_example[self.rnn_output_idxs_double[i]])
            # assert counter_example[self.rnn_start_idxs[0]] == counter_example[self.rnn_start_idxs[1]]
            times.append(counter_example[self.rnn_start_idxs[0]])
        return outputs, times

    def extract_hyptoesis_from_counter_example(self, counter_examples=[{}]):
        '''

        :param counter_examples: Array of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of memory cell values (as number of alpha_u), times the assingment
        for t  (len(times) == len(outputs)
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
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert,
        weaker otherwise
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
            new_min_sum = None  # sum(self.alphas) * np.random.normal(0.5, 0.1)
            direction = -1
        else:
            new_min_sum = sum(self.alphas) * 2
            direction = 1

        i = random.randint(0, len(self.alphas) - 1)
        if self.same_step_counter > self.random_threshold:
            # print("random step")
            self.do_random_step(strengthen)
            new_alphas = None
        else:
            counter_examples = self.extract_equation_from_counter_example(sat_vars)
            # hyptoesis_ce = self.extract_hyptoesis_from_counter_example(sat_vars)

            new_alphas = self.do_gurobi_step(strengthen, alphas_sum=new_min_sum, previous_alphas=self.alphas)
            if new_alphas == self.alphas:
                new_alphas = self.do_gurobi_step(strengthen, alphas_sum=new_min_sum, counter_examples=counter_examples,
                                                 loop_detected=True)

            if new_alphas is None:
                # No fesabile solution, maybe to much over approximation, improve at random
                # TODO: Can we get out of this situation? fall back to something else or doomed to random?
                self.do_random_step(strengthen)
                # self.alphas[i] = self.alphas[i] + (direction * 10)
                # print("********************************\nproblem\n**************************************")
            else:
                self.alphas = new_alphas

        for i in range(len(self.alphas)):
            self.update_equation(i)
        # print(self.alphas)
        self.alpha_history.append(self.get_alphas())

        # self.update_next_idx_step()
        # self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
        return self.equations

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
