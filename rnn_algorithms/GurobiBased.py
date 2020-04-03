import random
from itertools import product

from typing import List

random.seed(0)
import numpy as np
from gurobipy import *
from datetime import datetime
from maraboupy import MarabouCore
from maraboupy.MarabouRNNMultiDim import alpha_to_equation
from rnn_algorithms.Update_Strategy import Absolute_Step

sign = lambda x: 1 if x >= 0 else -1

SMALL = 10 ** -4
LARGE = 10 ** 5

MAX_EQS_PER_DIMENSION = 1
RANDOM_THRESHOLD = 200  #
PRINT_GUROBI = False

# e = Env()
setParam('Threads', 1)
setParam('NodefileStart', 0.5)


class Bound():
    def __init__(self, gmodel: Model, upper: bool, initial_value: float, bound_idx: int, polyhedron_idx: int):
        self.upper = upper
        self.alpha = LARGE
        if not self.upper:
            self.alpha = -self.alpha
        self.initial_value = initial_value
        self.bound_idx = bound_idx
        self.polyhedron_idx = polyhedron_idx
        self.alpha_var = None
        self.beta_var = None
        self.gmodel = gmodel

        first_letter = 'u' if self.is_upper else 'l'
        self.alpha_var = \
            gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS,
                          name="a{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx))
        self.beta_var = gmodel.addVar(lb=self.initial_value, ub=LARGE, vtype=GRB.CONTINUOUS,
                                      name="b{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx))

    def get_rhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first add vars")
        return self.alpha_var * t + self.beta_var

    def get_lhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first add vars")
        return self.alpha_var * (t + 1) + self.beta_var

    def get_objective(self) -> LinExpr():
        obj = self.alpha_var + self.beta_var
        if not self.is_upper:
            obj = -obj
        return obj

    def is_upper(self) -> bool:
        return self.upper

    def get_equation(self, loop_idx, rnn_out_idx):
        if not self.alpha_var.x:
            raise Exception("First optimize the model")

        inv_type = MarabouCore.Equation.LE if self.is_upper() else MarabouCore.Equation.GE
        if inv_type == MarabouCore.Equation.LE:
            ge_better = -1
        else:
            # TODO: I don't like this either
            ge_better = 1
            # ge_better = -1

        invariant_equation = MarabouCore.Equation()
        invariant_equation.addAddend(1, rnn_out_idx)  # b_i
        invariant_equation.addAddend(self.alpha_var.x * ge_better, loop_idx)  # i
        invariant_equation.setScalar(self.beta_var.x)
        return invariant_equation


class AlphasGurobiBasedMultiLayer:
    # Alpha Search Algorithm for multilayer recurrent, assume the recurrent layers are one following the other
    # we need this assumptions in proved_invariant method, if we don't have it we need to extract bounds in another way
    # not sure it is even possible to create multi recurrent layer NOT in a row
    def __init__(self, rnnModel, xlim, update_strategy_ptr=Absolute_Step, random_threshold=RANDOM_THRESHOLD,
                 use_relu=True, use_counter_example=False, add_alpha_constraint=False, use_polyhedron=False, ):
        '''

        :param rnnModel: multi layer rnn model (MarabouRnnModel class)
        :param xlim: limit on the input layer
        :param update_strategy_ptr: not used
        :param random_threshold: not used
        :param use_relu: if to encode relus in gurobi
        :param use_counter_example: if property is not implied wheter to use counter_example to  create better _u
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
                prev_layer_lim = None  # [(-LARGE, LARGE) for _ in range(len(xlim))]
            self.alphas_algorithm_per_layer.append(
                AlphasGurobiBased(rnnModel, prev_layer_lim, update_strategy_ptr, random_threshold, use_relu,
                                  use_counter_example, add_alpha_constraint, layer_idx=i,
                                  use_polyhedron=use_polyhedron))

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
                 use_relu=True, use_counter_example=False, add_alpha_constraint=False, use_polyhedron=False,
                 layer_idx=0):
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
        self.initial_values = None
        self.random_threshold = random_threshold
        self.n_iterations = rnnModel.n_iterations
        self.layer_idx = layer_idx
        self.prev_layer_beta = [None] * self.dim
        self.alphas_u = []
        self.alphas_l = []
        self.added_constraints = None
        self.temp_alpha = []
        # initalize alphas to -infty +infty
        self.alphas = [-LARGE] * self.dim + [LARGE] * self.dim
        self.update_strategy = update_strategy_ptr()
        self.same_step_counter = 0
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(layer_idx)
        self.rnn_output_idxs = rnn_output_idxs
        self.rnn_start_idxs = rnn_start_idxs + rnn_start_idxs
        self.rnn_output_idxs_double = rnn_output_idxs + rnn_output_idxs
        self.is_time_limit = False
        self.UNSAT = False
        self.equations_per_dimension = 1
        self.use_polyhedron = use_polyhedron

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
        # gmodel = self.get_gurobi_basic_model()

        # Create first alphas

        # self.alphas = self.do_gurobi_step(strengthen=True)
        #
        # for i in range(len(self.alphas)):
        #     self.update_equation(i)

    def __del__(self):
        return

    # def presolve_basic_model(self):
    #     # TODO: We can't return the presolved model since it changes the variables, we need to somehow map between the
    #     #  new variables and the old ones
    #     # Another options is to use incremental solving, don't know how
    #     model = self.get_gurobi_basic_model()
    #     presolved_model = model.presolve()
    #     status = presolved_model.status
    #     error = None
    #     if status == GRB.CUTOFF:
    #         print("CUTOFF")
    #         error = ValueError("CUTOFF problem")
    #     elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
    #         print("INFEASIBLE")
    #         # self.is_infesiable = True
    #         error = ValueError("INFEASIBLE problem")
    #
    #     if error:
    #         model.dispose()
    #         raise error
    #     return model

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

        # TODO: This is good only if using the box (or polyhedron with n=1)
        if isinstance(lower_bound[0], list):
            lower_bound = [l[0] for l in lower_bound]
            upper_bound = [u[0] for u in upper_bound]
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

        initial_values = self.rnnModel.get_rnn_min_max_value_one_iteration(xlim, layer_idx=self.layer_idx,
                                                                           prev_layer_beta=beta)
        initial_values = ([0] * len(initial_values[0]), initial_values[1])
        self.initial_values = initial_values  # [1] + initial_values[0]

    def do_random_step(self, strengthen):
        assert False
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

    def calc_x_val(self, i: int, t: int) -> (int, int):
        cond_x_u = 0  # LinExpr()
        cond_x_l = 0  # LinExpr()
        for j in range(len(self.xlim)):
            if self.prev_layer_beta[0] is not None:
                # Not first RNN layer, previous layer bound on the memory unit is in  alpha*time + beta
                # In this case we know xlim > 0
                if self.w_in[j, i] >= 0:
                    cond_x_u += (self.xlim[j][1] * (t + 1) + self.prev_layer_beta[1][j]) * self.w_in[j, i]
                    cond_x_l += (self.xlim[j][0] * (t + 1) + self.prev_layer_beta[0][j]) * self.w_in[j, i]
                else:
                    cond_x_u += (self.xlim[j][0] * (t + 1) + self.prev_layer_beta[0][j]) * self.w_in[j, i]
                    cond_x_l += (self.xlim[j][1] * (t + 1) + self.prev_layer_beta[1][j]) * self.w_in[j, i]
            else:
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
        return cond_x_l, cond_x_u

    def get_gurobi_rhs(self, i: int, t: int, alphas_l: List[Var], alphas_u: List[Var]) -> (LinExpr, LinExpr):
        '''
        The upper bound for guroib is: alpha_u[0] >= w_h * t * (alpha_u[i] + initial) (for all i) + x + b
        :param i: The index on which we want the rhs
        :param t: time stamp
        :param alphas_l: gurobi variable for lower bound for each recurrent node
        :param alphas_u: gurobi variable for upper bound for each recurrent node
        :return: (cond_l, cond_u) each of type LinExpr
        '''
        #
        cond_u = LinExpr()
        cond_l = LinExpr()

        for j in range(self.w_h.shape[0]):
            if self.w_h[i, j] > 0:
                cond_l += (alphas_l[j] * t + self.initial_values[0][j]) * self.w_h[i, j]
                cond_u += (alphas_u[j] * t + self.initial_values[1][j]) * self.w_h[i, j]
                # cond_l += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]
                # cond_u += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
            else:
                cond_l += (alphas_u[j] * t + self.initial_values[1][j])  * self.w_h[i, j]
                cond_u += (alphas_l[j] * t + self.initial_values[0][j]) * self.w_h[i, j]
                # cond_l += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
                # cond_u += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]

        cond_x_l, cond_x_u = self.calc_x_val(i, t)
        cond_u += cond_x_u + self.b[i]  # + SMALL
        cond_l += cond_x_l + self.b[i]  # - SMALL

        return cond_l, cond_u

    @staticmethod
    def get_relu_constraint(gmodel, cond: LinExpr, i: int, t: int, upper_bound: bool):
        first_letter = "u" if upper_bound else "l"
        cond_f = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="{}b{}_t_{}".format(first_letter, i, t))
        delta = gmodel.addVar(vtype=GRB.BINARY)
        if upper_bound:
            gmodel.addConstr(cond_f >= cond, "cond_{}_relu0_i{}_t{}".format(first_letter, i, t))
            gmodel.addConstr(cond_f <= cond + LARGE * delta, "cond_{}_relu1_i{}_t{}".format(first_letter, i, t))
            gmodel.addConstr(cond_f <= LARGE * (1 - delta), "cond_{}_relu2_i{}_t{}".format(first_letter, i, t))
        else:
            gmodel.addConstr(cond_f >= cond, "cond_{}_relu0_i{}_t{}".format(first_letter, i, t))
            gmodel.addConstr(cond_f <= cond + LARGE * delta, "cond_{}_relu1_i{}_t{}".format(first_letter, i, t))
            gmodel.addConstr(cond_f <= LARGE * (1 - delta), "cond_{}_relu2_i{}_t{}".format(first_letter, i, t))
        return cond_f, delta

    @staticmethod
    def add_disjunction_rhs(gmodel, conds: List[LinExpr], lhs: Var, greater: bool, cond_name: str):
        '''
        Add all the conditions in conds as a disjunction (using binary variables)
        :param gmodel: model to add condition to
        :param conds: list of rhs expressions
        :param lhs: lhs condition
        :param greater: if True them lhs >= rhs else lhs <= rhs
        :param cond_name: name to add in gurobi
        :return:
        '''
        deltas = []
        cond_delta = LinExpr()
        for cond in conds:
            deltas.append(gmodel.addVar(vtype=GRB.BINARY))
            cond_delta += deltas[-1]
            if greater:
                gmodel.addConstr(lhs >= cond - (LARGE * deltas[-1]), cond_name)
            else:
                gmodel.addConstr(lhs <= cond + (LARGE * deltas[-1]), cond_name)
            gmodel.addConstr(deltas[-1] <= 0)

        gmodel.addConstr(cond_delta <= len(deltas) - 1, "{}_deltas".format(cond_name))

    def get_gurobi_polyhedron_model(self, n):
        env = Env()
        gmodel = Model("test", env)

        if self.alphas_l:
            self.alphas_l = []
            self.alphas_u = []

        obj = LinExpr()
        for hidden_idx in range(self.w_h.shape[0]):
            self.alphas_l.append([])
            self.alphas_u.append([])
            for j in range(n):
                self.alphas_l[hidden_idx].append(
                    gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_l^{}_{}".format(j, hidden_idx)))
                self.alphas_u[hidden_idx].append(
                    gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_u^{}_{}".format(j, hidden_idx)))

                # we want the lower bound to be as tight as possibole so we should prefer large numbers on small numbers
                obj += -self.alphas_l[hidden_idx][-1]
                obj += self.alphas_u[hidden_idx][-1]
            # TODO: Add constraint that the alphas needs to be different?

        gmodel.setObjective(obj, GRB.MINIMIZE)

        # To understand the next line use:
        #       [(list(a_l), list(a_u)) for a_u in product(*[[1, 2], [3, 4]]) for a_l in product(*[[10,20] ,[30,40]])]
        # all_alphas = [(list(a_l), list(a_u)) for a_l in product(*self.alphas_l) for a_u in product(*self.alphas_u)]
        all_alphas = [[list(a_l) for a_l in product(*self.alphas_l)], [list(a_u) for a_u in product(*self.alphas_u)]]
        for hidden_idx in range(self.w_h.shape[0]):
            for t in range(self.n_iterations):
                conds_l = []
                conds_u = []
                for alphas_l, alphas_u in zip(*all_alphas):
                    cond_l, cond_u = self.get_gurobi_rhs(hidden_idx, t, alphas_l, alphas_u)
                    if self.use_relu:
                        cond_u, _ = self.get_relu_constraint(gmodel, cond_u, hidden_idx, t, True)
                        cond_l, _ = self.get_relu_constraint(gmodel, cond_l, hidden_idx, t, False)
                    conds_l.append(cond_l)
                    conds_u.append(cond_u)

                    self.add_disjunction_rhs(gmodel, conds_l, alphas_l[hidden_idx] * (t + 1), False,
                                             "alpha_l{}_t{}".format(hidden_idx, j, t))
                    self.add_disjunction_rhs(gmodel, conds_u, alphas_u[hidden_idx] * (t + 1), True,
                                             "alpha_u{}_t{}".format(hidden_idx, j, t))

        if not PRINT_GUROBI:
            gmodel.setParam('OutputFlag', False)

        gmodel.write("get_gurobi_polyhedron_model.lp")
        return env, gmodel

    def get_gurobi_basic_model(self):
        env = Env()
        gmodel = Model("test", env)

        if self.alphas_l:
            # TODO: Remove, it's here only to make we don't re append constraints
            self.alphas_l = []
            self.alphas_u = []
        obj = LinExpr()
        for hidden_idx in range(self.w_h.shape[0]):
            self.alphas_l.append(
                gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_l_{}".format(hidden_idx)))

            # we want the lower bound to be as tight as possibole so we should prefer large numbers on small numbers
            obj += -self.alphas_l[-1]
            self.alphas_u.append(
                gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="alpha_u_{}".format(hidden_idx)))
            obj += self.alphas_u[-1]

        gmodel.setObjective(obj, GRB.MINIMIZE)

        for hidden_idx in range(self.w_h.shape[0]):
            # TODO: Should t start from zero or 1? start from zero for now
            for t in range(self.n_iterations):
                cond_l, cond_u = self.get_gurobi_rhs(hidden_idx, t, self.alphas_l, self.alphas_u)
                if self.use_relu:
                    cond_u, d = self.get_relu_constraint(gmodel, cond_u, hidden_idx, t, True)
                    cond_l, d = self.get_relu_constraint(gmodel, cond_l, hidden_idx, t, False)

                gmodel.addConstr(self.alphas_u[hidden_idx] * (t + 1) >= cond_u, "alpha_u{}_t{}".format(hidden_idx, t))
                gmodel.addConstr(self.alphas_l[hidden_idx] * (t + 1) <= cond_l, "alpha_l{}_t{}".format(hidden_idx, t))

        if not PRINT_GUROBI:
            gmodel.setParam('OutputFlag', False)

        gmodel.write("get_gurobi_basic_model.lp")
        return env, gmodel

    def gurobi_step_in_random_direction(self, previous_alphas, failed_improves=set()):
        valid_idx = [i for i in range(len(previous_alphas)) if i not in failed_improves and previous_alphas[i] != 0]
        if len(valid_idx) == 0:
            self.UNSAT = True
            return None, None, None
        else:
            idx = random.choice(valid_idx)
        assert previous_alphas[idx] != 0
        assert idx not in failed_improves

        # idx = np.random.randint(0, len(previous_alphas))
        # while previous_alphas[idx] == 0 or idx in failed_improves:
        #     idx = np.random.randint(0, len(previous_alphas))

        if idx < len(previous_alphas) / 2:
            return self.alphas_l[idx] >= previous_alphas[idx] * 2, "ce_output_alpha_l", idx
        else:
            print("adding constraint, alpha_u{} <= {}".format(idx - len(self.alphas_u), previous_alphas[idx] - SMALL))
            return self.alphas_u[idx - len(self.alphas_u)] <= previous_alphas[idx] - SMALL, 'ce_output_alpha_u', idx

    def do_gurobi_step(self, strengthen, alphas_sum=None, counter_examples=([], []), hyptoesis_ce=([], []),
                       loop_detected=False, previous_alphas=None, tried_vars_improve=None):
        if self.use_polyhedron:
            env, gmodel = self.get_gurobi_polyhedron_model(self.equations_per_dimension)
        else:
            env, gmodel = self.get_gurobi_basic_model()

        if tried_vars_improve is None:
            tried_vars_improve = set()
        if self.use_counter_example:
            if not strengthen:
                # Invariant failed, does not suppose to happen
                assert False

            if False and strengthen and previous_alphas is not None:
                # We proved invariant but the property is not implied do a big step in one of the invariants
                random_constraint, constraing_description, improve_idx = \
                    self.gurobi_step_in_random_direction(previous_alphas, tried_vars_improve)
                if random_constraint is None:
                    return None
                tried_vars_improve.add(improve_idx)
                self.added_constraints.append(gmodel.addConstr(random_constraint, constraing_description))

        # if self.add_alpha_constraint and loop_detected:
        #     for j in range(len(self.alphas_u)):
        #         loop_cons_u = self.alphas_u[j] >= self.alphas[j + len(self.alphas_u)] + SMALL
        #         self.added_constraints.append(self.gmodel.addConstr(loop_cons_u, 'loop_constraint_u'))
        #         if self.alphas[j] > SMALL:
        #             loop_cons_l = self.alphas_l[j] <= self.alphas[j] - SMALL
        #             self.added_constraints.append(self.gmodel.addConstr(loop_cons_l, 'loop_constraint_l'))

        gmodel.optimize()

        status = gmodel.status
        error = None
        alphas = None
        if status == GRB.CUTOFF:
            print("CUTOFF")
            error = ValueError("CUTOFF problem")
        elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
            gmodel.computeIIS()
            gmodel.write('get_gurobi_polyhedron.ilp')

            print("INFEASIBLE")
            self.is_infesiable = True
            self.equations_per_dimension += 1
            if self.equations_per_dimension > MAX_EQS_PER_DIMENSION:
                error = ValueError("INFEASIBLE problem")
            else:
                assert False
                self.do_gurobi_step(True)
        elif isinstance(self.alphas_u[0], list):
            # Using the polyhedron model
            alphas = [[a.x for a in ls] for ls in self.alphas_l] + [[a.x for a in ls] for ls in self.alphas_u]
        else:
            alphas = [1 * a.x for a in self.alphas_l] + [a.x for a in self.alphas_u]

        if error:
            # TODO: Keep track on the recursion depth and use it for generating new bounds
            if self.added_constraints is not None:
                assert False
                # If the problem is infeasible and it's not the first try, add constraint and try again
                for con in self.added_constraints:
                    gmodel.remove(con)

                self.added_constraints = []
                # self.last_fail = None
                self.temp_alpha.append(alphas)
                alphas = self.do_gurobi_step(strengthen, previous_alphas=self.alphas,
                                             tried_vars_improve=tried_vars_improve)
            else:
                self.UNSAT = True

        if self.UNSAT:
            self.equations = None
            return None

        if alphas is None and not self.UNSAT:
            assert False

        gmodel.dispose()
        env.dispose()

        if isinstance(alphas[0], list):
            print("{}: FEASIBLE alpahs = {}".format(str(datetime.now()).split(".")[0],
                                                    [round(a, 3) for a_ls in alphas for a in a_ls]))
        else:
            print("{}: FEASIBLE alpahs = {}".format(str(datetime.now()).split(".")[0], [round(a, 3) for a in alphas]))

        return alphas

    def update_all_equations(self):
        initial_values = self.initial_values[0] + self.initial_values[1]
        self.equations = []
        if not isinstance(self.alphas[0], list):
            self.alphas = [[a] for a in self.alphas]

        for i, alpha in enumerate(self.alphas):
            self.equations.append([])
            for a in alpha:
                self.equations[-1].append(
                    alpha_to_equation(self.rnn_start_idxs[i], self.rnn_output_idxs_double[i],
                                      initial_values[i], a, self.inv_type[i]))

    def update_equation(self, idx):
        initial_values = self.initial_values[0] + self.initial_values[1]

        # Change the formal of initial values, first those for LE equations next to GE
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
        if counter_examples is None:
            return None

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
            print("random step")
            self.do_random_step(strengthen)
            new_alphas = None
        else:
            counter_examples = self.extract_equation_from_counter_example(sat_vars)
            # hyptoesis_ce = self.extract_hyptoesis_from_counter_example(sat_vars)

            new_alphas = self.do_gurobi_step(strengthen, alphas_sum=new_min_sum, previous_alphas=self.alphas)
            if self.UNSAT:
                return None

            if new_alphas == self.alphas:
                new_alphas = self.do_gurobi_step(strengthen, alphas_sum=new_min_sum, counter_examples=counter_examples,
                                                 loop_detected=True)

            if new_alphas is None:
                assert False
                # No fesabile solution, maybe to much over approximation, improve at random
                # TODO: Can we get out of this situation? fall back to something else or doomed to random?
                self.do_random_step(strengthen)
                # self.alphas[i] = self.alphas[i] + (direction * 10)
                # print("********************************\nproblem\n**************************************")
            else:
                self.alphas = new_alphas

        self.update_all_equations()
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
        if self.UNSAT:
            return None

        if self.equations[0] is None:
            # First call, first update the equations
            self.do_step(True)
        return self.equations

    def get_alphas(self):
        return self.alphas
