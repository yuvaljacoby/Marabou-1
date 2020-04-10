from timeit import default_timer as timer
import random
from datetime import datetime
from itertools import product
from typing import List, Tuple, Union

from gurobipy import LinExpr, Var, Model, Env, GRB, setParam

from RNN.MarabouRNNMultiDim import alpha_to_equation
from maraboupy import MarabouCore
from polyhedron_algorithms.GurobiBased.Bound import Bound

random.seed(0)
sign = lambda x: 1 if x >= 0 else -1

SMALL = 10 ** -4
LARGE = 10 ** 5
PRINT_GUROBI = False

setParam('Threads', 1)
setParam('NodefileStart', 0.5)


class GurobiSingleLayer:
    def __init__(self, rnnModel, xlim, polyhedron_max_dim, use_relu=True, use_counter_example=False,
                 add_alpha_constraint=False,
                 layer_idx=0, **kwargs):
        '''

        :param rnnModel:
        :param xlim:
        '''
        self.w_in, self.w_h, self.b = rnnModel.get_weights()[layer_idx]
        self.rnnModel = rnnModel
        self.dim = self.w_h.shape[0]
        self.add_alpha_constraint = add_alpha_constraint
        self.use_counter_example = use_counter_example
        self.use_relu = use_relu
        self.return_vars = True
        self.xlim = xlim
        self.initial_values = None
        self.n_iterations = rnnModel.n_iterations
        self.layer_idx = layer_idx
        self.prev_layer_beta = [None] * self.dim
        self.alphas_u = []
        self.alphas_l = []
        self.added_constraints = None
        self.polyhedron_max_dim = polyhedron_max_dim
        self.step_num = 1
        self.same_step_counter = 0
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(layer_idx)
        self.rnn_output_idxs = rnn_output_idxs
        self.rnn_start_idxs = rnn_start_idxs + rnn_start_idxs
        self.rnn_output_idxs_double = rnn_output_idxs + rnn_output_idxs
        self.is_time_limit = False
        self.UNSAT = False

        self.inv_type = [MarabouCore.Equation.GE] * self.dim + [MarabouCore.Equation.LE] * self.dim
        self.equations = [None] * self.dim * 2
        if xlim is not None:
            self.update_xlim([x[0] for x in xlim], [x[1] for x in xlim])

        self.alphas = None
        # [Bound(False, self.initial_values[0][i], i) for i in range(self.dim)] + [Bound(True, self.initial_values[1][i], i) for i in range(self.dim)]
        # [-LARGE] * self.dim + [LARGE] * self.dim

        # assert len(self.alphas) == (2 * self.dim)
        # assert (2 * self.dim) == len(self.inv_type)
        assert (2 * self.dim) == len(self.rnn_output_idxs_double)

        self.last_fail = None
        self.alpha_history = []

    def __del__(self):
        return

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

    def calc_prev_layer_in_val(self, i: int, t: int) -> (int, int):
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

    def get_gurobi_rhs(self, i: int, t: int, alphas_l: List[Bound], alphas_u: List[Bound]) -> (LinExpr, LinExpr):
        '''
        The upper bound for guroib is: alpha_u[0] >= w_h * t * (alpha_u[i] + initial) (for all i) + x + b
        :param i: The index on which we want the rhs
        :param t: time stamp
        :param alphas_l: gurobi variable for lower bound for each recurrent node
        :param alphas_u: gurobi variable for upper bound for each recurrent node
        :return: (cond_l, cond_u) each of type LinExpr
        '''

        cond_u = LinExpr()
        cond_l = LinExpr()

        for j in range(self.w_h.shape[0]):
            if self.w_h[i, j] > 0:
                if hasattr(alphas_l[j], 'get_rhs'):
                    cond_l += alphas_l[j].get_rhs(t) * self.w_h[i, j]
                    cond_u += alphas_u[j].get_rhs(t) * self.w_h[i, j]
                else:
                    cond_l += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]
                    cond_u += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
            else:
                if hasattr(alphas_l[j], 'get_rhs'):
                    cond_l += alphas_u[j].get_rhs(t) * self.w_h[i, j]
                    cond_u += alphas_l[j].get_rhs(t) * self.w_h[i, j]
                else:
                    cond_l += (alphas_u[j] + self.initial_values[1][j]) * (t) * self.w_h[i, j]
                    cond_u += (alphas_l[j] + self.initial_values[0][j]) * (t) * self.w_h[i, j]

        cond_x_l, cond_x_u = self.calc_prev_layer_in_val(i, t)
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

    def get_gurobi_polyhedron_model(self):
        env = Env()
        gmodel = Model("test", env)

        obj = LinExpr()
        self.alphas_l, self.alphas_u = self.set_gurobi_vars(gmodel)

        for hidden_idx in range(self.w_h.shape[0]):
            for a in self.alphas_l[hidden_idx]:
                obj += a.get_objective()
            for a in self.alphas_u[hidden_idx]:
                obj += a.get_objective()
        # print("*" * 100)
        # print("l: ", self.alphas_l)
        # print("u: ", self.alphas_u)
        # print("*"*100)
        gmodel.setObjective(obj, GRB.MINIMIZE)

        # To understand the next line use:
        #       [list(a_l) for a_l in product(*[[1,2],[3]])], [list(a_u) for a_u in product(*[[10],[20,30]])]
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

                    if hasattr(alphas_l[hidden_idx], 'get_lhs'):
                        self.add_disjunction_rhs(gmodel, conds_l, alphas_l[hidden_idx].get_lhs(t), False,
                                                 "alpha_l{}_t{}".format(hidden_idx, t))
                        self.add_disjunction_rhs(gmodel, conds_u, alphas_u[hidden_idx].get_lhs(t), True,
                                                 "alpha_u{}_t{}".format(hidden_idx, t))
                    else:
                        assert False
                        self.add_disjunction_rhs(gmodel, conds_l, alphas_l[hidden_idx] * (t + 1), False,
                                                 "alpha_l{}_t{}".format(hidden_idx, t))
                        self.add_disjunction_rhs(gmodel, conds_u, alphas_u[hidden_idx] * (t + 1), True,
                                                 "alpha_u{}_t{}".format(hidden_idx, t))
        if not PRINT_GUROBI:
            gmodel.setParam('OutputFlag', False)

        # gmodel.write("get_gurobi_polyhedron_model.lp")
        return env, gmodel

    def set_gurobi_vars(self, gmodel: Model) -> Tuple[List[List[Bound]], List[List[Bound]]]:
        alphas_u = []
        alphas_l = []
        for hidden_idx in range(self.w_h.shape[0]):
            alphas_l.append([])
            alphas_u.append([])
            for j in range(self.step_num):
                cur_init_vals = (self.initial_values[0][hidden_idx], self.initial_values[1][hidden_idx])
                alphas_l[hidden_idx].append(Bound(gmodel, False, cur_init_vals[0], hidden_idx, j))
                alphas_u[hidden_idx].append(Bound(gmodel, True, cur_init_vals[1], hidden_idx, j))

        self.step_num += 1
        return alphas_l, alphas_u

    def improve_gurobi_model(self, gmodel: Model) -> bool:
        '''
        Got o model, extract anything needed to improve in the next iteration
        :param gmodel: infeasible model
        :return wheater to do another step or not
        '''
        return self.step_num <= self.polyhedron_max_dim


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

        # gmodel.write("get_gurobi_basic_model.lp")
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

    def do_gurobi_step(self, strengthen: bool, alphas_sum=None, counter_examples=([], []), hyptoesis_ce=([], []),
                       loop_detected=False, previous_alphas=None, tried_vars_improve=None) \
            -> Union[List[int], List['Bound']]:
        if not strengthen:
            # Invariant failed, does not suppose to happen
            assert False

        start_time = timer()
        env, gmodel = self.get_gurobi_polyhedron_model()

        if tried_vars_improve is None:
            tried_vars_improve = set()
        # not using this for now
        if False and self.use_counter_example:
            if strengthen and previous_alphas is not None:
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
        if status == GRB.OPTIMAL:
            alphas = [[a.model_optimized() for a in ls] for ls in self.alphas_l] + [
                [a.model_optimized() for a in ls]
                for ls in self.alphas_u]
            print("{}: FEASIBLE alpahs = {}".format(str(datetime.now()).split(".")[0],
                                                    [str(a) for a_ls in alphas for a in a_ls]))

        elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
            error = ValueError("INFEASIBLE problem")
        else:
            # Not sure which other statuses can be ...
            assert False, status

        if error:
            # TODO: Keep track on the recursion depth and use it for generating new bounds
            if self.improve_gurobi_model(gmodel):
                end_time = timer()
                print("FAIL Gurobi Step, retry, seconds:", round(end_time - start_time, 3))
                gmodel.dispose()
                env.dispose()
                self.do_gurobi_step(True)
            elif self.added_constraints is not None:
                assert False
                # If the problem is infeasible and it's not the first try, add constraint and try again
                for con in self.added_constraints:
                    gmodel.remove(con)

                self.added_constraints = []
                # self.last_fail = None

                alphas = self.do_gurobi_step(strengthen, previous_alphas=self.alphas,
                                             tried_vars_improve=tried_vars_improve)
            else:
                self.UNSAT = True
                self.equations = None
                alphas = None


        # if self.UNSAT:
        #     self.equations = None
        #     return None

        # if alphas is None and not self.UNSAT:
        #     assert False

        gmodel.dispose()
        env.dispose()

        return alphas

    def update_all_equations(self):

        initial_values = self.initial_values[0] + self.initial_values[1]
        self.equations = []
        if not isinstance(self.alphas[0], list):
            self.alphas = [[a] for a in self.alphas]

        for i, alpha in enumerate(self.alphas):
            self.equations.append([])
            for a in alpha:
                if isinstance(a, Bound):
                    eq = a.get_equation(self.rnn_start_idxs[i], self.rnn_output_idxs_double[i])
                else:
                    # TODO: delete isinstance
                    assert False
                    eq = alpha_to_equation(self.rnn_start_idxs[i], self.rnn_output_idxs_double[i], initial_values[i], a,
                                           self.inv_type[i])

                self.equations[-1].append(eq)

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

    def get_bounds(self) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        '''
        :return: (lower_bounds, upper_bounds), each bound is a list of (alpha,beta)
        '''
        if isinstance(self.alphas_l[0], list) and isinstance(self.alphas_l[0][0], Bound):
            return [[b.get_bound() for b in al] for al in self.alphas_l], \
                   [[b.get_bound() for b in au] for au in self.alphas_u]
        elif isinstance(self.alphas[0], list):
            # TODO: delete isinstance
            assert False
            alphas = [a[0] for a in self.alphas]
        else:
            # TODO: delete isinstance
            assert False
            alphas = self.alphas

        alphas_l = alphas[:len(alphas) // 2]
        alphas_u = alphas[len(alphas) // 2:]

        return ([[(a, b)] for a, b in zip(alphas_l, self.initial_values[0])],
                [[(a, b)] for a, b in zip(alphas_u, self.initial_values[1])])
        # else:
        #     raise Exception
