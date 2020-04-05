from z3 import *
from RNN.MarabouRNNMultiDim import alpha_to_equation, double_list
from maraboupy import MarabouCore

ALPHA_SUM_HIGH_END_INIT = 1000
ALPHA_SUM_LOW_END_INIT = -1000
# ALPHA_START = 100
NUMBER_OF_ITERATIONS = 1000


# TODO
# PROBLEM
# The current z3 query does not work for multiple variables since it is not using the over approximation of i-1
# (i.e. the induction hyptoesis) therefore it gets stuck with bad alphas...

def z3_real_to_float(z3_real):
    return float(z3_real.as_decimal(4).replace('?', '').replace("+", ''))


def ReLU(after, before):
    # return after == before
    return after == If(before > 0, before, 0)


class SmtAlphaSearch:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, w_h, w_i, bias,
                 prev_layer_min, prev_layer_max, n_iterations):
        '''

        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        :param w_h: square matrix with recurrent hidden weights
        :param w_i: matrix for the input to the recurrent
        :param bias: bias for the recurrent output
        '''

        self.rnn_start_idxs = double_list(rnn_start_idxs)
        self.rnn_output_idxs = double_list(rnn_output_idxs)
        min_values = initial_values[0]
        max_values = initial_values[1]
        # self.initial_values = [item for i in range(len(min_values)) for item in [min_values[i], max_values[i]]]
        self.initial_values = [item for i in range(len(min_values)) for item in [min_values[i], max_values[i]]]
        # times two to have for each invariant ge and le
        self.high_end = [ALPHA_SUM_HIGH_END_INIT, ALPHA_SUM_HIGH_END_INIT]
        self.low_end = [ALPHA_SUM_LOW_END_INIT, ALPHA_SUM_LOW_END_INIT]
        self.prev_layer_min = prev_layer_min
        self.prev_layer_max = prev_layer_max
        self.iterations = n_iterations
        self.w_h = w_h
        self.w_in = w_i
        self.bias = bias
        self.inv_type = [MarabouCore.Equation.GE if i % 2 == 0 else MarabouCore.Equation.LE for i in
                         range(len(self.initial_values))]
        # self.inv_type = [MarabouCore.Equation.LE for _ in range(len(min_values))]
        self.invariant_equations = [None] * len(self.inv_type)

        # self._update_alphas()
        # self._update_invariant_equation()

        # the property steps are much slower, for each step we do all inductive steps at the worst case
        self.property_steps = 15
        # self.alphas_le = [AlphaSearchSGD() for _ in range(len(initial_values[0]))]
        assert len(self.rnn_output_idxs) == len(self.rnn_start_idxs)
        # assert len(self.rnn_output_idxs) == len(self.alphas)
        assert len(self.rnn_output_idxs) == len(self.initial_values)
        assert len(self.rnn_output_idxs) == len(self.invariant_equations)

    def _update_invariant_equation(self):
        for i in range(len(self.invariant_equations)):
            self.invariant_equations[i] = alpha_to_equation(self.rnn_start_idxs[i], self.rnn_output_idxs[i],
                                                            self.initial_values[i], self.alphas[i],
                                                            self.inv_type[i])

    def _update_alphas(self):

        def create_z3_query(lower_bound: bool):
            # run z3 until SAT and update self.alphas
            s = Solver()
            s.push()
            in_tensor = []

            # Limit the previous layer
            for i in range(len(self.prev_layer_min)):
                in_tensor.append(Real('in_{}'.format(i)))
                s.add(in_tensor[-1] >= self.prev_layer_min[i])
                s.add(in_tensor[-1] <= self.prev_layer_max[i])

            # Create a 2d array for all recurrent nodes
            # The variable name is Rdim_iteration (for example for 3 iterations 2d recurrent we will have r0_0, r0_1, r0_2, r1_0 ...)
            recurrent = [[Real('R{}_0'.format(t))] for t in range(len(self.w_h))]
            # Initialize the first variable to zero
            for node in recurrent:
                s.add(node[0] == 0)

            alphas_variables = [Real('alpha_max_{}'.format(j)) for j in range(len(recurrent))]
            # alphas_min = [Real('alpha_min_{}'.format(j)) for j in range(len(recurrent))]
            # Bound the next iterations
            for t in range(1, self.iterations):
                # First create new variables for all recurrent nodes, we do so in order to use -2 in next loop
                for j in range(len(recurrent)):
                    recurrent[j].append(Real('R{}_{}'.format(j, t)))

                for j in range(len(recurrent)):
                    expression = 0
                    for idx in range(len(recurrent)):
                        expression += self.w_h[j, idx] * recurrent[j][-2]
                    expression += self.w_in[j, idx] * in_tensor[idx]
                    s.add(ReLU(recurrent[j][-1], expression + self.bias[j]))

                    # Make sure a*t + beta >= Rj_t
                    # TODO: The sign here depndes on the invariant type
                    if lower_bound:
                        s.add(recurrent[j][-1] >= alphas_variables[j] * t + self.initial_values[j])  # 2*j + 1])
                    else:
                        s.add(recurrent[j][-1] <= alphas_variables[j] * t + self.initial_values[j])  # 2*j + 1])

            return s, alphas_variables

        def find_alphas(lower_bound: bool, low_end, high_end):
            query, alphas = create_z3_query(lower_bound)
            found = False
            count_iterations = 0
            while high_end >= 0.0005 + low_end and (not found):
                count_iterations += 1
                cur_sum = (high_end + low_end) / 2
                query.push()

                # limit the sum of alphas
                alpha_sum_exp = alphas[0]
                for i in range(1, len(alphas)):
                    alpha_sum_exp += alphas[i]
                query.add(alpha_sum_exp <= cur_sum)
                query.add(alpha_sum_exp > low_end)

                print('iteration: {}, high_end: {}, low_end: {}'.format(count_iterations, high_end, low_end))
                if query.check() == sat:
                    model = query.model()
                    alpha_vars = [model[a] for a in alphas]
                    # In inv_type LE we negate the alphas
                    if lower_bound:
                        factor = -1
                    else:
                        factor = 1
                    alpha_values = [factor * z3_real_to_float(a) for a in alpha_vars]
                    # print('found alphas: {}'.format(alpha_values))
                    found = True
                else:
                    query.pop()
                    low_end = cur_sum
            if not found:
                alpha_values = None
                assert False
            return low_end, high_end, alpha_values

        self.low_end[0], self.high_end[0], ge_alphas = find_alphas(False, self.low_end[0], self.high_end[0])
        self.low_end[1], self.high_end[1], le_alphas = find_alphas(True, self.low_end[1], self.high_end[1])
        self.alphas = [a[i] for a in zip(ge_alphas, le_alphas) for i in range(2)]
        print('finished to improve alphas, new values: {}'.format(self.alphas))
        self._update_invariant_equation()

    def getAlphasThatProveProperty(self, invariant_oracle, property_oracle):
        '''
        Look for alphas that prove the property using the "SGD" algorithm.
        i.e. we first check if basic alphas work, if not
        for each alpha:
            check that it's inductive (if not fix)
            check if property holds
        we do this loop property_steps times
        :param invariant_oracle: function pointer, input is list of marabou equations, output is whether this is a valid invariant
        :param property_oracle: function pointer, input is list of marabou equations, output is whether the property holds using this invariants
        :return: list of alphas if proved, None otherwise
        '''
        # TODO: How can I do this I did not prove the invariant holds...
        # First check if need if the property holds
        # if property_oracle(self.invariant_equations):
        #     return self.alphas
        # Run maximum property_steps
        counter = 0
        while counter < self.property_steps:
            counter += 1
            self._update_alphas()
            # self._update_invariant_equation()

            # TODO: separate ge and le invariants?
            if self._proveInductiveAlphasOnce(invariant_oracle, range(len(self.invariant_equations))):
                if property_oracle(self.invariant_equations):
                    return self.alphas
                else:
                    self.high_end[0] = sum([self.alphas[i] for i in range(0, len(self.alphas), 2)])
                    self.high_end[1] = sum([self.alphas[i] for i in range(1, len(self.alphas), 2)])
            else:
                # We are not suppose to get here, the only reason to get here is preseicion difference between marabou and z3
                self.low_end[0] = sum([self.alphas[i] for i in range(1, len(self.alphas), 2)])
                self.low_end[1] = sum([self.alphas[i] for i in range(0, len(self.alphas), 2)])
                # self.low_end = sum(self.alphas)
                # assert False
        return None

    def _proveInductiveAlphasOnce(self, invariant_oracle, invariant_equations_idx):
        prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])

        # for i, res in enumerate(prove_inv_res):
        #     # i is an index in the current invariant_equations which is a subset of the entire invariants
        #     idx = invariant_equations_idx[i]
        #     if res:
        #         self.alphas[idx].proved_alpha()
        #     else:
        #         # Doing a step
        #         self.alphas[idx].update_invariant_fail()
        #         self._update_invariant_equation(idx)

        # Update the results (might be that we proved something new)
        # prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])
        # print("after invariant fail, new_alphas:", [a.get() for a in self.alphas])
        return all(prove_inv_res)

    def __name__(self):
        return 'SMT_binary_search'

