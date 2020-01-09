import numpy as np
from maraboupy.MarabouRNNMultiDim import alpha_to_equation, double_list
from maraboupy import MarabouCore

ALPHA_START = 100
NUMBER_OF_ITERATIONS = 1000

class AlphaSearchSGD:
    '''
    This is a class for a single alpha, how to update it
    '''

    def __init__(self):
        self.alpha = ALPHA_START
        self.old_alpha = None
        self.large = 10
        self.next_step = None
        # self.reset_search()

    def proved_alpha(self):
        '''
        Update with the last used alpha that is proved to work
        :param used_alpha:
        :return:
        '''
        pass
        # temp_alpha = self.get()
        # if self.alpha is not None and temp_alpha > self.alpha:
        #     self.alpha = temp_alpha

    def reset_search(self):
        self.max_val = self.large
        self.min_val = -self.large
        self.old_alpha = self.get()

    def update_invariant_fail(self):
        '''
        Next time the will get a larger alpha
        :param used_alpha:
        :return:
        '''

        direction = 1
        self.step(direction)

    def update_property_fail(self):
        '''
        Update the using this alpha a property failed, next time will get smaller alpha
        :param used_alpha:
        :return: Wheather we can still improve this alpha or not
        '''
        direction = -1
        # self.max_val = self.get()
        self.step(direction)
        # print("property fail use smaller alpha, new alpha:", self.alpha)
        return self.alpha < self.large

    def step(self, direction):
        sign = lambda x: 1 if x >= 0 else -1
        self.prev_alpha = self.alpha
        if abs(self.alpha) > 0.2:
            self.alpha = self.alpha + (
                    direction * self.alpha * 0.3 * sign(self.alpha))  # do step size 0.3 to the next direction
        else:
            self.alpha = 0.5 * direction
        return self.alpha

    def get(self):
        return self.alpha
        # if self.old_alpha is not None:
        #     old_alpha = self.old_alpha
        #     self.old_alpha = None
        #     return old_alpha

        # return (self.max_val + self.min_val) / 2


class MaxAlphasSGD:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs):
        '''

        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''

        self.rnn_start_idxs = double_list(rnn_start_idxs)
        self.rnn_output_idxs = double_list(rnn_output_idxs)
        min_values = initial_values[0]
        max_values = initial_values[1]
        self.initial_values = [item for i in range(len(min_values)) for item in [min_values[i], max_values[i]]]
        # times two to have for each invariant ge and le
        self.alphas = [AlphaSearchSGD() for _ in range(len(self.initial_values))]
        self.inv_type = [MarabouCore.Equation.GE if i % 2 == 0 else MarabouCore.Equation.LE for i in
                         range(len(self.alphas))]

        self.invariant_equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self._update_invariant_equation(i)

        # the property steps are much slower, for each step we do all inductive steps at the worst case
        self.property_steps = NUMBER_OF_ITERATIONS
        # self.alphas_le = [AlphaSearchSGD() for _ in range(len(initial_values[0]))]
        assert len(self.rnn_output_idxs) == len(self.rnn_start_idxs)
        assert len(self.rnn_output_idxs) == len(self.alphas)
        assert len(self.rnn_output_idxs) == len(self.initial_values)
        assert len(self.rnn_output_idxs) == len(self.invariant_equations)

    def __name__(self):
        return 'random_sgd'

    def _update_invariant_equation(self, i):
        self.invariant_equations[i] = alpha_to_equation(self.rnn_start_idxs[i], self.rnn_output_idxs[i],
                                                        self.initial_values[i], self.alphas[i].get(), self.inv_type[i])

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
        counter = 1

        while counter < self.property_steps:
            # Choose alpha to do the step on
            # For each update we iterate through the values from max to min, and try to update one of the alphas
            # on success reset the search
            def arg_sort_alpha(alphas):
                f = lambda z: -alphas[z].alpha
                return sorted(range(len(alphas)), key=f)

            proved_inv = self.checkIfInductive(invariant_oracle)
            if all(proved_inv):
                # print("proved an invariant:", [a.get() for a in self.alphas])
                if property_oracle(self.invariant_equations):
                    return self.alphas
                else:
                    # The alpha improved but not enough to prove thr property, reset the search
                    direction = 1
            else:
                print("fail to prove inductive alphas cur alphas::", [a.get() for a in self.alphas])
                direction = -1
            for i in range(len(self.alphas)):
                alpha_idx = arg_sort_alpha(self.alphas)[i]
                if not proved_inv[alpha_idx]:
                    counter += 1

                    alpha = self.alphas[alpha_idx]
                    sign = lambda x: 1 if x >= 0 else -1
                    if alpha.alpha < 10:
                        step_size = 0.1
                        alpha.alpha += direction * sign(alpha.alpha) * step_size
                    else:
                        alpha.alpha += direction * sign(alpha.alpha) * 0.1 * alpha.alpha
                    self._update_invariant_equation(alpha_idx)
                    break
                    # proved_inv = self.checkIfInductive(invariant_oracle)

        return None


    def checkIfInductive(self, invariant_oracle):
        ge_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.GE]
        le_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.LE]
        assert sorted(ge_invariant_eq_idx + le_invariant_eq_idx) == list(range(len(self.inv_type)))

        all_ge_proved = self._proveInductiveAlphasOnce(invariant_oracle, ge_invariant_eq_idx)
        all_le_proved = self._proveInductiveAlphasOnce(invariant_oracle, le_invariant_eq_idx)

        return all_ge_proved + all_le_proved

    def _proveInductiveAlphasOnce(self, invariant_oracle, invariant_equations_idx):
        prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])

        return prove_inv_res