from maraboupy.MarabouRNNMultiDim import alpha_to_equation, double_list
from maraboupy import MarabouCore

class AlphaSearchSGD:
    '''
    This is a class for a single alpha, how to update it
    '''

    def __init__(self):
        self.alpha = 0
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

class IterateAlphasSGD:
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

        self.inductive_steps = 50
        self.invariant_equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self._update_invariant_equation(i)

        # the property steps are much slower, for each step we do all inductive steps at the worst case
        self.property_steps = 60
        # self.alphas_le = [AlphaSearchSGD() for _ in range(len(initial_values[0]))]
        assert len(self.rnn_output_idxs) == len(self.rnn_start_idxs)
        assert len(self.rnn_output_idxs) == len(self.alphas)
        assert len(self.rnn_output_idxs) == len(self.initial_values)
        assert len(self.rnn_output_idxs) == len(self.invariant_equations)

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
        # TODO: How can I do this I did not prove the invariant holds...
        # First check if need if the property holds
        # if property_oracle(self.invariant_equations):
        #     return self.alphas
        # Run maximum property_steps
        counter = 0
        while counter < self.property_steps:
            counter += 1
            # Do a step to improve each alpha, after that make sure they are inductive and try to prove the propety
            for i, alpha in enumerate(self.alphas):
                alpha.update_property_fail()
                self._update_invariant_equation(i)
                # Get inductive Alphas updates the equations
                if self.getInductiveAlphas(invariant_oracle):
                    print("proved an invariant:", [a.get() for a in self.alphas])
                    if property_oracle(self.invariant_equations):
                        return self.alphas
                else:
                    # print("fail to prove inductive alphas cur alphas::", [a.get() for a in self.alphas])
                    return None
        return None

    def _proveInductiveAlphasOnce(self, invariant_oracle, invariant_equations_idx):
        prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])

        for i, res in enumerate(prove_inv_res):
            # i is an index in the current invariant_equations which is a subset of the entire invariants
            idx = invariant_equations_idx[i]
            if res:
                self.alphas[idx].proved_alpha()
            else:
                # Doing a step
                self.alphas[idx].update_invariant_fail()
                self._update_invariant_equation(idx)

                # Update the results (might be that we proved something new)
                # prove_inv_res = invariant_oracle([self.invariant_equations[i] for i in invariant_equations_idx])
                # print("after invariant fail, new_alphas:", [a.get() for a in self.alphas])
        return all(prove_inv_res)

    def getInductiveAlphas(self, invariant_oracle):
        '''
        Do a step to get better alphas, use the oracle to verify that they are valid invariant
        :param invariant_oracle: function pointer, input is list marabou equations values output is whether this is a valid invariant
        :return: list of alphas if found better invariant, otherwise None
        '''
        ge_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.GE]
        le_invariant_eq_idx = [i for i in range(len(self.inv_type)) if self.inv_type[i] == MarabouCore.Equation.LE]
        assert sorted(ge_invariant_eq_idx + le_invariant_eq_idx) == list(range(len(self.inv_type)))
        all_ge_proved = False
        all_le_proved = False
        counter = 0
        while counter <= self.inductive_steps:
            counter += 1
            if not all_ge_proved:
                prev_proved = all_ge_proved
                all_ge_proved = self._proveInductiveAlphasOnce(invariant_oracle, ge_invariant_eq_idx)
                assert prev_proved in all_ge_proved
            if not all_le_proved:
                all_le_proved = self._proveInductiveAlphasOnce(invariant_oracle, le_invariant_eq_idx)

            if all_ge_proved and all_le_proved:
                # Make sure the invariants do not contrdicte each other (we have a lower bound and an upper bound on each cell)
                for i in range(0, len(self.alphas), 2):
                    # The invariant order is GE, LE, GE, LE ....
                    # But when we create an LE equation we multiply alpha by -1
                    assert self.alphas[i].alpha >= -1 * self.alphas[i + 1].alpha, [a.alpha for a in self.alphas]

                return True

        print("didn't find invariant")
        return False

    def __name__(self):
        return 'iterate_sgd'