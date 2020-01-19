from maraboupy.MarabouRNNMultiDim import alpha_to_equation, double_list
from maraboupy import MarabouCore
import random
from rnn_algorithms.Update_Strategy import Absolute_Step, Relative_Step

sign = lambda x: 1 if x >= 0 else -1


class AllAlphaSGDOneSideBound:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, inv_type, alpha_initial_value=0,
                 update_strategy=Absolute_Step()):
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

        self.first_call = True
        assert inv_type in [MarabouCore.Equation.LE, MarabouCore.Equation.GE]

        self.alphas = [alpha_initial_value] * len(initial_values)
        self.equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self.update_equation(i)

    def do_step(self, strengthen=True):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''
        self.first_call = False
        if strengthen:
            direction = -1
        else:
            direction = 1

        # i = self.next_idx_step

        # self.prev_alpha = self.alphas[self.next_idx_step]
        #  self.prev_idx = self.next_idx_step
        for i in range(len(self.alphas)):
            self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)
            self.update_equation(i)

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


class AllAlphasSGD:
    def __init__(self, rnnModel, xlim, alpha_initial_value=0,
                 update_strategy_ptr=Absolute_Step):
        '''
        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''


        self.next_is_max = True

        self.alpha_initial_value = alpha_initial_value
        self.update_strategy = update_strategy_ptr()
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs()
        initial_values = rnnModel.get_rnn_min_max_value_one_iteration(xlim)

        # The initial values are opposite to the intuition, for LE we use max_value
        self.min_invariants = AllAlphaSGDOneSideBound(initial_values[1], rnn_start_idxs, rnn_output_idxs,
                                                         MarabouCore.Equation.LE, alpha_initial_value,
                                                         self.update_strategy)
        self.max_invariants = AllAlphaSGDOneSideBound(initial_values[0], rnn_start_idxs, rnn_output_idxs,
                                                         MarabouCore.Equation.GE, alpha_initial_value,
                                                         self.update_strategy)
        self.last_fail = None
        self.alpha_history= []

    def name(self):
        return 'all_sgd_init{}_step{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)

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
