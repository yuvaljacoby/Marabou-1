from RNN.MarabouRNNMultiDim import alpha_to_equation
from maraboupy import MarabouCore
from RNN.MarabouRnnModel import RnnMarabouModel

import random
import numpy as np
from rnn_algorithms.Update_Strategy import Absolute_Step

sign = lambda x: 1 if x >= 0 else -1

MAX_IDX_IN_A_ROW = 4

# def relative_step(threshold_min=0.2, threshold_max = 1e5, relative_step_size=0.3, init_after_threshold=0.5):
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
#         if abs(alpha) > threshold_min and abs(alpha) < threshold_max :
#             return alpha + (
#                     direction * alpha * relative_step_size * sign(alpha))  # do step size 0.3 to the next direction
#         elif abs(alpha) < threshold_min:
#             return init_after_threshold * direction
#         elif abs(alpha) > threshold_max:
#             return alpha + (
#                     direction * alpha * 0.001 * sign(alpha))  # do step size 0.3 to the next direction
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


class InverseWeightedAlphaSGDOneSideBound:
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
        self.idx_history = []

    def choose_idx_to_update(self, influence_sum):
        # if influence_sum != []:
        #     self.next_idx_step = np.argmax(influence_sum)
        # else:
        #     self.next_idx_step = random.randint(0, len(self.alphas) - 1)

        influence_sum = np.abs(influence_sum)
        # inverse the probability
        if np.sum(influence_sum) > 0:
            influence_sum = 1 / influence_sum
            self.next_idx_step = np.random.choice(range(len(influence_sum)), 1, p=influence_sum / np.sum(influence_sum))[0]
        else:
            self.next_idx_step = random.randint(0, len(self.alphas) - 1)

        # Override the selection?
        if len(self.idx_history) >= MAX_IDX_IN_A_ROW:
            while all([self.idx_history[i] == self.next_idx_step for i in range(MAX_IDX_IN_A_ROW)]):
                self.next_idx_step = random.randint(0, len(self.alphas) - 1)
        self.idx_history.append(self.next_idx_step)
        # assert all([self.idx_history[i] == self.next_idx_step for i in range(MAX_IDX_IN_A_ROW + 1)])

        if len(self.idx_history) > MAX_IDX_IN_A_ROW:
            self.idx_history.pop(0)
        return self.next_idx_step

    def do_step(self, strengthen=True, influnce_sum=[]):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :param invariants_results: Boolean array with results of the previous round of proving invariants, if failed in property can be None
        :return list of invariant equations (that still needs to be proven)
        '''
        assert len(influnce_sum) <= len(self.alphas)
        self.first_call = False
        if strengthen:
            direction = -1
        else:
            direction = 1

        i = self.choose_idx_to_update(influnce_sum)

        self.prev_alpha = self.alphas[self.next_idx_step]
        self.alphas[i] = self.update_strategy.do_step(self.alphas[i], direction)

        self.update_equation(i)
        self.prev_idx = i
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


class InverseWeightedAlphasSGD:
    def __init__(self, rnnModel:RnnMarabouModel, xlim, alpha_initial_value=0,
                 update_strategy_ptr=Absolute_Step):
        '''
        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''
        self.next_is_max = True

        self.rnn_hidden_weights = rnnModel.rnn_weights[0][1]

        self.alpha_initial_value = alpha_initial_value

        self.update_strategy = update_strategy_ptr()

        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs()
        initial_values = rnnModel.get_rnn_min_max_value_one_iteration(xlim)

        # The initial values are opposite to the intuition, for LE we use max_value
        self.min_invariants = InverseWeightedAlphaSGDOneSideBound(initial_values[1], rnn_start_idxs, rnn_output_idxs,
                                                           MarabouCore.Equation.LE, alpha_initial_value,
                                                           self.update_strategy)
        self.max_invariants = InverseWeightedAlphaSGDOneSideBound(initial_values[0], rnn_start_idxs, rnn_output_idxs,
                                                           MarabouCore.Equation.GE, alpha_initial_value,
                                                           self.update_strategy)
        self.last_fail = None
        self.alpha_history = []

    def name(self):
        return 'iterate_sgd_init{}_step{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)

    def do_step(self, strengthen=True, invariants_results=[]):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :param invariants_results: Boolean array with results of the previous round of proving invariants, if not passing, using all weights to calculate next value
        :return list of invariant equations (that still needs to be proven)
        '''

        # TODO: If this condition is true it means the last step we did was not good, and we can decide what to do next
        #  (for example revert, and once passing all directions do a big step)
        if self.last_fail == strengthen:
            pass
        self.last_fail = strengthen

        if invariants_results != [] and invariants_results is not None:
            # Keep only the rows where we did not prove invariants, and sum the columns
            min_invariants_results = invariants_results[len(self.min_invariants.alphas):]
            max_invariants_results = invariants_results[:len(self.min_invariants.alphas)]

            # If we all invariants from above or bottom are done do step in the other
            # if all(min_invariants_results):
            #     self.next_is_max = True
            # elif all(max_invariants_results):
            #     self.next_is_max = False

        else:
            min_invariants_results = [True] * len(self.min_invariants.alphas)
            max_invariants_results = [True] * len(self.max_invariants.alphas)



        #  The equations order matters, keep it min and then max (we use it in the invariants_results)
        if self.next_is_max:
            max_hidden_weights_no_invariants = np.sum(self.rnn_hidden_weights[max_invariants_results, :], axis=0)
            res = self.min_invariants.get_equations() + self.max_invariants.do_step(strengthen, max_hidden_weights_no_invariants)
        else:
            min_hidden_weights_no_invariants = np.sum(self.rnn_hidden_weights[min_invariants_results, :], axis=0)
            res = self.min_invariants.do_step(strengthen, min_hidden_weights_no_invariants) + self.max_invariants.get_equations()

        self.alpha_history.append(self.get_alphas())
        # Alternate between lower and upper bound
        self.next_is_max = not self.next_is_max
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
        return self.min_invariants.get_equations() + self.max_invariants.get_equations()

    def get_alphas(self):
        return self.min_invariants.get_alphas() + self.max_invariants.get_alphas()
