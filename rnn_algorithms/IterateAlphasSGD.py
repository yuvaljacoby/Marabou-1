from maraboupy.MarabouRNNMultiDim import alpha_to_equation, double_list
from maraboupy import MarabouCore


sign = lambda x: 1 if x >= 0 else -1


def relative_step(threshold=0.2, relative_step_size=0.3, init_after_threshold=0.5):
    '''
    Create a function that is doing a relative step in the form:
    if alpha <= threshold:
        return init_after_threshold * direction
    else:
        return alphas + step
    where step is: direction * alpha * relative_step_size * sign(alpha)
    :return: function pointer that given alpha and direction returning the new alpha
    '''

    def do_relative_step(alpha, direction):
        if abs(alpha) > threshold:
            return alpha + (
                    direction * alpha * relative_step_size * sign(alpha))  # do step size 0.3 to the next direction
        else:
            return init_after_threshold * direction

    return do_relative_step


def absolute_step(step_size=0.1):
    '''
     Create a function that is doing an absolute step in the form:
     alpha + (step_size * direction)
     :return: function pointer that given alpha and direction returning the new alpha
     '''

    def do_relative_step(alpha, direction):
        return alpha + (direction * step_size)

    return do_relative_step


class IterateAlphaSGDOneSideBound:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, inv_type, alpha_initial_value=0,
                 alpha_step_policy=relative_step()):
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
        self.alpha_step_func = alpha_step_policy

        self.first_call = True
        assert inv_type in [MarabouCore.Equation.LE, MarabouCore.Equation.GE]

        self.alphas = [alpha_initial_value] * len(initial_values)
        self.equations = [None] * len(self.alphas)
        for i in range(len(self.alphas)):
            self.update_equation(i)

    def update_next_idx_step(self):
        self.next_idx_step = (self.next_idx_step + 1) % len(self.alphas)
        return self.next_idx_step

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

        i = self.next_idx_step

        self.prev_alpha = self.alphas[self.next_idx_step]
        self.prev_idx = self.next_idx_step
        self.alphas[i] = self.alpha_step_func(self.alphas[i], direction)

        self.update_equation(i)
        self.update_next_idx_step()
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


class IterateAlphasSGD:
    def __init__(self, initial_values, rnn_start_idxs, rnn_output_idxs, alpha_initial_value=0,
                 alpha_step_policy_ptr=relative_step):
        '''
        :param initial_values: tuple of lists, [0] is list of min values, [1] for max values
        :param rnn_start_idxs:
        :param rnn_output_idxs:
        '''

        # The initial values are opposite to the intuition, for LE we use max_value
        self.next_is_max = True

        self.alpha_initial_value = alpha_initial_value
        self.alpha_step_policy_ptr = alpha_step_policy_ptr
        self.alpha_step_policy = self.alpha_step_policy_ptr()

        self.min_invariants = IterateAlphaSGDOneSideBound(initial_values[1], rnn_start_idxs, rnn_output_idxs,
                                                          MarabouCore.Equation.LE, alpha_initial_value,
                                                          self.alpha_step_policy)
        self.max_invariants = IterateAlphaSGDOneSideBound(initial_values[0], rnn_start_idxs, rnn_output_idxs,
                                                          MarabouCore.Equation.GE, alpha_initial_value,
                                                          self.alpha_step_policy)
        self.last_fail = None

    def name(self):
        return 'iterate_sgd_init{}_step{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__  )

    def do_step(self, strengthen=True):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert, weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''

        # TODO: If this condition is true it means the last step we did was not good, and we can decide what to do next
        #  (for example revert, and once passing all directions do a big step)
        if self.last_fail == strengthen:
            pass
        self.last_fail = strengthen

        if self.next_is_max:
            res = self.max_invariants.do_step(strengthen) + self.min_invariants.get_equations()
        else:
            res = self.min_invariants.do_step(strengthen) + self.max_invariants.get_equations()

        self.next_is_max = not (self.next_is_max)
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
