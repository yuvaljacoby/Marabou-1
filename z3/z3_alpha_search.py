from z3 import *
import math
import numpy as np
from tqdm import tqdm

def z3_real_to_float(z3_real):
    return float(z3_real.as_decimal(4).replace('?', ''))


def ReLU(after, before):
    # return after == before
    return after == If(before > 0, before, 0)


high_end = 100000
low_end = -10000
iteration = 0
# for iteration in tqdm(range(35)):
while high_end >= 0.0005 + low_end:
    iteration += 1
    cur_sum = (high_end + low_end) / 2
    print('iteration: {}, alpha sum to be less then: {}, high: {}, low: {}'.format(iteration, cur_sum, high_end, low_end))
    # 1d input, 2d rnn (r0, r1)
    s = Solver()
    s.push()

    # create weights and limit the input
    w_h = np.array([[0.1, -0.5], [0.2, 0.3]])
    w_i = np.array([1, 1])
    radius = 0.01
    x = [1]
    n_iterations = 10

    beta_1 = w_i[0] * (x[0] * (1 + radius))
    beta_2 = w_i[1] * (x[0] * (1 + radius))

    # Create variables
    in_tensor = Real('in')
    # n_bytes = math.ceil(math.log2(n_iterations))
    s.add(in_tensor >= x[0] * (1 - radius))
    s.add(in_tensor <= x[0] * (1 + radius))
    # i = BitVec('i', n_bytes)
    # s.add(i <= BitVecVal(n_iterations, n_bytes))
    # s.add(i >= BitVecVal(0, n_bytes))

    R0 = []
    R1 = []
    R0.append(Real('R0_0'))
    R1.append(Real('R1_0'))
    # R1_0 = Real('R1_0')
    s.add(R0[0] == 0)
    s.add(R1[0] == 0)

    alpha1 = Real('alpha1')
    alpha2 = Real('alpha2')
    for t in range(1, n_iterations):
        R0.append(Real('R0_{}'.format(t)))
        R1.append(Real('R1_{}'.format(t)))
        # TODO: Add ReLU to the update equations
        s.add(ReLU(R0[-1], R0[-2] * w_h[0, 0] + R1[-2] * w_h[0, 1] + in_tensor * w_i[0]))
        s.add(ReLU(R1[-1], R0[-2] * w_h[1, 0] + R1[-2] * w_h[1, 1] + in_tensor * w_i[1]))

        s.add(R0[-1] <= alpha1 * t + beta_1)
        s.add(R1[-1] <= alpha2 * t + beta_2)

    s.add(alpha1 + alpha2 <= cur_sum)
    # print(s.sexpr())
    # print(s.check())

    if (s.check() == sat):
        model = s.model()
        a1_val = z3_real_to_float(model[alpha1])
        a2_val = z3_real_to_float(model[alpha2])
        high_end = (a1_val + a2_val)
        print('iteration: {}, alpha1: {}, alpha2: {}'.format(iteration, a1_val, a2_val))
    else:
        low_end = cur_sum
