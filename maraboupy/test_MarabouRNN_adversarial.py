from maraboupy import MarabouCore
from maraboupy.MarabouRNN import *


def relu(num):
    return max(0, num)


def adversarial_robustness_sum_results(x):
    '''
    Get a list of inputs and calculate A,B according to the robustness network
    :param x:
    :return:
    '''

    s_i_1_f = 0
    z_i_1_f = 0

    for num in x:
        s_i_f = relu(2 * num + 1 * s_i_1_f)
        z_i_f = relu(1 * num + 1 * z_i_1_f)
        s_i_1_f = s_i_f
        z_i_1_f = z_i_f

    A = 2 * s_i_f  # + z_i_f
    B = 2 * z_i_f  # + s_i_f

    return A, B


def z3_adversarial_unfold(n=10, to_pass=True):
    from z3 import Reals, Solver, sat
    s = Solver()
    A0, B0 = Reals('A0 B0')
    s.add(A0 == 0)
    # if to_pass:
    s.add(B0 == n)
    #
    # s.add(B0 == n+1)

    A_prev = A0
    B_prev = B0
    for i in range(1, n + 1):
        A_temp, B_temp = Reals('A{} B{}'.format(i, i))
        # This is the invariant
        if to_pass:
            s.add(A_temp == A_prev + 1)
        else:
            s.add(A_temp == A_prev + 0.9)
        s.add(B_temp == B_prev)
        A_prev = A_temp
        B_prev = B_temp

    s.add(A_temp < B_temp)

    # print(s)
    t = s.check()
    if t == sat:
        # print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def check_adversarial_robustness_z3():
    from z3 import Reals, Int, Solver, If, And
    sk, sk_1, zk, zk_1 = Reals('sk sk_1 zk zk_1')
    i = Int('i')

    s = Solver()
    s.add(And(i >= 0, i <= 20, sk_1 >= 0, sk >= 0, zk >= 0, zk_1 >= 0))
    A = If(sk * 1 >= 0, sk * 1, 0)
    B = If(zk * 1 >= 0, zk * 1, 0)

    s.add(If(i == 0,
             And(sk >= 0, sk <= 3, zk >= 10, zk <= 21, sk_1 == 0, zk_1 == 0),
             sk - zk >= sk_1 - zk_1 + 21 / i))

    s.add(And(A < B, i == 20))
    # # we negate the condition, instead if for all sk condition we check if there exists sk not condition
    # s.add(sk_ReLU * w > ylim)

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def define_adversarial_robustness_step_fail(xlim, n_iterations):
    '''
    Define an adversarial roustness examples
    0 <= x_0 <= 1
    1 <= x_1 <= 2
    s_i = 1 * x_0 + 5 * x_1 + 1 * s_(i-1)
    z_i = 2 * x_0 + 1 * x_1 + 1.1 * z_(i-1)
    A = s_i
    B = z_i
    therefore:
        5 <= A <= 6
        1 <= B <= 4
    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1, x2

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # x2
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    s_hidden_w = 1
    z_hidden_w = 100
    x0_s_w = 1
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (2 * n_iterations) + max_b
    print('min_a', min_a)
    print('max_b', max_b)
    print('initial_diff', initial_diff)
    print('alpha', alpha)


    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, s_i_f_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)

    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, z_i_f_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), (-alpha, alpha)


def define_adversarial_robustness(xlim, n_iterations):
    '''
    Define an adversarial roustness examples
    0 <= x_0 <= 1
    1 <= x_1 <= 2
    s_i = 1 * x_0 + 5 * x_1 + 1 * s_(i-1)
    z_i = 2 * x_0 + 1 * x_1 + 1.1 * z_(i-1)
    A = s_i
    B = z_i
    therefore:
        5 <= A <= 6
        1 <= B <= 4
    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1, x2

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # x2
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    s_hidden_w = 1
    z_hidden_w = 1
    x0_s_w = 1
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (2 * n_iterations) + max_b
    print('min_a', min_a)
    print('max_b', max_b)
    print('initial_diff', initial_diff)
    print('alpha', alpha)

    # a_i >= a_0 - (alpha + max_b) * i
    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, s_i_f_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)
    # invariant_equation.dump()

    # b_i <= b_0 + (alpha + max_b) * i
    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, z_i_f_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), (-alpha, alpha)


def define_weak_adversarial_robustness(xlim, n_iterations):
    '''
    Defines adversarial robustness where the property cannot be derived from the invariants.
    We expect that the invariant will hold but the property will not
    :param xlim:
    :param n_iterations:
    :return:
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1, x2

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # x2
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    s_hidden_w = 1
    z_hidden_w = 1
    x0_s_w = 1
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (n_iterations * 2) + max_b
    print('min_a', min_a)
    print('max_b', max_b)
    print('initial_diff', initial_diff)
    print('alpha', alpha)


    # a_i - b_i >= a_i-1 - b_i-1 + alpha <--> a_i - a_i-1 - b_i + b_i-1 >= alpha
    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, a_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)
    # invariant_equation.dump()

    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, b_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), (alpha, alpha + 100)


def define_sum_adversarial_robustness(xlim, ylim, n_iterations):
    '''
    Defines the zero network in a marabou way
    The zero network is a network with two rnn cells, that always outputs zero
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network, will effect how we create the invariant
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(1)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    # s_i_f = relu(2 * x + s_i-1_f)
    s_cell_iterator = 1  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 2)], 1, n_iterations)
    # z_i_f = relu(x + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    network.setNumberOfVariables(a_idx + 2)  # +2 for A, B

    # A
    network.setLowerBound(a_idx, -large)
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)
    network.setUpperBound(b_idx, large)

    # i = i, we create iterator for each cell, make sure they are the same
    iterator_equation = MarabouCore.Equation()
    iterator_equation.addAddend(1, s_cell_iterator)
    iterator_equation.addAddend(-1, z_cell_iterator)
    iterator_equation.setScalar(0)
    network.addEquation(iterator_equation)

    # A = 2*skf <--> A - 2*skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-2, s_i_f_idx)
    a_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(a_output_eq)

    # B = 2*zkf <--> B - 2*z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-2, z_i_f_idx)
    b_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(b_output_eq)

    # s_i_f <= 2*z_i_f <--> s_i_f - 2*z_i_f <= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, s_i_f_idx)  # s_i f
    invariant_equation.addAddend(-2, z_i_f_idx)  # z_i f
    invariant_equation.setScalar(0)

    # z_i_b <= z_i_f <-- z_i_b - z_i_f <= 0
    # TODO: This is stupid we don't need this
    temp_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    temp_eq.addAddend(1, 7)  # s_i f
    temp_eq.addAddend(-1, 8)  # z_i f
    temp_eq.setScalar(0)
    network.addEquation(temp_eq)

    # A <= 2* B <--> A - 2*B <= 0
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, a_idx)
    property_eq.addAddend(-2, b_idx)
    property_eq.setScalar(0)

    return network, [s_cell_iterator, z_cell_iterator], invariant_equation, [property_eq]


def define_bias_sum_adversarial(xlim, ylim, n_iterations):
    '''
    A = 100 (0*x + 100)
    z_k_f = sum(x_i)
    B = 1 * z_k_f
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network, will effect how we create the invariant
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(2)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    # A
    a_idx = 1
    network.setLowerBound(a_idx, -large)
    network.setUpperBound(a_idx, large)

    # A = 0 * x + 100 <-- > A == 100
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.setScalar(100)
    # output_equation.dump()
    network.addEquation(a_output_eq)

    # s_i_f = relu(x + s_i-1_f)
    s_cell_iterator = 2  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    b_idx = s_i_f_idx + 1

    network.setNumberOfVariables(b_idx + 1)  # for B

    # B
    network.setLowerBound(b_idx, -large)
    network.setUpperBound(b_idx, large)

    # B = 1*zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-1, s_i_f_idx)
    b_output_eq.setScalar(0)
    # output_equation.dump()
    network.addEquation(b_output_eq)

    # b_i <= i <--> b_i - i <= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, b_idx)
    invariant_equation.addAddend(-1, s_cell_iterator)
    invariant_equation.setScalar(0)

    # B <= A <--> B- A <= 0
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, b_idx)
    property_eq.addAddend(-1, a_idx)
    property_eq.setScalar(0)

    return network, [s_cell_iterator], invariant_equation, [property_eq]


def test_sum_adversarial():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert prove_using_invariant(invariant_xlim, None, num_iterations, define_sum_adversarial_robustness)


def test_bias_sum_adversarial_fail():
    num_iterations = 105
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert not prove_using_invariant(invariant_xlim, None, num_iterations, define_bias_sum_adversarial)


def test_bias_sum_adversarial():
    num_iterations = 99
    invariant_xlim = (-1, 1)
    # y_lim = 10 ** -2
    assert prove_using_invariant(invariant_xlim, None, num_iterations, define_bias_sum_adversarial)


def test_adversarial_robustness():
    num_iterations = 10
    invariant_xlim = [(0, 1), (1, 2)]
    # y_lim = 10 ** -2
    assert prove_adversarial_using_invariant(invariant_xlim, num_iterations, define_adversarial_robustness)


def test_adversarial_robustness_step_fail():
    num_iterations = 10
    invariant_xlim = [(0, 1), (1, 2)]
    # y_lim = 10 ** -2
    assert not prove_adversarial_using_invariant(invariant_xlim, num_iterations, define_adversarial_robustness_step_fail)


def test_adversarial_robustness_base_fail():
    num_iterations = 10
    invariant_xlim = [(0, 10), (1, 2)]
    # y_lim = 10 ** -2
    assert not prove_adversarial_using_invariant(invariant_xlim, num_iterations, define_adversarial_robustness)

    # assert check_adversarial_robustness_z3()


def test_adversarial_robustness_conclusion_fail():
    num_iterations = 100
    invariant_xlim = [(0, 1), (1, 2)]
    # y_lim = 10 ** -2
    assert not prove_adversarial_using_invariant(invariant_xlim, num_iterations, define_weak_adversarial_robustness)


def test_z3_adversarial_robustness():
    a_pace = -1
    b_pace = 1
    min_a = 5
    max_b = 0
    n_iterations = (min_a // 2) - 1
    print("\n\n", n_iterations)
    assert prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)


# def test_z3_adversarial_robustness_fail():
#     a_pace = -1
#     b_pace = 1
#     min_a = 5
#     max_b = 0
#     n_iterations = 3
#     assert prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)
