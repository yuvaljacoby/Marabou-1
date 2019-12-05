from maraboupy import MarabouCore
import tensorflow as tf
import numpy as np
from maraboupy.MarabouRNNMultiDim import prove_multidim_property, add_rnn_multidim_cells, SGDAlphaAlgorithm, \
    negate_equation

H5_FILE_PATH = './model_classes20_1rnn8_0_64_100.h5'
SMALL = 10 ** -2
LARGE = 5000


def relu(x):
    return max(x, 0)


class RnnMarabouModel():
    def __init__(self, h5_file_path, n_iterations=10):
        self.network = MarabouCore.InputQuery()
        self.model = tf.keras.models.load_model(h5_file_path)
        # TODO: If the input is 2d wouldn't work
        n_input_nodes = self.model.input_shape[-1]
        prev_layer_idx = list(range(0, n_input_nodes))
        self.input_idx = prev_layer_idx
        self.n_iterations = n_iterations
        self.rnn_out_idx = []

        # save spot for the input nodes
        self.network.setNumberOfVariables(n_input_nodes)
        for layer in self.model.layers:
            if type(layer) == tf.keras.layers.SimpleRNN:
                prev_layer_idx = self.add_rnn_simple_layer(layer, prev_layer_idx)
                self.rnn_out_idx += prev_layer_idx
            elif type(layer) == tf.keras.layers.Dense:
                prev_layer_idx = self.add_dense_layer(layer, prev_layer_idx)
            else:
                #
                raise NotImplementedError("{} layer is not supported".format(layer.name))

        # Save the last layer output indcies
        self.output_idx = list(range(*prev_layer_idx))
        self._rnn_loop_idx = [i - 3 for i in self.rnn_out_idx]
        self._rnn_prev_iteration_idx = [i - 2 for i in self.rnn_out_idx]

    def _get_value_by_layer(self, layer, input_tensor):
        from tensorflow.keras.models import Model
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=layer.output)
        intermediate_output = intermediate_layer_model.predict(input_tensor)

        # assert len(input_tensor.shape) == len(self.model.input_shape)
        # assert sum([input_tensor.shape[i] == self.model.input_shape[i] or self.model.input_shape[i] is None
        #         for i in range(len(input_tensor.shape))]) == len(input_tensor.shape)
        # out_ten = layer.output
        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()
        #     output = sess.run(out_ten, {self.model.input: input_tensor})

        return np.squeeze(intermediate_output)

    def _get_rnn_value_one_iteration(self, in_tensor):
        outputs = []
        for l in self.model.layers:
            if type(l) == tf.keras.layers.SimpleRNN:
                outputs.append(self._get_value_by_layer(l, in_tensor))
        return outputs

    def find_max_by_weights(self, xlim, layer):
        total = 0
        for i, w in enumerate(layer.get_weights()[0]):
            if w < 0:
                total += xlim[i][0] * w
            else:
                total += xlim[i][1] * w
        return total

    def get_rnn_max_value_one_iteration(self, xlim):
        xlim_max = [x[1] for x in xlim]
        max_in_tensor = np.array(xlim_max)[None, None, :]  # add two dimensions
        return self._get_rnn_value_one_iteration(max_in_tensor)

    def get_rnn_min_max_value_one_iteration(self, xlim):
        for i, layer in enumerate(self.model.layers):
            initial_values = []
            if type(layer) == tf.keras.layers.SimpleRNN:
                if i == 0:
                    # There is no non linarity and the constraints are simple just take upper bound if weight is positive
                    #  and lower otherwise

                    # It's only one iteration so the hidden weights (and bias) is zeroed
                    # TODO: Is the bias zero?
                    in_w, _, _ = layer.get_weights()
                    for rnn_dim_weights in in_w.T:
                        max_val = 0
                        min_val = 0
                        for j, w in enumerate(rnn_dim_weights):
                            w = round(w, 6)
                            if w > 0:
                                max_val += w * xlim[j][1]
                                min_val += w * xlim[j][0]
                            else:
                                max_val += w * xlim[j][0]
                                min_val += w * xlim[j][1]
                        # TODO: +- SMALL is not ideal here (SMALL = 10**-2) but otherwise there are rounding problems
                        initial_values.append((relu(min_val) - SMALL, relu(max_val) + SMALL))
                else:
                    # Need to query gurobi here...
                    raise NotImplementedError()
            print('initial_values:', initial_values)
            return initial_values

    def get_rnn_min_max_value_one_iteration_marabou(self, xlim):
        xlim_min = [x[0] for x in xlim]
        min_in_tensor = np.array(xlim_min)[None, None, :]  # add two dimensions
        xlim_max = [x[1] for x in xlim]
        max_in_tensor = np.array(xlim_max)[None, None, :]  # add two dimensions
        # TODO: remove the assumption we have only one layer of RNN for initial_values (do we need to remove it?)
        max_values = self._get_rnn_value_one_iteration(max_in_tensor)[0]
        min_values = self._get_rnn_value_one_iteration(min_in_tensor)[0]
        # Not sure that the max_in tensor will yield the cell value to be larger then the min_in_tensor
        initial_values = []
        for i in range(len(max_values)):
            if max_values[i] >= min_values[i]:
                initial_values.append((min_values[i], max_values[i]))
            else:
                initial_values.append((max_values[i], min_values[i]))

        # def create_initial_run_equations(loop_indices, rnn_prev_iteration_idx):
        #     '''
        #     Zero the loop indcies and the rnn hidden values (the previous iteration output)
        #     :return: list of equations to add to marabou
        #     '''
        #     loop_equations = []
        #     for i in loop_indices:
        #         loop_eq = MarabouCore.Equation()
        #         loop_eq.addAddend(1, i)
        #         loop_eq.setScalar(0)
        #         loop_equations.append(loop_eq)
        #
        #     # s_i-1 f == 0
        #     zero_rnn_hidden = []
        #     for idx in rnn_prev_iteration_idx:
        #         base_hypothesis = MarabouCore.Equation()
        #         base_hypothesis.addAddend(1, idx)
        #         base_hypothesis.setScalar(0)
        #         zero_rnn_hidden.append(base_hypothesis)
        #     return loop_equations + zero_rnn_hidden
        #
        # def improve_beta(eq):
        #     '''
        #     Run the equation on marabou until it is satisfied.
        #     If not satisfied taking the value from the index and using it as a s scalar
        #     using self.network to verify
        #     :param eq: Marabou equation of the form: +-1.000xINDEX >= SCALAR
        #     :return: a scalar that satisfies the equation
        #     '''
        #     proved = False
        #     assert len(eq.getAddends()) == 1
        #     idx = eq.getAddends()[0].getVariable()
        #     beta = eq.getScalar()
        #     while not proved:
        #         eq.setScalar(beta)
        #         self.network.addEquation(eq)
        #         vars1, stats1 = MarabouCore.solve(self.network, "", 0, 0)
        #         if len(vars1) > 0:
        #             proved = False
        #             beta = vars1[idx] + SMALL
        #             print("proof fail, trying with beta: {}".format(beta))
        #         else:
        #             # print("UNSAT")
        #             proved = True
        #             print("proof worked, with beta: {}".format(beta))
        #             self.network.dump()
        #             eq.dump()
        #             beta = beta + (10 * SMALL)
        #         self.network.removeEquation(eq)
        #     return beta
        #
        # def query_marabou_to_improve_values():
        #     initial_run_eq = create_initial_run_equations(self._rnn_loop_idx, self._rnn_prev_iteration_idx)
        #     for init_eq in initial_run_eq:
        #         self.network.addEquation(init_eq)
        #
        #     # not(R_i_f >= beta) <-> -R_i_f >= beta - epsilon
        #     for i in range(len(max_values)):
        #         beta_eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        #         beta_eq2.addAddend(1, self.rnn_out_idx[i])
        #         beta_eq2.setScalar(min_values[i] + SMALL)
        #         min_val = improve_beta(beta_eq2)
        #
        #         beta_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        #         beta_eq.addAddend(1, self.rnn_out_idx[i])
        #         beta_eq.setScalar(max_values[i] - SMALL)
        #         max_val = improve_beta(beta_eq)
        #
        #         initial_values[i] = (min_val, max_val)
        #
        #     for init_eq in initial_run_eq:
        #         self.network.removeEquation(init_eq)

        # query_marabou_to_improve_values()
        return initial_values

    def add_rnn_simple_layer(self, layer, prev_layer_idx):
        '''
        Append to the marabou encoding (self.netowrk) a SimpleRNN layer
        :param layer: The layer object (using get_weights, expecting to get list length 3 of ndarrays)
        :param prev_layer_idx: Marabou indcies of the previous layer (to use to create the equations)
        :return: List of the added layer output variables
        '''
        if layer.activation == tf.keras.activations.relu:
            # Internal layer
            pass
        elif layer.activation == tf.keras.activations.softmax:
            # last layer
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))
        else:
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))

        rnn_input_weights, rnn_hidden_weights, rnn_bias_weights = layer.get_weights()
        assert rnn_hidden_weights.shape[0] == rnn_input_weights.shape[1]
        assert rnn_hidden_weights.shape[0] == rnn_hidden_weights.shape[1]
        assert rnn_hidden_weights.shape[1] == rnn_bias_weights.shape[0]
        # first_idx = self.network.getNumberOfVariables()
        # self.network.setNumberOfVariables(first_idx + rnn_hidden_weights.shape[1])

        # TODO: Get number of iterations and pass it here instead of the 10
        output_idx = add_rnn_multidim_cells(self.network, prev_layer_idx, rnn_input_weights.T, rnn_hidden_weights,
                                            rnn_bias_weights, self.n_iterations)
        return output_idx

    def add_dense_layer(self, layer, prev_layer_idx):
        output_weights, output_bias_weights = layer.get_weights()
        assert output_weights.shape[1] == output_bias_weights.shape[0]

        def add_last_layer_equations():
            first_idx = self.network.getNumberOfVariables()
            self.network.setNumberOfVariables(first_idx + output_weights.shape[1])

            for i in range(output_weights.shape[1]):
                self.network.setLowerBound(i, -LARGE)
                self.network.setUpperBound(i, LARGE)
                eq = MarabouCore.Equation()
                for j, w in enumerate(output_weights[:, i]):
                    eq.addAddend(w, prev_layer_idx[j])
                eq.setScalar(-output_bias_weights[i])
                eq.addAddend(-1, first_idx + i)
                self.network.addEquation(eq)
            return first_idx, first_idx + output_weights.shape[1]

        def add_intermediate_layer_equations():
            first_idx = self.network.getNumberOfVariables()
            # times 2 for the b and f variables
            self.network.setNumberOfVariables(first_idx + (output_weights.shape[1] * 2))
            b_indices = range(first_idx, first_idx + (output_weights.shape[1] * 2), 2)
            f_indices = range(first_idx + 1, first_idx + (output_weights.shape[1] * 2), 2)
            for i in range(output_weights.shape[1]):
                cur_b_idx = b_indices[i]
                cur_f_idx = f_indices[i]
                # b variable
                self.network.setLowerBound(cur_b_idx, -LARGE)
                self.network.setUpperBound(cur_b_idx, LARGE)
                # f variable
                self.network.setLowerBound(cur_f_idx, 0)
                self.network.setUpperBound(cur_f_idx, LARGE)

                MarabouCore.addReluConstraint(self.network, cur_b_idx, cur_f_idx)
                # b equation
                eq = MarabouCore.Equation()
                for j, w in enumerate(output_weights[:, i]):
                    eq.addAddend(w, prev_layer_idx[j])
                eq.setScalar(-output_bias_weights[i])
                eq.addAddend(-1, cur_b_idx)
                self.network.addEquation(eq)
            return f_indices

        if layer.activation == tf.keras.activations.relu:
            # Internal layer
            return add_intermediate_layer_equations()
            # raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))
        elif layer.activation == tf.keras.activations.softmax:
            # last layer
            return add_last_layer_equations()
        else:
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))

    def set_input_bounds(self, xlim: list):
        '''
        set bounds on the input variables
        :param xlim: list of tuples each tuple is (lower_bound, upper_bound)
        '''
        assert len(xlim) == len(self.input_idx)
        for i, marabou_idx in enumerate(self.input_idx):
            self.network.setLowerBound(marabou_idx, xlim[i][0])
            self.network.setUpperBound(marabou_idx, xlim[i][1])

    def set_input_bounds_template(self, xlim: list, radius: float):
        '''
        set bounds on the input variables
        For example if xlim is [(5,3)], and radius is 0.1, the limit will be:
        0.9 * (5i + 3) <= x[0] <= 1.1 * (5i + 3)
        :param xlim: list of tuples, each tuple is (alpha, beta) which will be used as alpha * i + beta
        :param radius: non negative number, l_infinity around each of the points
        '''
        assert radius >= 0
        assert len(xlim) == len(self.input_idx)
        assert len(self._rnn_loop_idx) > 0
        u_r = 1 + radius  # upper radius
        l_r = 1 - radius  # lower radius
        i_idx = self._rnn_loop_idx[0]
        for i, marabou_idx in enumerate(self.input_idx):
            alpha, beta = xlim[i]
            self.network.setLowerBound(marabou_idx, -LARGE)
            self.network.setUpperBound(marabou_idx, LARGE)
            # TODO: What if alpha / beta == 0?
            # x <= r * alpha * i + r * beta <--> x - r * alpha * i <= r * beta
            ub_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
            ub_eq.addAddend(1, marabou_idx)
            ub_eq.addAddend(-u_r * alpha, i_idx)
            ub_eq.setScalar(u_r * beta)
            self.network.addEquation(ub_eq)

            # x >= r * alpha * i + r * beta <--> x - r * alpha * i >= r * beta
            lb_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
            lb_eq.addAddend(1, marabou_idx)
            lb_eq.addAddend(-l_r * alpha, i_idx)
            lb_eq.setScalar(l_r * beta)
            self.network.addEquation(lb_eq)


def calc_min_max_template_by_radius(x: list, radius: float):
    '''
    Calculate bounds of first iteration when the input vector is by template (i == 0)
    :param x: list of tuples, each tuple is (alpha, beta) which will be used as alpha * i + beta
    :param radius: non negative number, l_infinity around each of the points
    :return: xlim -  list of tuples each tuple is (lower_bound, upper_bound)
    '''
    assert radius >= 0
    xlim = []
    for (alpha, beta) in x:
        if beta != 0:
            xlim.append((beta * (1 - radius), beta * (1 + radius)))
        else:
            xlim.append((-radius, radius))
    return xlim


def calc_min_max_by_radius(x, radius):
    '''
    :param x: base_vector (input vector that we want to find a ball around it), need to be valid shape for the model
    :param radius: determines the limit of the inputs around the base_vector, non negative number
    :return: xlim -  list of tuples each tuple is (lower_bound, upper_bound)
    '''
    assert radius >= 0
    xlim = []
    for val in x:
        if val != 0:
            xlim.append((val * (1 - radius), val * (1 + radius)))
        else:
            xlim.append((-radius, radius))
    return xlim


def assert_adversarial_query(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test=False):
    '''
    Make sure that when running the network with input x the output at index y_idx_max is larger then other_idx
    if_fail_test negates this in order to let failing tests run
    :param x: 3d input vector where (1, n_iteration, input_dim), or 1d vector (with only input_dim), or list of input values
    :param y_idx_max: max index in the out vector
    :param other_idx: other index in the out vector
    :param h5_file_path: path to h5 file with the network
    :param n_iterations:  number of iterations. if x is a 3d vector ignoring
    :param is_fail_test: to negate the asseration or not
    :return: assert that predict(x)[y_idx_max] >= predict(x)[other_idx]
    '''
    model = tf.keras.models.load_model(h5_file_path)
    if type(x) == list:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.repeat(x[None, None, :], n_iterations, axis=1)
        # x = np.repeat(x, n_iterations).reshape(1, n_iterations, -1)

    out_vec = np.squeeze(model.predict(x))
    res = out_vec[y_idx_max] >= out_vec[other_idx]
    res = not res if is_fail_test else res
    assert res, out_vec


def assert_adversarial_query_template(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test=False):
    '''
    Make sure that when running the network with input x the output at index y_idx_max is larger then other_idx
    if_fail_test negates this in order to let failing tests run
    :param x: 3d input vector where (1, n_iteration, input_dim), or 1d vector (with only input_dim), or list of input values
    :param y_idx_max: max index in the out vector
    :param other_idx: other index in the out vector
    :param h5_file_path: path to h5 file with the network
    :param n_iterations:  number of iterations. if x is a 3d vector ignoring
    :param is_fail_test: to negate the asseration or not
    :return: assert that predict(x)[y_idx_max] >= predict(x)[other_idx]
    '''
    model = tf.keras.models.load_model(h5_file_path)
    if type(x) == list:
        x = template_to_vector(x, n_iterations)

    out_vec = np.squeeze(model.predict(x))
    res = out_vec[y_idx_max] >= out_vec[other_idx]
    res = not res if is_fail_test else res
    assert res, out_vec


def get_output_vector(h5_file_path: str, x: list, n_iterations: int):
    '''
    predict on the model using x (i.e. the matrix will be of shape (1, n_iterations, len(x))
    :param h5_file_path: path to the model file
    :param x: list of values
    :param n_iterations: number of iteration to create
    :return: ndarray, shape is according to the model output
    '''
    model = tf.keras.models.load_model(h5_file_path)
    tensor = np.repeat(np.array(x)[None, None, :], n_iterations, axis=1)
    return model.predict(tensor)


def template_to_vector(x: list, n_iterations: int):
    '''
    Create a np.array dimension [1, n_iterations, len(x)] according to the template in x
    :param x: list of tuples, where a tuple is (alpha, beta) and the value will be alpha * i + beta (where i is time)
    :param n_iterations:
    :return: np.array dim: [1, n_iterations, len(x)]
    '''

    beta = np.array([t[1] for t in x])
    alpha = np.array([t[0] for t in x])
    tensor = np.zeros(shape=(1, len(x))) + beta
    for i in range(1, n_iterations):
        vec = alpha * i + beta
        tensor = np.vstack((tensor, vec))
    return tensor[None, :, :]


def get_output_vector_template(h5_file_path: str, x: list, n_iterations: int):
    model = tf.keras.models.load_model(h5_file_path)
    tensor = template_to_vector(x, n_iterations)
    return model.predict(tensor)


def adversarial_query_template(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str,
                               n_iterations=10, is_fail_test=False):
    '''
    Query marabou with adversarial query
    :param x: template of the input vector, each cell is a tuple (alpha, beta) which will limit the input as alpha *i  + beta
    :param radius: determines the limit of the inputs around the base_vector
    :param y_idx_max: max index in the output layer, if None run the model with x for n_iterations and extract it
    :param other_idx: which index to compare max idx, if None the minimum when running the model with x for n iterations
    :param h5_file_path: path to keras model which we will check on
    :param n_iterations: number of iterations to run
    :param is_fail_test: we make sure that the output in y_idx_max >= other_idx, if this is True we negate it
    :return: True / False
    '''
    if y_idx_max is None or other_idx is None:
        out = get_output_vector_template(h5_file_path, x, n_iterations)
        if other_idx is None:
            other_idx = np.argmin(out)
        if y_idx_max is None:
            y_idx_max = np.argmax(out)
        print(y_idx_max, other_idx)

    assert_adversarial_query_template(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test)

    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    xlim = calc_min_max_template_by_radius(x, radius)
    rnn_model.set_input_bounds_template(x, radius)
    initial_values = rnn_model.get_rnn_min_max_value_one_iteration(xlim)

    # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])  # The zero'th element in the output layer
    adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])  # The y_idx_max of the output layer
    adv_eq.setScalar(0)

    rnn_output_idxs = rnn_model.rnn_out_idx
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]

    # Convert the initial values to the SGDAlphaAlgorithm format
    rnn_max_values = [val[1] for val in initial_values]
    rnn_min_values = [val[0] for val in initial_values]

    assert sum([rnn_max_values[i] >= rnn_min_values[i] for i in range(len(rnn_max_values))]) == len(rnn_max_values)
    algorithm = SGDAlphaAlgorithm((rnn_min_values, rnn_max_values), rnn_start_idxs, rnn_output_idxs)
    # rnn_model.network.dump()
    return prove_multidim_property(rnn_model.network, rnn_start_idxs, rnn_output_idxs, [negate_equation(adv_eq)],
                                   algorithm)


def adversarial_query(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str, n_iterations=10,
                      is_fail_test=False):
    '''
    Query marabou with adversarial query
    :param x: base_vector (input vector that we want to find a ball around it)
    :param radius: determines the limit of the inputs around the base_vector
    :param y_idx_max: max index in the output layer, if None run the model with x for n_iterations and extract it
    :param other_idx: which index to compare max idx, if None the minimum when running the model with x for n iterations
    :param h5_file_path: path to keras model which we will check on
    :param n_iterations: number of iterations to run
    :param is_fail_test: we make sure that the output in y_idx_max >= other_idx, if this is True we negate it
    :return: True / False
    '''
    if y_idx_max is None or other_idx is None:
        out = get_output_vector(h5_file_path, x, n_iterations)
        if other_idx is None:
            other_idx = np.argmin(out)
        if y_idx_max is None:
            y_idx_max = np.argmax(out)
        print(y_idx_max, other_idx)

    assert_adversarial_query(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test)
    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    xlim = calc_min_max_by_radius(x, radius)
    rnn_model.set_input_bounds(xlim)
    initial_values = rnn_model.get_rnn_min_max_value_one_iteration(xlim)

    # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])  # The zero'th element in the output layer
    adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])  # The y_idx_max of the output layer
    # adv_eq
    # .addAddend(-1, 3)  # The y_idx_max of the output layer
    adv_eq.setScalar(0)

    rnn_output_idxs = rnn_model.rnn_out_idx
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]

    # Convert the initial values to the SGDAlphaAlgorithm format
    rnn_max_values = [val[1] for val in initial_values]
    rnn_min_values = [val[0] for val in initial_values]

    assert sum([rnn_max_values[i] >= rnn_min_values[i] for i in range(len(rnn_max_values))]) == len(rnn_max_values)
    algorithm = SGDAlphaAlgorithm((rnn_min_values, rnn_max_values), rnn_start_idxs, rnn_output_idxs)
    # rnn_model.network.dump()
    return prove_multidim_property(rnn_model.network, rnn_start_idxs, rnn_output_idxs, [negate_equation(adv_eq)],
                                   algorithm)


def test_20classes_1rnn2_1fc2_fail():
    n_inputs = 40
    y_idx_max = 10
    other_idx = 19

    assert not adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_2_4.h5", is_fail_test=True)


def test_20classes_1rnn2_1fc2_pass():
    n_inputs = 40
    y_idx_max = 19
    other_idx = 10

    assert adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_2_4.h5", is_fail_test=False)


def test_20classes_1rnn2_1fc32_pass():
    n_inputs = 40
    y_idx_max = 13
    other_idx = 19

    results = []
    for i in range(20):
        if other_idx != y_idx_max:
            other_idx = i
            results.append(adversarial_query([1] * n_inputs, 0, y_idx_max, other_idx,
                                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_32_4.h5",
                                             is_fail_test=False))
            print(results)
    assert sum(results) == 19, 'managed to prove on {}%'.fromat((19 - sum(results)) / 19)


def test_20classes_1rnn2_1fc32_fail():
    n_inputs = 40
    y_idx_max = 0
    other_idx = 13

    assert not adversarial_query([1] * n_inputs, 0.1, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_32_4.h5",
                                 is_fail_test=True)


def test_20classes_1rnn2_1fc32_pass():
    n_inputs = 40
    y_idx_max = 13
    other_idx = 0

    assert adversarial_query([1] * n_inputs, 0.05, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_1_32_4.h5",
                             is_fail_test=False)


# def test_5classes_1rnn2_0fc_pass():
#     n_inputs = 40
#     y_idx_max = 4  # 2
#     other_idx = 0
#     # assert adversarial_query([1] * n_inputs, 0, y_idx_max, "/home/yuval/projects/Marabou/model_classes20_1rnn8_0_64_100.h5")
#     assert adversarial_query([10] * n_inputs, 0, y_idx_max, other_idx,
#                              "/home/yuval/projects/Marabou/model_classes5_1rnn2_0_64_4.h5")


def test_5classes_1rnn2_0fc_fail():
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 0
    other_idx = 4
    # assert adversarial_query([1] * n_inputs, 0, y_idx_max, "/home/yuval/projects/Marabou/model_classes20_1rnn8_0_64_100.h5")
    assert not adversarial_query([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes5_1rnn2_0_64_4.h5", is_fail_test=True)


def test_20classes_1rnn2_0fc_template_input_pass():
    n_iterations = 5
    n_inputs = 40

    template = [(i, 0) for i in range(n_inputs)]
    assert adversarial_query_template(template, 0.1, None, None,
                                      "/home/yuval/projects/Marabou/model_classes20_1rnn2_0_64_4.h5",
                                      n_iterations=n_iterations)


def test_20classes_1rnn2_0fc_template_input_fail():
    n_iterations = 50
    n_inputs = 40

    template = [(i, 0) for i in range(n_inputs)]
    assert not adversarial_query_template(template, 0.1, 8, 9,
                                          "/home/yuval/projects/Marabou/model_classes20_1rnn2_0_64_4.h5",
                                          n_iterations=n_iterations, is_fail_test=True)


def test_20classes_1rnn2_0fc_pass():
    n_iterations = 1000
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = None
    other_idx = None
    assert adversarial_query([10] * n_inputs, 0.1, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn2_0_64_4.h5",
                             n_iterations=n_iterations)
    return
    # results = []
    # for i in range(20):
    #     if i != y_idx_max:
    #         other_idx = i
    #         results.append(adversarial_query([10] * n_inputs, 0.1, y_idx_max, other_idx,
    #                                          "/home/yuval/projects/Marabou/model_classes20_1rnn2_0_64_4.h5"))
    # print(results)
    # assert sum(results) == 19, 'managed to prove on: {}%'.format((19 - sum(results)) / 19)


def test_20classes_1rnn2_0fc_fail():
    n_inputs = 40
    y_idx_max = 2
    other_idx = 9
    # 6.199209
    assert not adversarial_query([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn2_0_64_4.h5", is_fail_test=True)


def test_20classes_1rnn3_0fc_pass():
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 8
    other_idx = 5
    assert adversarial_query([1] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn3_0_2_4.h5")
    # return
    results = []
    for i in range(20):
        if i != y_idx_max:
            other_idx = i
            results.append(adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                                             "/home/yuval/projects/Marabou/model_classes20_1rnn3_0_2_4.h5"))
    print(results)
    assert sum(results) == 19, 'managed to prove on: {}%'.format((19 - sum(results)) / 19)


def test_20classes_1rnn3_0fc_fail():
    n_inputs = 40
    y_idx_max = 13
    other_idx = 8
    # 6.199209
    assert not adversarial_query([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn3_0_2_4.h5", is_fail_test=True)


def test_20classes_1rnn4_0fc_pass():
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 19
    other_idx = 1
    assert adversarial_query([79] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn4_0_64_4.h5")


def test_20classes_1rnn4_0fc_fail():
    n_inputs = 40
    y_idx_max = 1
    other_idx = 19
    assert not adversarial_query([39] * n_inputs, 0, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn4_0_64_4.h5", is_fail_test=True)


def test_20classes_1rnn8_0fc_pass():
    n_inputs = 40
    y_idx_max = 1
    other_idx = 0
    assert adversarial_query([5] * n_inputs, 0, y_idx_max, other_idx,
                             "/home/yuval/projects/Marabou/model_classes20_1rnn8_0_64_100.h5")


def test_20classes_1rnn8_0fc_fail():
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 0
    other_idx = 1
    assert not adversarial_query([5] * n_inputs, 0, y_idx_max, other_idx,
                                 "/home/yuval/projects/Marabou/model_classes20_1rnn8_0_64_100.h5", is_fail_test=True)


if __name__ == "__main__":
    # test_20classes_1rnn3_0fc_pass()
    # test_20classes_1rnn3_0fc_fail()
    test_20classes_1rnn2_0fc_template_input_pass()
    test_20classes_1rnn2_0fc_template_input_fail()
    # test_20classes_1rnn2_0fc_pass()
    # test_20classes_1rnn2_0fc_fail()
    # test_20classes_1rnn2_1fc32_pass()
    # test_20classes_1rnn2_1fc32_fail()

    # test_20classes_1rnn4_0fc_pass()

    # test_20classes_1rnn8_0fc_pass()
    # test_20classes_1rnn4_0fc_fail()
    # test_5classes_1rnn2_0fc_pass()
    # test_5classes_1rnn2_0fc_fail()
    # n_inputs = 40
    # y_idx_max = 1  # 2
    # assert adversarial_query([1] * n_inputs, 0, y_idx_max, "/home/yuval/projects/Marabou/model_classes20_1rnn8_0_64_100.h5")
    # assert adversarial_query([1] * n_inputs, 0, y_idx_max,
    #                          "/home/yuval/projects/Marabou/model_classes5_1rnn2_0_64_4.h5")
