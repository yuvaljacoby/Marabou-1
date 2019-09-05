from maraboupy import MarabouCore

H5_FILE_PATH = 'pendulum_example/weights_pendulum_rnn.h5'
PI = 360

class PendulumModel():
    def __init__(self, h5_file_path):
        from maraboupy.pendulum_example.models import build_model
        self.model = build_model()
        self.model.load_weights(h5_file_path)

    def get_rnn_weights(self):
        '''
        rnn_input_weights.shape = [3, 16]
        rnn_hidden_weights.shape = [16,16]
        rnn_bias_weights.shape = [16,]
        :return: rnn_input_weights, rnn_hidden_weights, rnn_bias_weights
        '''
        rnn_input_weights, rnn_hidden_weights, rnn_bias_weights = self.model.layers[2].get_weights()
        return rnn_input_weights, rnn_hidden_weights, rnn_bias_weights

    def get_output_weights(self):
        '''
        output_weights.shape = [16,2]
        output_bias_weights.shape = [2,]
        :return: output_weights, output_bias_weights
        '''
        output_weights, output_bias_weights = self.model.layers[3].get_weights()
        return output_weights, output_bias_weights


def build_query( xlim : list, ylim : list, pendulum_model : PendulumModel):
    '''

    :param xlim: list of tuples, each cell is for input variable, and the tuple is (min_val, max_val)
    :param ylim: list of tuples, each cell is for an output variable, and the tuple is (min_val, max_val)
    :param pendulum_model: initialized model with get_rnn_weights and get_output_weights
    :return:
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x

    # angle
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # angular velocity
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    rnn_input_weights, rnn_hidden_weights, rnn_bias_weights = pendulum_model.get_rnn_weights()



if __name__ == "__main__":
    xlim = []
    # we know that 0 <= theta <= PI / 64
    xlim.append([-1, 1]) #cos theta
    xlim.append([-1, 1]) #sin theta
    xlim.append([0, 0.3]) # theta dot <--> angular velocity

    ylim = [[PI / 10]]
    pendulum_model = PendulumModel(H5_FILE_PATH)
