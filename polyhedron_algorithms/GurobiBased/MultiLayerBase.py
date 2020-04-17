from RNN import MarabouRnnModel
from polyhedron_algorithms.GurobiBased.SingleLayerBase import GurobiSingleLayer

POLYHEDRON_MAX_DIM = 1


class GurobiMultiLayer:
    # Alpha Search Algorithm for multilayer recurrent, assume the recurrent layers are one following the other
    # we need this assumptions in proved_invariant method, if we don't have it we need to extract bounds in another way
    # not sure it is even possible to create multi recurrent layer NOT in a row
    def __init__(self, rnnModel: MarabouRnnModel, xlim, polyhedron_max_dim=POLYHEDRON_MAX_DIM, use_relu=True, use_counter_example=False,
                 add_alpha_constraint=False, **kwargs):
        '''

        :param rnnModel: multi layer rnn model (MarabouRnnModel class)
        :param xlim: limit on the input layer
        :param update_strategy_ptr: not used
        :param random_threshold: not used
        :param use_relu: if to encode relus in gurobi
        :param use_counter_example: if property is not implied wheter to use counter_example to  create better _u
        :param add_alpha_constraint: if detecting a loop (proved invariant and then proving the same invariant) wheter to add a demand that every alpha will change in epsilon
        '''
        self.return_vars = True
        self.support_multi_layer = True
        self.num_layers = len(rnnModel.rnn_out_idx)
        self.alphas_algorithm_per_layer = []
        self.alpha_history = None
        self.rnnModel = rnnModel
        self.polyhedron_max_dim = polyhedron_max_dim
        self.use_relu = use_relu
        self.use_counter_example = use_counter_example
        self.add_alpha_constraint = add_alpha_constraint
        self.input_lim = xlim
        for i in range(self.num_layers):
            self.alphas_algorithm_per_layer.append(self._initialize_single_layer(i))

    def _initialize_single_layer(self, layer_idx: int) -> GurobiSingleLayer:
        if layer_idx == 0:
            prev_layer_lim = self.input_lim
        else:
            prev_layer_lim = None
        return GurobiSingleLayer(self.rnnModel, prev_layer_lim, polyhedron_max_dim=self.polyhedron_max_dim,
                                 use_relu=self.use_relu, use_counter_example=self.use_counter_example,
                                 add_alpha_constraint=self.add_alpha_constraint, layer_idx=layer_idx)

    def proved_invariant(self, layer_idx=0, equations=None):
        # Proved invariant on layer_idx --> we can update bounds for layer_idx +1
        l_bound, u_bound = self.alphas_algorithm_per_layer[layer_idx].get_bounds()
        alphas_l, alphas_u, betas_l, betas_u = [], [], [], []

        for l_bound_node in l_bound:
            min_alpha_bound = l_bound_node[0][1] #* self.rnnModel.n_iterations
            min_beta_bound = l_bound_node[0][1]
            for l in l_bound_node[1:]:
                if l[0] + l[1] > min_alpha_bound + min_beta_bound:
                    min_alpha_bound = l[0]
                    min_beta_bound = l[1]
            alphas_l.append(min_alpha_bound)
            betas_l.append(min_beta_bound)
        for u_bound_node in u_bound:
            max_alpha_bound = u_bound_node[0][0] #* self.rnnModel.n_iterations
            max_beta_bound = u_bound_node[0][1]
            for u in u_bound_node[1:]:
                if u[0] + u[1] < max_alpha_bound + max_beta_bound:
                    max_alpha_bound = u[0]
                    max_beta_bound = u[1]
            alphas_u.append(max_alpha_bound)
            betas_u.append(max_beta_bound)

        if len(self.alphas_algorithm_per_layer) > layer_idx + 1:
            print(alphas_l, alphas_u, (betas_l, betas_u))
            self.alphas_algorithm_per_layer[layer_idx + 1].update_xlim(alphas_l, alphas_u, (betas_l, betas_u))

    def do_step(self, strengthen=True, invariants_results=[], sat_vars=None, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].do_step(strengthen, invariants_results, sat_vars)

    def get_equations(self, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].get_equations()

    def get_alphas(self, layer_idx=0):
        return self.alphas_algorithm_per_layer[layer_idx].get_alphas()
