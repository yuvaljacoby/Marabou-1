from typing import Tuple

from gurobipy import Model, LinExpr, GRB

from maraboupy import MarabouCore
from RNN.MarabouRnnModel import LARGE


class Bound:
    def __init__(self, gmodel: Model, upper: bool, initial_value: float, bound_idx: int, polyhedron_idx: int):
        self.upper = upper
        self.alpha = LARGE
        if not self.upper:
            self.alpha = -self.alpha
        self.initial_value = initial_value
        self.bound_idx = bound_idx
        self.polyhedron_idx = polyhedron_idx

        self.alpha_val = None
        self.beta_val = None

        # def attach_bound_to_model(self, gmodel: Model) -> None:
        self.gmodel = gmodel
        first_letter = 'u' if self.upper else 'l'
        self.name = "a{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx)
        self.alpha_var = \
            gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS,
                          name=self.name)
        self.beta_var = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS,
                                      name="b{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx))
        if self.upper:
            gmodel.addConstr(self.beta_var >= initial_value, name='{}_init_val'.format(self.name))
        else:
            gmodel.addConstr(self.beta_var <= initial_value, name='{}_init_val'.format(self.name))

    def __eq__(self, other):
        if not isinstance(other, Bound):
            return False
        if self._was_model_optimized() and other._was_model_optimized():
            return self.alpha_val == other.alpha_val and self.beta_val == other.beta_val
        else:
            return self.name == other.name

    def get_rhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first attach to model")
        return self.alpha_var * t + self.beta_var

    def get_lhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first attach to model")
        return self.alpha_var * (t + 1) + self.beta_var

    def get_objective(self, alpha_weight=1, beta_weight=1) -> LinExpr():
        obj = self.alpha_var * alpha_weight + self.beta_var * beta_weight
        # we want the lower bound to be as tight as possible so we should prefer large numbers on small numbers
        if not self.is_upper:
            obj = -obj
        return obj

    def is_upper(self) -> bool:
        return self.upper

    def _was_model_optimized(self):
        return self.alpha_val is not None and self.beta_val is not None

    def __str__(self):
        if self._was_model_optimized():
            return '{}: {}*i + {}'.format(self.name, round(self.alpha_val, 3), round(self.beta_val), 3)
        else:
            return self.name

    def __repr__(self):
        return self.__str__()

    def get_equation(self, loop_idx, rnn_out_idx):
        if not self._was_model_optimized():
            raise Exception("First optimize the attached model")

        inv_type = MarabouCore.Equation.LE if self.is_upper() else MarabouCore.Equation.GE
        if inv_type == MarabouCore.Equation.LE:
            ge_better = -1
        else:
            # TODO: I don't like this either
            ge_better = 1
            # ge_better = -1

        invariant_equation = MarabouCore.Equation(inv_type)
        invariant_equation.addAddend(1, rnn_out_idx)  # b_i
        invariant_equation.addAddend(self.alpha_val * ge_better, loop_idx)  # i
        invariant_equation.setScalar(self.beta_val)
        return invariant_equation

    def model_optimized(self) -> 'Bound':
        self.alpha_val = self.alpha_var.x
        self.beta_val = self.beta_var.x

        return self

    def get_bound(self) -> Tuple[int, int]:
        return self.alpha_val, self.beta_val
