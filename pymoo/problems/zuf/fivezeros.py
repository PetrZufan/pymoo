import numpy as np

from pymoo.model.problem import Problem
from pymoo.problems.zuf.quantum_problem import QuantumProblem


class FiveZeros(Problem):
    def __init__(
        self,
        n_var=8,  # number of all bits
        zeros=5,  # number of zeros
        n_obj=1,
    ):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=1, type_var=np.bool)

        self.n_var = n_var
        self.zeros = zeros
        self.ones = n_var - zeros

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.abs(np.sum(X, axis=1) - self.ones)


class MultiObjectiveFiveZeros(FiveZeros):
    def __init__(self, *args, **kwargs):
        super().__init__(n_obj=2, *args, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.abs(np.sum(X, axis=1) - self.ones)
        f2 = np.array([self._toInt(x) for x in X])

        out["F"] = np.column_stack([f1, f2])

    def _toInt(self, x):
        return np.sum([a * 2 ** (self.n_var - i - 1) for i, a in enumerate(x)])
