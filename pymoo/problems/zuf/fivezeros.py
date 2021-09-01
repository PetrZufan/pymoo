import numpy as np

from pymoo.model.problem import Problem


class FiveZeros(Problem):
    def __init__(self,
                 length=8,  # number of all bits
                 zeros=5,  # number of zeros
                 ):
        super().__init__(n_var=length, n_obj=1, xl=0, xu=1, type_var=np.bool)

        self.length = length
        self.zeros = zeros
        self.ones = length - zeros

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.abs(np.sum(X, axis=1) - self.ones)


class MultiObjectiveFiveZeros(FiveZeros):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.abs(np.sum(X, axis=1) - self.ones)
        f2 = np.array([self._toInt(x) for x in X])

        out["F"] = np.column_stack([f1, f2])

    def _toInt(self, x):
        return np.sum([a * 2**(self.length-i-1) for i, a in enumerate(x)])