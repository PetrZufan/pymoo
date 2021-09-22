
import numpy as np
from pymoo.model.mutation import Mutation


class QuantumBitflipMutation(Mutation):
    """
    Represents quantum NOT gate applied to each Q-bit with given probability.
    Literally swap a and b values in Q-bit.
    """

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        M = np.random.random(X.shape[:2])
        X[M < self.prob] = np.flip(X[M < self.prob], 1)
        return X
