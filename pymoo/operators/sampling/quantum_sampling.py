
import numpy as np
from pymoo.model.sampling import Sampling
from pymoo.util.normalization import denormalize


class QuantumRandomSampling(Sampling):
    """
        Samples all Q-bits randomly.
        Sets a to random real number between -1 and 1.
        Sets b to +/-sqrt(1-a**2).

        See rQIEA in https://link.springer.com/article/10.1007/s10732-010-9136-0 for more information.
    """

    def __init__(self, var_type=np.ndarray) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        a = denormalize(val, problem.xl, problem.xu)
        sign = (np.random.randint(2, size=a.shape) * 2) - 1
        b = sign * np.sqrt(1 - a**2)
        return np.stack((a, b), axis=2)


class QuantumSuperpositionSampling(Sampling):
    """
        Samples all Q-bits to value (1/sqrt(2), 1/sqrt(2)).

        See bQIEAo in https://link.springer.com/article/10.1007/s10732-010-9136-0 for more information.
    """

    def __init__(self, var_type=float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return np.full((n_samples, problem.n_var, 2), [1/np.sqrt(2), 1/np.sqrt(2)])
