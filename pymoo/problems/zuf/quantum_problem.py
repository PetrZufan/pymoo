
import numpy as np
from pymoo.model.problem import Problem
from pymoo.util.normalization import denormalize


class QuantumProblem(Problem):
    def __init__(
            self,
            classic_problem_clazz,
            encoding_type="binary",
            bcr_size=8,
            *args,
            **kwargs
    ):
        """
        Transforms any classical problem into quantum one.
        It means that each gene variable is represented by Q-bit(s).
        See https://link.springer.com/article/10.1007/s10732-010-9136-0 for more information.

        Parameters
        ----------

        classic_problem_clazz: Problem
            classical version of given quantum problem after observation.
        encoding_type: String
            "binary": One Q-bit represents one binary number
            "real": One Q-bit represents one real number
            "binary_coded_real" or "bcr": Sequence of Q-bits represent one real number.
                                          Parameter bcr_size defines the length of this sequence.
        bcr_size: Int
            The length of Q-bit sequence representing one real number.
            This has meaning only when encoding_type="bcr"

        All other parameters are passed to classical problem initialization.
        """
        self.classic_problem_clazz = classic_problem_clazz
        self.classic_problem = None
        self._init_classic_problem(*args, **kwargs)

        self.encoding_type = encoding_type
        if encoding_type == "bcr" or encoding_type == "binary_coded_real":
            self.bcr_size = bcr_size
        else:
            self.bcr_size = 1

        super().__init__(
            type_var=np.ndarray,
            xl=-1,
            xu=1,
            n_var=self.classic_problem.n_var * self.bcr_size,
            n_obj=self.classic_problem.n_obj,
            n_constr=self.classic_problem.n_constr,
            evaluation_of=self.classic_problem.evaluation_of,
            replace_nan_values_of=self.classic_problem.replace_nan_values_of,
            parallelization=self.classic_problem.parallelization,
            elementwise_evaluation=self.classic_problem.elementwise_evaluation,
            exclude_from_serialization=self.classic_problem.exclude_from_serialization,
            callback=self.classic_problem.callback
        )

    def _init_classic_problem(self, *args, **kwargs):
        """
        Override in need to pass other params.
        """
        self.classic_problem = self.classic_problem_clazz(*args, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        O, E = self._observe(x)
        out["observed"] = O
        out["eval"] = E
        self.classic_problem._evaluate(E, out, *args, **kwargs)

    # TODO: make separated Observe operators from these
    def _observe(self, X, *args, **kwargs):
        if self.encoding_type == "binary":
            B = self._observe_binary(X)
            return B, B
        elif self.encoding_type == "bcr" or self.encoding_type == "binary_coded_real":
            B = self._observe_binary(X)
            R = self._bcr_decode(B)
            return B, R
        elif self.encoding_type == "real":
            R = self._observe_real(X)
            return R, R
        else:
            raise Exception(f"Unrecognized encoding type {self.encoding_type}.")

    def _observe_binary(self, X):
        return np.random.random((X.shape[0], X.shape[1])) > self._observe_val(0, X)

    def _observe_real(self, X):
        mask = np.random.random((X.shape[0], X.shape[1])) < 0.5
        O = np.where(mask, self._observe_val(0, X), self._observe_val(1, X))
        return denormalize(O, self.classic_problem.xl, self.classic_problem.xu)

    def _observe_val(self, val, X):
        return apply_to_qbit(lambda x: x[val] ** 2, X)

    def _bcr_decode(self, X):
        f_split = lambda x: np.split(x, self.classic_problem.n_var)
        X_splited = np.apply_along_axis(f_split, 1, X)
        f_2real = lambda x: self._bin2real(x)
        X_real = np.apply_along_axis(f_2real, 2, X_splited)
        f_norm = lambda x: denormalize(x, self.classic_problem.xl, self.classic_problem.xu)
        X_norm = np.apply_along_axis(f_norm, 1, X_real)
        return X_norm

    def _bin2real(self, X):
        Y = X * 2 ** np.indices(X.shape)
        O = np.sum(Y) / (2 ** self.bcr_size -1)
        return O


def apply_to_qbit(func, X):
    return np.apply_along_axis(lambda x: func(x), 2, X)
