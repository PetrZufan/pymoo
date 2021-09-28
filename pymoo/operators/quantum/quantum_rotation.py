import numpy as np

from pymoo.model.quantum.rotation import Rotation

# TODO: adapt all to multi objective


class SOOriginalBinaryQuantumRotation(Rotation):
    """
        As Q-gate in bQIEAo in https://link.springer.com/article/10.1007/s10732-010-9136-0
    """
    def _do(self, problem, pop, opt, **kwargs):
        X = pop.get("X")
        O = pop.get("observed")
        F = pop.get("F")

        angle = 0.01*np.pi

        do_rotation = np.reshape(F > opt[0].F, F.shape[0])

        _X = X[do_rotation]
        _O = O[do_rotation]

        qbit_rotation = _O != opt[0].get("observed")

        _Xq = _X[qbit_rotation]
        _Oq = _O[qbit_rotation]

        sign = np.sign(_Xq[:, 0] * _Xq[:, 1])
        sign = np.where(sign == 0, 1, sign)
        sign = np.where(_Oq, -1*sign, sign)

        theta = sign * angle

        R = np.stack((
            np.stack((np.cos(theta), -1 * np.sin(theta)), axis=2),
            np.stack((np.sin(theta), np.cos(theta)), axis=2)
        ), axis=2)

        _Xq = np.einsum('lj,ljk->lk', _Xq, R)

        _X[qbit_rotation] = _Xq
        X[do_rotation] = _X

        return X


class SONovelBinaryQuantumRotation(Rotation):
    """
        As Q-gate in bQIEAn in https://link.springer.com/article/10.1007/s10732-010-9136-0
    """
    def _do(self, problem, pop, opt, **kwargs):
        X = pop.get("X")

        alg = kwargs["algorithm"]
        gen = alg.n_gen
        gen_max = alg.termination.n_max_gen
        angle = 0.5 * np.pi * np.exp(-5 * gen / gen_max)

        d = np.multiply(X[:, :, 0], X[:, :, 1])
        d_b = np.multiply(opt[0].X[:, 0], opt[0].X[:, 1])

        xi = np.abs(np.arctan(np.divide(X[:, :, 1], X[:, :, 0])))
        xi_b = np.abs(np.arctan(np.divide(opt[0].X[:, 1], opt[0].X[:, 0])))

        d_flags = (d_b >= 0) == (d > 0)
        xi_flags = xi_b >= xi
        flags = np.logical_xor(d_flags, xi_flags)

        sign = np.where(flags, -1, 1)

        theta = angle * sign
        R = np.stack((
            np.stack((np.cos(theta), -1 * np.sin(theta)), axis=2),
            np.stack((np.sin(theta), np.cos(theta)), axis=2)
        ), axis=2)
        X = np.einsum('ilj,iljk->ilk', X, R)
        return X


class SORealQuantumRotation(Rotation):
    """
        As Q-gate in rQIEA in https://link.springer.com/article/10.1007/s10732-010-9136-0
    """
    def _do(self, problem, pop, opt, **kwargs):
        X = pop.get("X")

        alg = kwargs["algorithm"]
        gen = alg.n_gen

        angle = np.pi/(100 + np.mod(gen, 100))
        xi = np.arctan(np.divide(X[:, :, 1], X[:, :, 0]))
        xi_b = np.arctan(np.divide(opt[0].X[:, 1], opt[0].X[:, 0]))

        xi_flags = (xi > 0) == (xi_b > 0)
        xi_zero_flags = np.logical_and(
            np.logical_or(np.logical_or(xi == 0.0, xi == np.pi/2), xi == np.pi/-2),
            np.logical_or(np.logical_or(xi_b == 0.0, xi_b == np.pi/2), xi_b == np.pi/-2)
        )
        xi_b_flags = np.logical_not(np.logical_or(xi_flags, xi_zero_flags))

        sign = np.where(xi_b >= xi, 1, -1)
        sign[xi_zero_flags] = np.where(np.random.randint(2, size=sign[xi_zero_flags].shape), 1, -1)
        s = np.sign((X[:, :, 0] * opt[0].X[:, 0]))
        sign[np.logical_and(xi_b_flags, xi_b > 0)] = s[np.logical_and(xi_b_flags, xi_b > 0)]
        sign[np.logical_and(xi_b_flags, xi_b <= 0)] = s[np.logical_and(xi_b_flags, xi_b <= 0)]

        theta = angle * sign
        R = np.stack((
            np.stack((np.cos(theta), -1 * np.sin(theta)), axis=2),
            np.stack((np.sin(theta), np.cos(theta)), axis=2)
        ), axis=2)
        X = np.einsum('ilj,iljk->ilk', X, R)
        return X


class MORealQuantumRotation(Rotation):
    """
        As Q-gate in rQIEA in https://link.springer.com/article/10.1007/s10732-010-9136-0
    """
    def _do(self, problem, pop, opt, **kwargs):
        X = pop.get("X")

        alg = kwargs["algorithm"]
        gen = alg.n_gen

        angle = np.pi/(100 + np.mod(gen, 100))
        xi = np.arctan(np.divide(X[:, :, 1], X[:, :, 0]))
        xi_b = np.arctan(np.divide(opt[0].X[:, 1], opt[0].X[:, 0]))

        xi_flags = (xi > 0) == (xi_b > 0)
        xi_zero_flags = np.logical_and(
            np.logical_or(np.logical_or(xi == 0.0, xi == np.pi/2), xi == np.pi/-2),
            np.logical_or(np.logical_or(xi_b == 0.0, xi_b == np.pi/2), xi_b == np.pi/-2)
        )
        xi_b_flags = np.logical_not(np.logical_or(xi_flags, xi_zero_flags))

        sign = np.where(xi_b >= xi, 1, -1)
        sign[xi_zero_flags] = np.where(np.random.randint(2, size=sign[xi_zero_flags].shape), 1, -1)
        s = np.sign((X[:, :, 0] * opt[0].X[:, 0]))
        sign[np.logical_and(xi_b_flags, xi_b > 0)] = s[np.logical_and(xi_b_flags, xi_b > 0)]
        sign[np.logical_and(xi_b_flags, xi_b <= 0)] = s[np.logical_and(xi_b_flags, xi_b <= 0)]

        theta = angle * sign
        R = np.stack((
            np.stack((np.cos(theta), -1 * np.sin(theta)), axis=2),
            np.stack((np.sin(theta), np.cos(theta)), axis=2)
        ), axis=2)
        X = np.einsum('ilj,iljk->ilk', X, R)
        return X
