from copy import copy

import numpy as np
import pytest

from pymoo.algorithms.soo.convex.nonderiv.pattern_search import PatternSearch
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere, Himmelblau

PROBLEMS = [
    Himmelblau(),
    Sphere(n_var=10)
]


@pytest.mark.parametrize('problem', PROBLEMS)
@pytest.mark.parametrize('seed', range(0, 5))
@pytest.mark.parametrize('bounds', [True])
def test_against_original_implementation(problem, seed, bounds):
    problem = copy(problem)

    np.random.seed(seed)
    x0 = FloatRandomSampling().do(problem, 1)[0].X

    if not bounds:
        problem.xl = None
        problem.xu = None

    algorithm = PatternSearch(x0=x0)

    ret = minimize(problem, algorithm, verbose=True)

    fmin = problem.pareto_front().flatten()[0]
    np.testing.assert_almost_equal(fmin, ret.F[0], decimal=3)

    assert True
