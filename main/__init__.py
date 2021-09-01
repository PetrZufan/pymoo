
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_crossover, get_mutation, get_sampling


def MOFZ():
    problem = get_problem("zuf-mofivezeros")

    algorithm = NSGA2(pop_size=10,
                      sampling=get_sampling("bin_random"),
                      crossover=get_crossover("bin_hux"),
                      mutation=get_mutation("bin_bitflip")
                      )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=False)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()


def QMOFZ():
    problem = get_problem("zuf-mofivezeros")

    algorithm = QNSGA2(pop_size=10,
                      sampling=get_sampling("bin_random"),
                      crossover=get_crossover("bin_hux"),
                      mutation=get_mutation("bin_bitflip")
                      )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=False)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()


if __name__ == "__main__":
    MOFZ()
