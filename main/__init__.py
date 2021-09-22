from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.qiga import QIGA
from pymoo.algorithms.so_genetic_algorithm import comp_by_cv_and_fitness
from pymoo.factory import get_problem
from pymoo.operators.quantum.quantum_rotation import OriginalBinaryQuantumRotation, RealQuantumRotation, \
    NovelBinaryQuantumRotation
from pymoo.optimize import minimize
from pymoo.problems.single.simple import SimpleMultiModal01
from pymoo.problems.zuf import QuantumProblem, FiveZeros
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_selection


def FZ_GA():
    problem = get_problem("zuf-fivezeros")

    algorithm = GeneticAlgorithm(
        pop_size=10,
        sampling=get_sampling("bin_random"),
        selection=get_selection("tournament", func_comp=comp_by_cv_and_fitness),
        crossover=get_crossover("bin_hux"),
        mutation=get_mutation("bin_bitflip"),
        )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        seed=1,
        verbose=False
    )
    return


def MOFZ_NSGA2():
    problem = get_problem("zuf-mofivezeros")

    algorithm = NSGA2(
        pop_size=10,
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_hux"),
        mutation=get_mutation("bin_bitflip")
        )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        seed=1,
        verbose=False
    )

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()


def FZ_QIEA():
    # problem = get_problem("zuf-quantum_fivezeros", encoding_type="binary", length=8, zeros=1)
    problem = QuantumProblem(classic_problem_clazz=FiveZeros, encoding_type="binary", n_var=8, zeros=1)

    algorithm = QIGA(
        pop_size=2,
        sampling=get_sampling("quantum_superposition"),
        crossover=get_crossover("real_one_point"),
        mutation=get_mutation("quantum_bitflip"),
        rotation=OriginalBinaryQuantumRotation(),
        verbose=True
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=1,
        verbose=False
    )
    print(f"result = {res.opt.get('observed')}")

    return


def simple_QIEA():
    problem = QuantumProblem(classic_problem_clazz=SimpleMultiModal01, encoding_type="real")

    algorithm = QIGA(
        pop_size=4,
        sampling=get_sampling("quantum_random"),
        crossover=get_crossover("real_one_point"),
        mutation=get_mutation("quantum_bitflip"),
        rotation=RealQuantumRotation(),
        verbose=True
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 10),
        seed=1,
        verbose=True,
    )
    print(f"result = {res.opt.get('observed')}")

    return


def simple_QIEA_bcr():
    problem = QuantumProblem(classic_problem_clazz=SimpleMultiModal01, encoding_type="bcr")

    algorithm = QIGA(
        pop_size=3,
        sampling=get_sampling("quantum_superposition"),
        crossover=get_crossover("real_one_point"),
        mutation=get_mutation("quantum_bitflip"),
        rotation=NovelBinaryQuantumRotation(),
        verbose=True
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 10),
        seed=1,
        verbose=True,
    )
    print(f"result = {res.opt.get('observed')}")

    return


if __name__ == "__main__":
    # MOFZ_NSGA2()
    # FZ_GA()
    # FZ_QIEA()
     simple_QIEA_bcr()
