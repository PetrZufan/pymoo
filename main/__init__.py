from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.qiga import QIGA
from pymoo.algorithms.qnsga2 import QNSGA2
from pymoo.algorithms.so_genetic_algorithm import comp_by_cv_and_fitness
from pymoo.factory import get_problem
from pymoo.operators.quantum.quantum_rotation import SOOriginalBinaryQuantumRotation, SORealQuantumRotation, \
    SONovelBinaryQuantumRotation, MORealQuantumRotation
from pymoo.optimize import minimize
from pymoo.problems.single.simple import SimpleMultiModal01
from pymoo.problems.zuf import QuantumProblem, FiveZeros, MultiObjectiveFiveZeros
from pymoo.problems.zuf.nn import NeuralNetwork, SparsityRepair, NeuralNetworkSampling
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
        rotation=SOOriginalBinaryQuantumRotation(),
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
        rotation=SORealQuantumRotation(),
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
        rotation=SONovelBinaryQuantumRotation(),
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


def QMOFZ_QNSGA2():
    problem = QuantumProblem(classic_problem_clazz=MultiObjectiveFiveZeros, encoding_type="binary", n_var=8, zeros=1)

    algorithm = QNSGA2(
        pop_size=100,
        mutation=get_mutation("quantum_bitflip"),
        crossover=get_crossover("real_two_point"),
        rotation=MORealQuantumRotation(),
        eliminate_duplicates=False,
        verbose=True,
        debug=True,
        )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        seed=1,
        verbose=True
    )

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
    return


def NN_NSGA2():
    problem = NeuralNetwork()
    # problem = QuantumProblem(classic_problem_clazz=MultiObjectiveFiveZeros, encoding_type="binary", n_var=8, zeros=1)

    algorithm = NSGA2(
        pop_size=10,
        sampling=NeuralNetworkSampling(),
        repair=SparsityRepair(),
        eliminate_duplicates=False,
        save_hisotry=True,
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        seed=1,
        verbose=True
    )

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
    return

def QNN_NSGA2():
    problem = QuantumProblem(classic_problem_clazz=NeuralNetwork, encoding_type="real")

    algorithm = QNSGA2(
        pop_size=10,
        mutation=get_mutation("quantum_bitflip"),
        crossover=get_crossover("real_two_point"),
        rotation=MORealQuantumRotation(),
        eliminate_duplicates=False,
        verbose=True,
        save_hisotry=True,
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        seed=1,
        verbose=True
    )

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
    return


if __name__ == "__main__":
    QNN_NSGA2()
