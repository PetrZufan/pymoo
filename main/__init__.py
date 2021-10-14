import os
import sys

from tensorflow.python.keras.losses import BinaryCrossentropy

sys.path.insert(0, "/home/zufan/git/pymoo/")

import numpy as np

from main.results import Results
from pymoo.datasets.dibco import DIBCO
from pymoo.model.callback import Callback
from pymoo.neural_network.models.binary import ModelDibcoClassifier
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


def QNN_NSGA2_dibco(pop_size=10, generations=50):
    problem = QuantumProblem(
        classic_problem_clazz=NeuralNetwork,
        encoding_type="real",
        model=ModelDibcoClassifier(),
        dataset=DIBCO(),
        loss=BinaryCrossentropy()
    )

    algorithm = QNSGA2(
        pop_size=pop_size,
        mutation=get_mutation("quantum_bitflip"),
        crossover=get_crossover("real_two_point"),
        rotation=MORealQuantumRotation(),
        eliminate_duplicates=False,
        callback=SaveProgressCallback()
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', generations),
        seed=None,
        verbose=False,
        save_history=True
    )

    storage = Results()
    res.history.append({"opt": res.opt, "pop": res.pop})
    storage.save_data(res.history, "res.p")

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.pop.get("F"), color="blue")
    plot.add(res.F, color="red")
    storage.save_graph(plot, "res.png")

    final_images(res)

    return res


class SaveProgressCallback(Callback):
    def notify(self, algorithm, **kwargs):
        storage = Results()
        algorithm.history.append({"opt": algorithm.opt, "pop": algorithm.pop})
        storage.save_data(algorithm.history, "progress.p")


def plot_result(folder, file):
    storage = Results(folder)
    res = storage.load_data(file)
    plot = Scatter()
    plot.add(res[len(res)-1].get("pop").get("F"), color="blue")
    plot.add(res[len(res)-1].get("opt").get("F"), color="red")
    plot.show()


def final_images(result):
    cl_problem = result.problem.classic_problem
    images_data = np.apply_along_axis(lambda x: cl_problem.get_test_outs(x), 1, result.opt.get("observed"))
    for i, d in enumerate(images_data):
        filename = str(os.getpid()) + "_final_" + str(i) + ".bmp"
        file = os.path.join(Results().get_results_file(), filename)
        cl_problem.dataset.save_image(file, d, is_normalized=True, reshape_to=cl_problem.dataset.shape_ts)


if __name__ == "__main__":
    QNN_NSGA2_dibco(50, 100)
    #plot_result("./16395_21", "res.p")
    exit(0)
