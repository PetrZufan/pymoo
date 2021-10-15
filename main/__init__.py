import argparse
import os
import sys
import random

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
from pymoo.algorithms.so_genetic_algorithm import comp_by_cv_and_fitness, GA
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


def final_images(result, is_train=False, is_quantum=False):
    if is_quantum:
        problem = result.problem.classic_problem
        field = "observed"
    else:
        problem = result.problem
        field = "X"

    if is_train:
        name = "train"
        shape = problem.dataset.shape_tr
    else:
        name = "test"
        shape = problem.dataset.shape_ts

    images_data = np.apply_along_axis(lambda x: problem.get_outs(x, is_train), 1, result.opt.get(field))
    for i, d in enumerate(images_data):
        filename = str(os.getpid()) + "_final_" + name + "_" + str(i) + ".bmp"
        file = os.path.join(Results().get_results_file(), filename)
        problem.dataset.save_image(file, d, is_normalized=True, reshape_to=shape)


def run(algorithm, population, generations, grid, batch):
    seed = random.SystemRandom().randint(0, 100000)
    if (algorithm == "bp"):
        return # TODO: back propagation

    is_quantum = (algorithm == "qnsga2") or (algorithm == "qiga")
    is_multi_objectiv = (algorithm == "qnsga2") or (algorithm == "nsga2")

    # problem
    problem = None
    n_obj = 2 if is_multi_objectiv else 1
    if is_quantum:
        problem = QuantumProblem(
            classic_problem_clazz=NeuralNetwork,
            encoding_type="real",
            model=ModelDibcoClassifier(),
            dataset=DIBCO(),
            loss=BinaryCrossentropy(),
            batch_size=batch,
            grid_size=grid,
            n_obj=n_obj
        )
    if not is_quantum:
        problem = NeuralNetwork(
            model=ModelDibcoClassifier(),
            dataset=DIBCO(),
            loss=BinaryCrossentropy(),
            batch_size=batch,
            grid_size=grid,
            n_obj=n_obj
        )

    # algorithm
    alg = None
    if algorithm == "qnsga2":
        alg = QNSGA2(
            pop_size=population,
            mutation=get_mutation("quantum_bitflip"),
            crossover=get_crossover("real_two_point"),
            rotation=MORealQuantumRotation(),
            eliminate_duplicates=False,
            # callback=SaveProgressCallback()
        )
    if algorithm == "nsga2":
        alg = NSGA2(
            pop_size=population,
            sampling=NeuralNetworkSampling(),
            repair=SparsityRepair(),
            eliminate_duplicates=False,
            # callback=SaveProgressCallback()
        )
    if algorithm == "qiga":
        alg = QIGA(
            pop_size=population,
            sampling=get_sampling("quantum_random"),
            crossover=get_crossover("real_one_point"),
            mutation=get_mutation("quantum_bitflip"),
            rotation=SORealQuantumRotation(),
            # callback=SaveProgressCallback()
        )
    if algorithm == "ga":
        alg = GA(
            pop_size=population,
            # callback=SaveProgressCallback()
        )

    res = minimize(
        problem,
        alg,
        ('n_gen', generations),
        seed=seed,
        verbose=False,
        save_history=True
    )

    storage = Results()

    # exceed disk space limit :(
    # res.history.append({"opt": res.opt, "pop": res.pop})
    # storage.save_data(res.history, "res.p")

    str_fit = "\n"
    for i, opt in enumerate(res.opt):
        str_fit = str_fit + "  i" + str(i) + ":\n"
        for j, f in enumerate(opt.get("F")):
            str_fit = str_fit + "    f" + str(j) + ": " + str(f) + "\n"

    storage.save_text(
        "seed: " + str(seed) + "\n" +
        "algorithm: " + algorithm + "\n" +
        "population: " + str(population) + "\n" +
        "generations: " + str(generations) + "\n" +
        "grid: " + str(grid) + "\n" +
        "batch: " + str(batch) + "\n" +
        "n_var: " + str(res.problem.n_var) + "\n" +
        "fitness: " + str(str_fit) + "\n",
        "res.txt"
    )

    # TODO: so vs mo graphs
    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(res.pop.get("F"), color="blue")
    # plot.add(res.F, color="red")
    # storage.save_graph(plot, "res.png")

    final_images(res, is_train=True, is_quantum=is_quantum)
    final_images(res, is_train=False, is_quantum=is_quantum)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--algorithm', type=str, choices=["qnsga2", "nsga2", "qiga", "ga", "bp"], default="qnsga2"
    )
    # parser.add_argument('-d', '--dataset')
    parser.add_argument('-p', '--population', type=int, default=50)
    parser.add_argument('-g', '--generations', type=int, default=100)
    parser.add_argument('-r', '--grid', type=int, default=11)
    parser.add_argument('-b', '--batch', type=int, default=64)
    args = parser.parse_args()

    algorithm = args.algorithm
    population = args.population
    generations = args.generations
    grid = args.grid
    batch = args.batch

    run(algorithm, population, generations, grid, batch)


    #QNN_NSGA2_dibco(5, 5)
    #plot_result("./16395_21", "res.p")
    exit(0)
