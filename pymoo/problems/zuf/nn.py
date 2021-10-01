import numpy as np
import tensorflow as tf
from pymoo.model.problem import Problem
from pymoo.model.repair import Repair
from pymoo.model.sampling import Sampling
from pymoo.neural_network.models.mnist import ModelMnistClassifier
from pymoo.operators.sampling.random_sampling import FloatRandomSampling


class NeuralNetwork(Problem):
    def __init__(
        self,
        n_var=None, # None means auto deduce
        n_obj=2,
        zero_approximation=0.0001,
        model=ModelMnistClassifier,
        dataset=tf.keras.datasets.mnist,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    ):
        self.zero_approximation = zero_approximation

        # load data
        self.dataset = dataset
        (self.train_ins, self.train_outs), (self.test_ins, self.test_outs) = self.load_dataset(self.dataset)

        # setup model
        self.model_clazz = model
        self.model = model()
        self.model.trainable = False

        # setup loss function
        self.loss = loss

        # deduce number of variables from model
        if n_var is not None:
            self.n_var = n_var
        else:
            self.n_var = self.size_from_weights(self.model.get_weights())

        # call super
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=-10.0, xu=10.0, type_var=np.double)

    def load_dataset(self, dataset):
        (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        return (train_images, train_labels), (test_images, test_labels)

    def get_batch(self, data_in, data_out, batch_size):
        select = np.random.randint(0, data_in.shape[0]-1, batch_size)
        return data_in[select], data_out[select]

    def _evaluate(self, X, out, *args, **kwargs):
        # get loss fitness
        f0 = np.apply_along_axis(lambda x: self.predict(x), 1, X)

        # get sparsity fitness
        f1 = self.n_var - np.sum(np.abs(X) < self.zero_approximation, axis=1)

        # set fitnesses
        out["F"] = np.column_stack([f0, f1])

    def predict(self, X):
        weights = self.x_to_weights(X, self.model.get_weights())
        self.model.set_weights(weights=weights)
        ins, referrals = self.get_batch(self.train_ins, self.train_outs, 10)
        predictions = self.model.predict(ins)
        return self.loss(referrals, predictions).numpy()

    def x_to_weights(self, list1, list2, last=0):
        res = []
        for ele in list2:
            if isinstance(ele[0], np.ndarray):
                res.append(np.array(self.x_to_weights(list1, ele, last)))
            else:
                res.append(list1[last: last + len(ele)])
                last += len(ele)
        return res

    def weights_to_x(self, weights):
        return np.concatenate([w.flatten() for w in weights])

    def size_from_weights(self, weights):
        return np.sum(np.array([it.size for it in weights]))


class SparsityRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        if not isinstance(problem, NeuralNetwork):
            return pop

        X = pop.get("X")
        X = np.where(X < problem.zero_approximation, 0.0, X)
        pop.set("X", X)
        return pop


class NeuralNetworkSampling(Sampling):

    def __init__(self, var_type=float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        if not isinstance(problem, NeuralNetwork):
            return FloatRandomSampling()._do(problem, n_samples)

        return np.array([self.get_x_from_model(problem) for x in np.zeros(n_samples)])

    def get_x_from_model(self, problem):
        w = problem.model_clazz().get_weights()
        return problem.weights_to_x(w)
