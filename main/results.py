import os
import pickle
import random


class Results:
    ps = os.getpid()
    rnd = random.randint(0, 100)
    result_folder = "./results"
    folder = result_folder + "/" + str(ps) + "_" + str(rnd)

    def __init__(self, folder=None):
        self.folder = Results.folder if folder is None else folder
        if not os.path.isdir(Results.result_folder):
            os.mkdir(Results.result_folder)
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def save_data(self, data, filename):
        file = os.path.join(self.folder, filename)
        pickle.dump(data, open(file, "wb"))

    def load_data(self, filename):
        file = os.path.join(self.folder, filename)
        if not os.path.isfile(file):
            raise Exception(file + " does not exist.")
        return pickle.load(open(file, "rb"))

    def save_graph(self, plot, filename):
        file = os.path.join(self.folder, filename)
        plot.save(file)
