

class Rotation:
    def __init__(self) -> None:
        super().__init__()
        self.algorithm = None
        self.problem = None

    def do(self, problem, pop, algorithm, **kwargs):
        """

        Rotate quantum variables.

        Parameters
        ----------
        problem : class
            The problem instance - specific information such as variable bounds might be needed.
        pop : Population
            A population object

        Returns
        -------
        Y : Population
            The rotated population.

        """

        Y = self._do(problem, pop, algorithm.opt, algorithm=algorithm, **kwargs)
        return pop.set("X", Y)

    def _do(self, problem, pop, opt, **kwargs):
        pass
