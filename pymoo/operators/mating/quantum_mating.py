
import math
from pymoo.model.mating import Mating


class QuantumMating(Mating):
    """
    Quantum-inspired evolutionary algorithms as described here https://link.springer.com/article/10.1007/s10732-010-9136-0
    offers more genetic operators to be applied.

    Just set any of them to None to ignore it.
    """

    def __init__(self, selection, crossover, mutation, rotation, migration, catastrophe, **kwargs):
        super().__init__(
            selection,
            crossover,
            mutation,
            **kwargs
        )
        self.rotation = rotation
        self.migration = migration
        self.catastrophe = catastrophe

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        _off = pop.copy(deep=True)
        if self.rotation is not None:
            _off = self.rotation.do(problem, _off, **kwargs)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:
            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)

            # select the parents for the mating - just an index array
            parents = self.selection.do(_off, n_select, self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        if self.crossover is not None:
            _off = self.crossover.do(problem, _off, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        if self.mutation is not None:
            _off = self.mutation.do(problem, _off, **kwargs)

        if self.migration is not None:
            _off = self.migration.do(problem, _off, **kwargs)

        if self.catastrophe is not None:
            _off = self.catastrophe.do(problem, _off, **kwargs)

        return _off
