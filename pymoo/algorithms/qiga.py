
from pymoo.algorithms.so_genetic_algorithm import GA, FitnessSurvival, comp_by_cv_and_fitness
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.mating.quantum_mating import QuantumMating
from pymoo.operators.mutation.quantum_mutation import QuantumBitflipMutation
from pymoo.operators.sampling.quantum_sampling import QuantumRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import SingleObjectiveDisplay


class QIGA(GA):
    """
    Extends classical genetic algorithm to its quantum-inspired version.

    See https://link.springer.com/article/10.1007/s10732-010-9136-0 for more information.
    """

    def __init__(
        self,
        pop_size=100,
        sampling=QuantumRandomSampling(),
        selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
        crossover=PointCrossover(n_points=1),
        mutation=QuantumBitflipMutation(prob=0.1),
        survival=FitnessSurvival(),
        rotation=None,
        migration=None,
        catastrophe=None,
        mating=None,
        eliminate_duplicates=False,
        n_offsprings=None,
        display=SingleObjectiveDisplay(),
        **kwargs
    ):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            mating=mating,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            display=display,
            **kwargs
        )

        self.rotation = rotation
        self.migration = migration
        self.catastrophe = catastrophe

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        if mating is None:
            self.mating = QuantumMating(
                selection=self.selection,
                crossover=self.crossover,
                mutation=self.mutation,
                rotation=self.rotation,
                migration=self.migration,
                catastrophe=self.catastrophe,
                repair=self.repair,
                eliminate_duplicates=self.eliminate_duplicates,
            )
        else:
            self.mating = mating
