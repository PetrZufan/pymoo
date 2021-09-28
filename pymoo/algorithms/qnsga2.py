
from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mating.quantum_mating import QuantumMating
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.quantum_sampling import QuantumRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay


class QNSGA2(NSGA2):

    def __init__(self,
                 pop_size=100,
                 sampling=QuantumRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 rotation=None,
                 migration=None,
                 catastrophe=None,
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        self.rotation = rotation
        self.migration = migration
        self.catastrophe = catastrophe

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

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


parse_doc_string(QNSGA2.__init__)
