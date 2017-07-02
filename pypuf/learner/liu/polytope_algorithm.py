from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from numpy.random import RandomState
from pypuf.learner.base import Learner
from numpy import full, double
from sys import stderr

class PolytopeAlgorithm(Learner):
    def __init__(self, t_set, n, k, transformation=LTFArray.transform_id, combiner=LTFArray.combiner_xor, weights_mu=0, weights_sigma=1, weights_prng=RandomState()):
        """
        Initialize a LTF Array Polytope Learner for the specified LTF Array.

        :param t_set: The training set, i.e. a data structure containing challenge response pairs
        :param n: Input length
        :param k: Number of parallel LTFs in the LTF Array
        :param transformation: Input transformation used by the LTF Array
        :param combiner: Combiner Function used by the LTF Array (Note that not all combiner functions are supported by this class.)
        :param weights_mu: mean of the Gaussian that is used to choose the initial model
        :param weights_sigma: standard deviation of the Gaussian that is used to choose the initial model
        :param weights_prng: PRNG to draw the initial model from. Defaults to fresh `numpy.random.RandomState` instance.
        """
        self.iteration_count = 0
        self.__training_set = t_set
        self.n = n
        self.k = k
        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.weights_prng = weights_prng
        self.iteration_limit = 100
        self.convergence_decimals = 3
        self.sign_combined_model_responses = None
        self.sigmoid_derivative = full(self.training_set.N, None, double)
        self.min_distance = 1
        self.transformation = transformation
        self.combiner = combiner
        self.transformed_challenges = self.transformation(self.training_set.challenges, k)

        assert self.n == len(self.training_set.challenges[0])

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, val):
        self.__training_set = val
    
    def learn(self):
        model = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, self.weights_mu, self.weights_sigma, self.weights_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.iteration_count = 0
        while self.iteration_count < self.iteration_limit:
            stderr.write('\riter %5i         ' % (self.iteration_count))
            self.iteration_count += 1
        return model
    
    def __chebyshev_center(self):
        return
    
    def __closest_challenge(self):
        return