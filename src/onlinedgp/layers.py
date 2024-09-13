from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from src.onlinedgp.distributions import Normal
from src.onlinedgp.functions import RandomFeatureGP
from src.onlinedgp.utils import normalize_weights, get_sequential_mse, get_sequential_mnll


class Layer(ABC):

    def __init__(self, dim: int, num_particle: int = 1000, num_rff: int = 50,
                 warm_start: int = 0, learning_rate: float = 0.001, din: int = None) -> None:
        # assign after initializing structure
        self.next_layer = None
        self.prev_layer = None

        self.din = din
        self.function = None

        # scalar
        self.t = 0
        self.dim = dim
        self.dout = self.dim

        self.M = num_particle
        self.J = num_rff
        self.warm_start = warm_start
        self.learning_rate = learning_rate

        # particles
        self.particle_weights_for_prev_layer = np.zeros(self.M)

        self.current_particle_state = None
        self.current_state = None
        self.stored_states = []

    def initialize_particle_state(self) -> np.ndarray:
        normal_generator = Normal()
        return normal_generator.sample_univariate(0, 1, (self.M, self.dim))

    def initialize_transition_function(self):
        if self.din is None:
            self.din = self.prev_layer.dim
        self.function = RandomFeatureGP(self.din, self.dout, self.J, self.warm_start, self.learning_rate)

    def get_input_particles(self) -> np.ndarray:
        return self.prev_layer.current_particle_state

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Class must override predict")

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Class must override update")


class RootLayer(Layer):

    def get_input_particles(self, x: np.ndarray = None) -> np.ndarray:
        replicate_x = np.repeat(x[np.newaxis, :], self.M, axis=0)  # M * Dx
        return replicate_x

    def predict(self, x: np.ndarray = None) -> None:
        predict_particle = self.function.predict(self.get_input_particles(x))
        self.current_particle_state = predict_particle

    def filter(self) -> None:
        weights = self.next_layer.particle_weights_for_prev_layer

        self.current_state = np.average(self.current_particle_state, weights=weights, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        # replicate_current_state = np.repeat(self.current_state[np.newaxis, :], self.M, axis=0)  # M * dim
        # log_likelihood = self.function.cal_log_likelihood(replicate_current_state)  # M
        # self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

    def update(self, x: np.ndarray = None) -> None:
        self.function.update(x, self.current_state)

        self.t += 1


class HiddenLayer(Layer):

    def predict(self) -> None:
        predict_particle = self.function.predict(self.get_input_particles())
        self.current_particle_state = predict_particle

    def filter(self) -> None:
        weights = self.next_layer.particle_weights_for_prev_layer

        self.current_state = np.average(self.current_particle_state, weights=weights, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        replicate_current_state = np.repeat(self.current_state[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_current_state)  # M
        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

    def update(self) -> None:
        self.function.update(self.prev_layer.current_state, self.current_state)

        self.t += 1


class ObservationLayer(Layer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.y = None

        self.y_log_likelihood = 0.0
        self.stored_y_log_likelihood = []

        self.y_log_likelihood_forward = 0.0
        self.stored_y_log_likelihood_forward = []

        self.mse = 0.0
        self.mnll = 0.0

    def predict(self) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles())
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

    def filter(self, y) -> None:
        self.y = y

        replicate_actual_realization = np.repeat(y[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_actual_realization)  # M

        self.particle_weights_for_prev_layer = normalize_weights(log_likelihood)

        input_vector = np.average(self.prev_layer.current_particle_state, axis=0)
        cum_log_likelihood_forward = self.function.cal_y_log_likelihood_forward(input_vector, y)
        self.y_log_likelihood_forward = cum_log_likelihood_forward - np.sum(self.stored_y_log_likelihood_forward)
        self.stored_y_log_likelihood_forward.append(self.y_log_likelihood_forward)

    def update(self) -> None:
        self.function.update(self.prev_layer.current_state, self.y)

        self.t += 1

        # log_likelihood
        cum_log_likelihood = self.function.cal_y_log_likelihood()
        self.y_log_likelihood = cum_log_likelihood - np.sum(self.stored_y_log_likelihood)  # np.average(log_likelihood)
        self.stored_y_log_likelihood.append(self.y_log_likelihood)

        # mse
        self.mse = get_sequential_mse(self.mse, self.t - 1, self.y, self.current_state)

        # mnll
        self.mnll = get_sequential_mnll(self.mnll, self.t - 1, self.y_log_likelihood_forward)
