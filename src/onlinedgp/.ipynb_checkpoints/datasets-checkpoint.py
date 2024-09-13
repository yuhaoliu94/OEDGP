import numpy as np


class DataSet:

    def __init__(self, name: str, fold: int, Y: np.ndarray, X: np.ndarray, Z: list = None):
        self._Y = self.correct_format(Y)
        self._X = self.correct_format(X)
        self._Z = self.correct_list(Z)
        self._name = name
        self._fold = fold
        self._Num_Observations, self._Dy = self._Y.shape
        self._Num_Observations_X, self._Dx = self._X.shape
        assert self.Num_Observations == self._Num_Observations_X, ("The sample sizes are not consistent for input and "
                                                                   "output.")

    @staticmethod
    def correct_format(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))

        return X

    @staticmethod
    def correct_list(X: list) -> list:
        if X is None:
            return []

        X_new = []
        for _X in X:
            _X = np.array(_X)
            if _X.ndim == 1:
                _X = _X.reshape(-1, 1)
            X_new.append(_X)

        return X_new

    @property
    def Num_Observations(self):
        return self._Num_Observations

    @property
    def Dy(self):
        return self._Dy

    @property
    def Dx(self):
        return self._Dx

    @property
    def Y(self):
        return self._Y

    @property
    def X(self):
        return self._X

    @property
    def Z(self):
        return self._Z

    @property
    def name(self):
        return self._name

    @property
    def fold(self):
        return self._fold
