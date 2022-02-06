"""
includes the code of regression for single DGP and ensemble DGP.
"""
import os
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import t, chi2
from copy import deepcopy

import utilities

warnings.filterwarnings('ignore')


class SRFDGP:

    def __init__(self, X, Y, config):
        ######## Stucture #########
        self.X = X  # Store Input. Dimension: (N, D = D_0)
        self.Y = Y  # Store Output. Dimension: (N, Q = D_{L+1})
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.Q = Y.shape[1]

        if config["structure"]["dim"] is None:
            self.dim = [self.D + i for i in range(self.Q - self.D + 1)] if self.Q > self.D else \
                [self.D - i for i in range(self.D - self.Q + 1)]  # Dimensions for L+2 layers
        else:
            self.dim = config["structure"]["dim"]  # [D_0, D_1, ..., D_L, D_{L+1}]

        self.L = len(self.dim) - 2

        ######## Hyper-Parameters ########
        self.J = config["hyper"]["J"]  # Number of random features
        self.M = config["hyper"]["M"]  # Number of particles
        self.max_iter = config["hyper"]["max_iter"]  # Number of iterations
        self.func_init_var = config["hyper"]["func_init_var"]  # Prior variance for \epsilon_l. Dimension: (L+1, )
        self.time_vary_var = config["hyper"][
            "time_vary_var"]  # Assigned variance for \nu_l^j. Dimension: (l, D_{l+1}), l=0,...,L

        ######## Parameters ########
        self.kernel_var = config["parameter"][
            "kernel_var"]  # Variance of kernel = prior variance for \theta_l^j. Dimension: (l,D_{l+1}), l=0,...,L
        self.kernel_length = config["parameter"][
            "kernel_length"]  # Lengthscales of kernel, used for sample \v_l. Dimension: (l, D_l), l=0,...,L

        self.mu = [[np.zeros(2 * self.J) for _ in range(Dl)] for Dl in
                   self.dim[1:]]  # Mean of \theta_l^j. Dimension: (l,D_{l+1},2J), l=0, ..., L
        self.Sigma = [[np.eye(2 * self.J) * self.kernel_var[l][j] for j in range(Dl)] for l, Dl in
                      enumerate(self.dim[1:])]  # Cov matrix of \theta_l^j. Dimension: (l,D_{l+1},2J,2J), l=0, ..., L
        self.v = [np.random.multivariate_normal(mean=np.zeros(Dl), cov=np.diag(
            1 / (4 * pow(np.pi, 2) * pow(np.array(self.kernel_length[l]), 2))), size=self.J) for l, Dl in
                  enumerate(self.dim[:-1])]  # \v_l. Dimension: (l,J,D_l), l=0,...,L

        ######## Hidden States ########
        self.Xs = [None] + [np.zeros((self.M, Dl)) for Dl in
                            self.dim[1:]]  # Particles in hidden states. Dimension: (l,M,D_l), l=0, ..., L+1
        self.Xmean = [None] + [np.zeros((self.M, Dl)) for Dl in
                               self.dim[1:]]  # Mean of particles in hidden states. Dimension: (l,M,D_l), l=0, ..., L+1
        self.Xvar = [None] + [np.zeros((self.M, Dl)) for Dl in self.dim[
                                                               1:]]  # Varainces of particles in hidden states. Dimension: (l,M,D_l), l=0, ..., L+1
        self.Xhat = [None] + [np.zeros(Dl) for Dl in
                              self.dim[1:]]  # MMSE in hidden states. Dimension: (l,D_l), l=0, ..., L+1

        ######## Stored Estimation and Prediction ########
        self.x_estimation = []  # (N, l, D_l), l=1,...,L
        self.y_prediction = []  # (N, Q)
        self.NLL = []

    def Transition(self, x):
        for l in range(1, len(self.dim)):
            Dl = self.dim[l]
            for m in range(self.M):
                state = self.Xs[l - 1][m, :].reshape(-1, 1) if l > 1 else x.reshape(-1, 1)  # x_{l-1}^{(m)}. (D_{l-1},1)
                project = np.dot(self.v[l - 1], state).ravel()  # (J, D_{l-1}) * (D_{l-1}, 1) = (J, )
                feature = np.array([np.sin(project), np.cos(project)]).T.ravel() / np.sqrt(
                    self.J)  # Random Feature \phi. Dimension: (2J, )
                for d in range(Dl):
                    sufficient = np.dot(self.Sigma[l - 1][d],
                                        feature.reshape(-1, 1)).ravel()  # (2J, 2J) * (2J, 1) = (2J,)
                    self.Xmean[l][m, d] = np.dot(feature, self.mu[l - 1][d])  # (2J, ) * (2J,) = (0, )
                    self.Xvar[l][m, d] = np.dot(feature, sufficient) + self.func_init_var[
                        l - 1]  # (2J, ) * (2J,) = (0, )
                    self.Xs[l][m, d] = np.random.normal(self.Xmean[l][m, d],
                                                        np.sqrt(self.Xvar[l][m, d]))  # x_l^{(m)}. (0, )
        self.Xhat[-1] = np.average(self.Xs[-1], axis=0)

    def Transition_TV(self, x):
        self.v = [self.v[l] * (np.ones((self.J, Dl)) + np.random.normal(0, 1e-4, (self.J, Dl))) for l, Dl in
                  enumerate(self.dim[:-1])]  # Dimension: (l,J,D_l), l=0,...,L
        self.Transition(x)

    def Prediction(self, x):
        tmp = x.reshape(-1, 1)
        for l in range(1, len(self.dim)):
            Dl = self.dim[l]
            state = tmp
            project = np.dot(self.v[l - 1], state).ravel()
            feature = np.array([np.sin(project), np.cos(project)]).T.ravel() / np.sqrt(self.J)
            tmp = np.dot(np.array(self.mu[l - 1]), feature.reshape(-1, 1)).ravel()
        self.y_prediction.append(tmp)

    def Filtering(self, y, approx=True):
        nll = -log_mvn_pdf(y, self.y_prediction[-1], np.eye(len(y)) * self.func_init_var[-1])
        self.NLL.append(nll)

        for l in reversed(range(2, len(self.dim))):
            if approx:
                Xhat = self.Xhat[l] if l < self.L + 1 else y
                w = [log_mvn_pdf(Xhat, self.Xmean[l][m, :], self.Xvar[l][m, :]) for m in range(self.M)]
                w -= max(w)
                w = np.exp(w)
                w /= sum(w)
                self.Xhat[l - 1] = np.average(self.Xs[l - 1], axis=0, weights=w)
            else:
                w = np.zeros(self.M)
                for m in range(self.M):
                    Xhat = self.Xs[l] if l < self.L + 1 else np.tile(y.reshape(1, -1), (self.M, 1))  # (M, D_l)
                    w[m] = np.mean(
                        [log_mvn_pdf(Xhat[mm, :], self.Xmean[l][m, :], self.Xvar[l][m, :]) for mm in range(self.M)])
                w -= max(w)
                w = np.exp(w)
                w /= sum(w)
                self.Xhat[l - 1] = np.average(self.Xs[l - 1], axis=0, weights=w)
                index = random.choices(range(self.M), w, k=self.M)
                self.Xs[l - 1] = np.array([self.Xs[l - 1][i, :] for i in index])

    def Bayesian(self, x, y):
        for l in range(1, len(self.dim)):
            Dl = self.dim[l]
            state = self.Xhat[l - 1].reshape(-1, 1) if l > 1 else x.reshape(-1, 1)  # \hat{x}_{l-1}. (D_{l-1},1)
            project = np.dot(self.v[l - 1], state).ravel()
            feature = np.array([np.sin(project), np.cos(project)]).T.ravel() / np.sqrt(self.J)
            Xhats = self.Xhat[l] if l < self.L + 1 else y  # \hat{x}_l. (D_l, )
            for d in range(Dl):
                predict_mean = np.dot(feature, self.mu[l - 1][d])
                sufficient = np.dot(self.Sigma[l - 1][d], feature.reshape(-1, 1)).ravel()
                predict_var = np.dot(feature, sufficient) + self.func_init_var[l - 1]

                self.mu[l - 1][d] += sufficient * (Xhats[d] - predict_mean) / predict_var
                self.Sigma[l - 1][d] -= np.dot(sufficient.reshape(-1, 1), sufficient.reshape(1, -1)) / predict_var

    def KF(self, x, y):
        for l in range(1, len(self.dim)):
            Dl = self.dim[l]
            state = self.Xhat[l - 1].reshape(-1, 1) if l > 1 else x.reshape(-1, 1)  # \hat{x}_{l-1}. (D_{l-1},1)
            project = np.dot(self.v[l - 1], state).ravel()
            feature = np.array([np.sin(project), np.cos(project)]).T.ravel() / np.sqrt(self.J)
            Xhats = self.Xhat[l] if l < self.L + 1 else y  # \hat{x}_l. (D_l, )
            for d in range(Dl):
                predict_mean = np.dot(feature, self.mu[l - 1][d])
                sufficient = np.dot(self.Sigma[l - 1][d] + self.time_vary_var[l - 1][d] * np.eye(2 * self.J),
                                    feature.reshape(-1, 1)).ravel()
                predict_var = np.dot(feature, sufficient) + self.func_init_var[l - 1]

                self.mu[l - 1][d] += sufficient * (Xhats[d] - predict_mean) / predict_var
                self.Sigma[l - 1][d] += self.time_vary_var[l - 1][d] * np.eye(2 * self.J) - np.dot(
                    sufficient.reshape(-1, 1), sufficient.reshape(1, -1)) / predict_var

    def State_Estimate(self):
        self.x_estimation.append(list(self.Xhat[1:-1]))

    def Train(self, x, y, method="Bayesian"):
        for i in range(self.max_iter):
            self.Transition(x)

            if i == 0:
                self.Prediction(x)

            self.Filtering(y)

            if method == "Bayesian":
                self.Bayesian(x, y)
            elif method == "KF":
                self.KF(x, y)

        self.State_Estimate()

class ERFDGP_TV:

    def __init__(self, X, Y, S, config):

        ######## Stucture #########
        self.X = X  # Store Input. Dimension: (N, D = D_0)
        self.Y = Y  # Store Output. Dimension: (N, Q = D_{L+1})
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.Q = Y.shape[1]
        self.S = S
        self.U = [None] * self.S

        if config["structure"]["dim"] is None:
            self.dim = [self.D + i for i in range(self.Q - self.D + 1)] if self.Q > self.D else [self.D - i for i in
                                                                                                 range(
                                                                                                     self.D - self.Q + 1)]  # Dimensions for L+2 layers
        else:
            self.dim = config["structure"]["dim"]  # [D_0, D_1, ..., D_L, D_{L+1}]

        self.L = len(self.dim) - 2

        ######## Hyper-Parameters ########
        self.J = config["hyper"]["J"]  # Number of random features
        self.M = config["hyper"]["M"]  # Number of particles
        self.max_iter = config["hyper"]["max_iter"]  # Number of iterations
        self.func_init_var = config["hyper"]["func_init_var"]  # Prior variance for \epsilon_l. Dimension: (L+1, )
        self.time_vary_var = config["hyper"][
            "time_vary_var"]  # Assigned variance for \nu_l^j. Dimension: (l, D_{l+1}), l=0,...,L

        ######## Parameters ########
        self.models = []
        for i in range(S):
            config["parameter"] = {
                "kernel_var": [[rdm() for _ in range(Dl)] for Dl in self.dim[1:]],
                "kernel_length": [[rdm() for _ in range(Dl)] for Dl in self.dim[:-1]]}
            model = SRFDGP(X, Y, config)
            self.models.append(model)

        ######## Stored Estimation and Prediction ########
        self.x_estimation = []  # (N, l, D_l), l=1,...,L
        self.y_prediction = []  # (N, Q)
        self.NLL = []
        self.weights = np.ones(self.S) / self.S

    def Transition(self, x):
        for s in range(self.S):
            self.models[s].Transition_TV(x)

    def Prediction(self, x):
        for s in range(self.S):
            self.models[s].Prediction(x)
        y_preds = np.array([self.models[s].y_prediction[-1] for s in range(self.S)])
        self.y_prediction.append(np.average(y_preds, axis=0))

    def Filtering(self, y, approx=True):
        for s in range(self.S):
            self.models[s].Filtering(y)
        w = np.array([-self.models[s].NLL[-1] for s in range(self.S)])
        w += np.log(self.weights)
        w -= max(w)
        w = np.exp(w)
        w /= sum(w)
        self.weights = w

    def Bayesian(self, x, y):
        for s in range(self.S):
            self.models[s].Bayesian(x, y)

    def KF(self, x, y):
        for s in range(self.S):
            self.models[s].KF(x, y)

    def State_Estimate(self):
        for s in range(self.S):
            self.models[s].State_Estimate()

    def Resample(self):
        index = np.random.choice(range(self.S), size=self.S, p=self.weights)
        self.models = [deepcopy(self.models[s]) for s in index]

    def Train(self, x, y, method="Bayesian"):
        self.Transition(x)

        self.Prediction(x)

        self.Filtering(y)

        if method == "Bayesian":
            self.Bayesian(x, y)
        elif method == "KF":
            self.KF(x, y)

        self.State_Estimate()

        self.Resample()

    def ensemble_states_2d(self):
        X_hat = [np.array(self.models[s].x_estimation)[:, 0, :] for s in range(self.S)]
        for s in range(self.S):
            u, _, _ = np.linalg.svd(X_hat[s])
            self.U[s] = u[:, :2]

        m = idx = 0
        for i in range(self.N):
            norms = np.linalg.norm(self.U[0][i, :])
            if norms >= m:
                m = norms
                idx = i

        U_guidence = [self.U[s][idx, :] / np.linalg.norm(self.U[s][idx, :]) for s in range(self.S)]

        for s in range(self.S):
            if U_guidence[0][0] * U_guidence[s][0] < 0:
                self.U[s][:, 0] = -self.U[s][:, 0]
                U_guidence[s][0] = -U_guidence[s][0]
            if U_guidence[0][1] * U_guidence[s][1] < 0:
                self.U[s][:, 1] = -self.U[s][:, 1]
                U_guidence[s][1] = -U_guidence[s][1]

        U_theta = [np.arccos(np.dot(U_guidence[0], U_guidence[s])) for s in range(self.S)]
        U_rotate = [self.rotate_2d(U_guidence[0], U_guidence[s], U_theta[s]) for s in range(self.S)]
        U_rotated = np.array([np.dot(self.U[s], U_rotate[s]) for s in range(self.S)])
        return np.average(U_rotated, axis=0, weights=self.weights)

    def rotate_2d(self, a, b, theta):
        rotate = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        if np.linalg.norm(a - np.dot(rotate, b)) > np.linalg.norm(a - np.dot(rotate.T, b)):
            return rotate
        return rotate.T
