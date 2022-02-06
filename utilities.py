import os
import math

from scipy.stats import norm, multivariate_normal


# Compute the pdf of multivariate normal distribution
def log_norm_pdf(y, mu, sigma2):  # y, mu, cov are numbers
    normal = norm(mu, np.sqrt(sigma2))
    return normal.logpdf(y)


def log_mvn_pdf(y, mu, cov):  # y and mu have size (dim,); cov has size (dim,dim);
    mvn = multivariate_normal(mean=mu, cov=cov)
    return mvn.logpdf(y)

def normalize(X):
    D = X.shape[1]
    for d in range(D):
        X[:, d] = (X[:, d] - np.mean(X[:, d])) / np.std(X[:, d])
    return X

def softmax(y_pred):
    m = max(y_pred)
    y_pred_2 = y_pred - m
    logsum = m + np.log(sum(np.exp(y_pred_2)))
    return np.exp(y_pred - logsum)
