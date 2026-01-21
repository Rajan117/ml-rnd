import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def calculate_new_cluster_mean(X: np.array, r_c: np.array):
    """Calculate new mean of GMM cluster with updated responsibilities.
    
    Args:
        X: The training data (n features x k samples)
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The new mean
    """

    sum = 0

    for i in range(r_c.size):
        r_ic = r_c[i]
        x_i = X[i]

        sum += r_ic * x_i

    return sum / calculate_cluster_responsibility_sum(r_c)

def calculate_new_cluster_covariance(X: np.array, r_c: np.array):
    """Calculate new covariance of GMM cluster with updated responsibilities.
    
    Args:
        X: The training data (n features x k samples)
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The new mean
    """

    u_c = calculate_new_cluster_mean(X, r_c)

    sum = 0

    for i in range(r_c.size):
        r_ic = r_c[i]
        x_i = X[i]

        diff = x_i - u_c
        sum += r_ic * np.outer(diff, diff)

    return sum / calculate_cluster_responsibility_sum(r_c)

def calculate_new_cluster_weight(r_c: np.array):
    """Calculate new weight of GMM cluster with updated responsibilities.
    
    Args:
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The new mean
    """

    return calculate_cluster_responsibility_sum(r_c) / r_c.size

def calculate_cluster_responsibility_sum(r_c: np.array):
    """Calculate the sum of all responsibilities for this cluster.
    
    Args:
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The responsibility sum
    """

    return r_c.sum(axis=0)