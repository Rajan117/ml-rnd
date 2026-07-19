import numpy as np

def update_weights(r: np.array):
    """
    Update all cluster mixing weights with updated responsibilities.
    
    Args:
        X: The training data (n features x k samples)
        r: The responsibilities for the cluster (k samples x d components)
        
    Returns:
        The new covariances
    """

    new_weights = np.zeros(r.shape[1])
    for i in range(r.shape[1]):
        new_weights[i] = calculate_new_cluster_weight(r[:, i])

    return new_weights

def calculate_new_cluster_weight(r_c: np.array):
    """
    Calculate new weight of GMM cluster with updated responsibilities.
    
    Args:
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The new mean
    """

    return calculate_cluster_responsibility_sum(r_c) / r_c.size

def calculate_cluster_responsibility_sum(r_c: np.array):
    """
    Calculate the sum of all responsibilities for this cluster.
    
    Args:
        r_c: The responsibilities for the cluster (k samples)
        
    Returns:
        The responsibility sum
    """

    return r_c.sum(axis=0)