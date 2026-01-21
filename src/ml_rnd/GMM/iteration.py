import numpy as np
import random

def convert_truth_to_cluster_indices(pairings: np.array, Y: np.array, n: int):
    """Converts the truth values to the clustering indices output by model
       Would normally be assigned by human reviewer
    
    Args:
        pairings: The set pairings calculated by Hungarian
        Y: The true classes (k samples)
        n: The number of components
        
    Returns:
        The converted truth values
    """

    # 2 stages prevent replacement of unreplaced value
    Y_mid = Y
    for pred, truth, _ in pairings:
        Y_mid[Y_mid == truth] = pred + n

    Y_out = Y_mid
    for i in range(Y_mid.size):
        Y_out[i] = Y_mid[i] - n

    return Y_out

def perform_corrections(X: np.array, Y: np.array, r_c: np.array, correction_count: int, n: int):
    """Update the responsibilties using the known classes (would normally be done by human review)
    
    Args:
        X: The training data (n features x k samples)
        Y: The true classes (k samples)
        r_c: The responsibilities for the cluster (k samples)
        correction_count: How many corrections to perform
        n: The number of components
        
    Returns:
        The updated responsibilities
    """

    r_new = r_c

    for _ in range(correction_count):
        index = random.randrange(0, Y.size)

        y_i = int(Y[index])
        r_ci_new = r_c[index]
        for j in range(n):
            if j == y_i:    # Force responsibility of known matching component to 1
                r_ci_new[j] = 1
            else:           # Force responsibility of known non-matching components to 0
                r_ci_new[j] = 0

        r_new[index] = r_ci_new

    return r_new