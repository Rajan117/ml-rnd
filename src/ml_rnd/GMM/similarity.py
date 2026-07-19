def jaccard_similarity(set1, set2):
    intersection = np.intersect1d(set1, set2)
    union = np.union1d(set1, set2)

    if union.size == 0:
        return 0
    else:
        return intersection.size / union.size

def calculate_mean_max_similarity(group1, group2):
    max_similarities = []

    for set_i in group1:
        best_match_score = 0
        for set_j in group2:
            score = jaccard_similarity(set_i, set_j)
            if score > best_match_score:
                best_match_score = score
        max_similarities.append(best_match_score)
    
    return sum(max_similarities) / len(max_similarities)

def calculate_hungarian_similarity(predictions, target):
    n = len(predictions)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = jaccard_similarity(predictions[i], target[j])
    
    row_ind, col_ind = linear_sum_assignment(-matrix)
    
    optimal_scores = matrix[row_ind, col_ind]
    avg_similarity = optimal_scores.mean()
    
    pairings = [(row_ind[i], col_ind[i], optimal_scores[i]) for i in range(n)]
    
    return avg_similarity, pairings

def check_similarity(X, Y, predictions, n_components):
    actual_sets = [[] for _ in range(n_components)]
    predicted_sets = [[] for _ in range(n_components)]
    for i in range(predictions.size):
        x_i = X[i]

        y_pred = int(predictions[i])
        y_i = int(Y[i])
        
        actual_sets[y_i].append(i)
        predicted_sets[y_pred].append(i)

    return calculate_hungarian_similarity(actual_sets, predicted_sets)

import numpy as np
from scipy.optimize import linear_sum_assignment
