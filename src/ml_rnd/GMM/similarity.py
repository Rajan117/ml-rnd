def jaccard_similarity(set1, set2):
    print(set1)
    print(set2)
    intersection = np.intersect1d(set1, set2)
    union = np.union1d(set1, set2)

    print(intersection.size)
    print(union.size)
    print("")

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

    print("Max similarities: {0}".format(max_similarities))
    print("")
    
    return sum(max_similarities) / len(max_similarities)

def calculate_hungarian_similarity(group1, group2):
    n = len(group1)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = jaccard_similarity(group1[i], group2[j])
    
    row_ind, col_ind = linear_sum_assignment(-matrix)
    
    optimal_scores = matrix[row_ind, col_ind]
    avg_similarity = optimal_scores.mean()
    
    pairings = [(row_ind[i], col_ind[i], optimal_scores[i]) for i in range(n)]
    
    return avg_similarity, pairings

import numpy as np
from scipy.optimize import linear_sum_assignment
