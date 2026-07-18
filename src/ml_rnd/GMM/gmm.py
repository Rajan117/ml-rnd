import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from similarity import check_similarity
from update_gmm import update_weights
from iteration import convert_truth_to_cluster_indices

def perform_iterations(data: pd.DataFrame, class_column: str):
    columns = data.columns.copy()
    feature_columns = columns[columns != class_column]

    # Standardize feature to zero mean and unit variance so high-magnitude features don't dominate cluster shapes.
    scaler = StandardScaler()
    X = data[feature_columns].to_numpy()
    X = scaler.fit_transform(X)
    print(f"{X.shape[0]} data points with {X.shape[1]} features")

    Y = data[[class_column]].to_numpy().reshape(-1)
    n_components = np.unique(Y).size

    # Align ground truth labels to optimal baseline clusters safely before looping
    temp_gmm = GaussianMixture(n_components=n_components, random_state=42)
    temp_gmm.fit(X)
    temp_preds = temp_gmm.predict(X)
    _, pairings = check_similarity(X, Y, temp_preds)
    Y_aligned = convert_truth_to_cluster_indices(pairings, Y.copy(), n_components)

    # Seed the initial GMM parameters using a small 5% subset of human-labeled ground truth data.
    labelledness = np.zeros(X.shape[0])
    initial_seed_count = int(X.shape[0] * CORRECTION_PERCENTAGE)
    
    initial_indices = np.random.choice(X.shape[0], size=initial_seed_count, replace=False)
    labelledness[initial_indices] = 1

    r_init = np.zeros((X.shape[0], n_components))
    r_init[labelledness == 0] = 1.0 / n_components 
    for idx in initial_indices:
        true_cluster = int(Y_aligned[idx])
        r_init[idx, :] = 0.0
        r_init[idx, true_cluster] = 1.0


    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.means_ = np.zeros((n_components, X.shape[1]))
    gmm.covariances_ = np.zeros((n_components, X.shape[1], X.shape[1]))
    
    # Vectorized initial parameter estimation based on seed labels
    for j in range(n_components):
        r_c = r_init[:, j]
        sum_r_c = r_c.sum()
        gmm.means_[j] = (X * r_c[:, None]).sum(axis=0) / sum_r_c
        diff = X - gmm.means_[j]
        gmm.covariances_[j] = np.dot(diff.T, diff * r_c[:, None]) / sum_r_c + (1e-6 * np.eye(X.shape[1]))
    gmm.weights_ = update_weights(r_init)

    # Calculate initial baseline performance with seed labels included
    r = r_init.copy()
    predictions = np.argmax(r, axis=1)
    similarity, _ = check_similarity(X, Y_aligned, predictions)
    total_labeled = int(labelledness.sum())
    print(f"Iteration 00 | Human-labelled Percentage: {total_labeled/len(labelledness) * 100:.2f} | Similarity Score: {similarity:.4f}")

    correction_count_per_iteration = int(X.shape[0] * CORRECTION_PERCENTAGE)
    iteration = 1
    
    while similarity < 1:
        unlabelled_indices = np.where(labelledness == 0)[0]
        
        if len(unlabelled_indices) == 0:
            print("All data points have been labeled, so no further improvement possible")
            break
            
        if len(unlabelled_indices) > 0:
            unlabelled_r = r[unlabelled_indices]
            
            # Request human-labelling of high uncertainty regions
            entropy = -np.sum(unlabelled_r * np.log(unlabelled_r + ZERO_SUBSTITUTE), axis=1)
            total_likelihood = np.sum(unlabelled_r, axis=1)
            selection_score = entropy * total_likelihood
            
            uncertainty_order = np.argsort(selection_score)[::-1]
            
            take_count = min(correction_count_per_iteration, len(unlabelled_indices))
            new_labeled_indices = unlabelled_indices[uncertainty_order[:take_count]]
            labelledness[new_labeled_indices] = 1

        for _ in range(SUB_ITERATION_COUNT):
            # E-Step: Calculate soft joint probabilities across components
            r = np.zeros((X.shape[0], n_components))
            for j in range(n_components):
                likelihood = multivariate_normal.pdf(X, mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
                r[:, j] = gmm.weights_[j] * likelihood
            
            row_sums = r.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = ZERO_SUBSTITUTE 
            r = r / row_sums
            
            # Re-enforce the hard human constraints over calculated posteriors
            labeled_rows = np.where(labelledness == 1)[0]
            for idx in labeled_rows:
                true_cluster = int(Y_aligned[idx])
                r[idx, :] = 0.0
                r[idx, true_cluster] = 1.0

            new_means = np.zeros((n_components, X.shape[1]))
            new_covariances = np.zeros((n_components, X.shape[1], X.shape[1]))
            
            for j in range(n_components):
                r_c = r[:, j]
                sum_r_c = r_c.sum()
                
                # Compute new means for clusters
                new_means[j] = (X * r_c[:, None]).sum(axis=0) / sum_r_c
                
                # Compute new covariances for clusters
                diff = X - new_means[j]
                new_covariances[j] = np.dot(diff.T, diff * r_c[:, None]) / sum_r_c + (1e-6 * np.eye(X.shape[1]))

            gmm.means_ = new_means
            gmm.covariances_ = new_covariances
            gmm.weights_ = update_weights(r)
        
        # Evaluate performance for the current iteration
        predictions = np.argmax(r, axis=1)
        similarity, _ = check_similarity(X, Y_aligned, predictions)
        total_labeled = int(labelledness.sum())
        print(f"Iteration {iteration:02d} | Human-labelled Percentage: {total_labeled/len(labelledness) * 100:.2f} | Similarity Score: {similarity:.4f}")

        iteration += 1

ZERO_SUBSTITUTE = 1e-12 # Prevents division by zero
SUB_ITERATION_COUNT = 3
CORRECTION_PERCENTAGE = 0.05

if __name__ == "__main__":
    wine: pd.DataFrame = datasets.load_wine(as_frame=True).frame
    perform_iterations(wine, "target")