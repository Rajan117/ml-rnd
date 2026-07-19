import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from similarity import check_similarity
from update_gmm import update_weights
from iteration import convert_truth_to_cluster_indices

def estimate_m_step(X: np.ndarray, r: np.ndarray, n_components: int):
    """
    Computes vectorized GMM parameter estimations for means, covariances, and weights.
    
    Args:
        X: The training data (n features x k samples)
        r: The responsibilities for the clusters (k samples x d components)
        n_components: The number of components
        
    Returns:
        Tuple containing new means, covariances and weights
    """

    new_means = np.zeros((n_components, X.shape[1]))
    new_covariances = np.zeros((n_components, X.shape[1], X.shape[1]))
    
    for j in range(n_components):
        r_c = r[:, j]
        sum_r_c = r_c.sum()
        
        new_means[j] = (X * r_c[:, None]).sum(axis=0) / sum_r_c
        
        diff = X - new_means[j]
        new_covariances[j] = np.dot(diff.T, diff * r_c[:, None]) / sum_r_c + (1e-6 * np.eye(X.shape[1]))
        
    new_weights = update_weights(r)
    return new_means, new_covariances, new_weights


def log_iteration_metrics(iteration_str: str, X: np.ndarray, Y_aligned: np.ndarray, r: np.ndarray, n_components: int, labelledness: np.ndarray, metrics_map: dict):
    """
    Outputs metrics and logs for final plot
    
    Args:
        iteration_str: The padded string format of the current loop iteration index
        X: The training data (n features x k samples)
        Y_aligned: The target classes mapped to the model's component indices (k samples)
        r: The responsibilities for the cluster (k samples x d components)
        n_components: The number of components
        labelledness: Tracking mask where 1 indicates a human-labeled sample (k samples)
        metrics_map: A dictionary accumulating similarity histories mapped to specific label percentages
        
    Returns:
        The calculated Hungarian Jaccard similarity score for the current state
    """

    predictions = np.argmax(r, axis=1)
    similarity, _ = check_similarity(X, Y_aligned, predictions, n_components)
    total_labeled = int(labelledness.sum())
    labelled_percentage = total_labeled / len(labelledness) * 100
    
    print(f"Iteration {iteration_str} | Human-labelled Percentage: {labelled_percentage:.2f} | Similarity Score: {similarity:.4f}")
    
    if labelled_percentage in metrics_map:
        metrics_map[labelled_percentage].append(similarity)
    else:
        metrics_map[labelled_percentage] = [similarity]
        
    return similarity


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
    _, pairings = check_similarity(X, Y, temp_preds, n_components)
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
    
    gmm.means_, gmm.covariances_, gmm.weights_ = estimate_m_step(X, r_init, n_components)

    r = r_init.copy()
    similarity = log_iteration_metrics("00", X, Y_aligned, r, n_components, labelledness, percentage_to_similarity_map)

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

            gmm.means_, gmm.covariances_, gmm.weights_ = estimate_m_step(X, r, n_components)
        
        similarity = log_iteration_metrics(f"{iteration:02d}", X, Y_aligned, r, n_components, labelledness, percentage_to_similarity_map)
        iteration += 1

ZERO_SUBSTITUTE = 1e-12 
SUB_ITERATION_COUNT = 3
CORRECTION_PERCENTAGE = 0.05

percentage_to_similarity_map = {}
if __name__ == "__main__":
    #df: pd.DataFrame = datasets.load_wine(as_frame=True).frame
    df: pd.DataFrame = datasets.load_digits(as_frame=True).frame
    REPEATS = 1000
    for _ in range(REPEATS):
        perform_iterations(df, "target")
        print("-------------------------------")

    x = np.array(list(percentage_to_similarity_map.keys()))
    similarity_lists = list(percentage_to_similarity_map.values())
    y = np.array([np.array(arr).mean() * 100 for arr in similarity_lists])

    print("AVERAGED labelled vs similarity")
    for i in range(len(x)):
        iteration_str = f"{i:02d}"
        percentage_labelled = x[i]
        similarity = y[i]

        print(f"Iteration {iteration_str} | Human-labelled Percentage: {percentage_labelled:.2f} | Similarity Score: {similarity:.4f}")
    
    # Plot data points for labelled percentage vs accuracy for semi-supervised classification
    plt.scatter(x, y, label="Semi-supervised classification")

    # Plot data points for labelled percentage vs accuracy for pure manual classification
    true_manual = [i for i in range(100)]
    plt.plot(true_manual, true_manual, color="crimson", linestyle="--", linewidth=2, label="Pure manual classification", zorder=2)

    plt.xlabel("Percentage of training data labelled manually (%)")
    plt.ylabel("Accuracy of training data clustering (%)")

    plt.legend()
    plt.show()