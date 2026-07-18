import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from similarity import check_similarity
from update_gmm import update_means, update_covariances, update_weights
from iteration import convert_truth_to_cluster_indices

data: pd.DataFrame = datasets.load_wine(as_frame=True).frame
columns = data.columns.copy()
class_column = "target"
feature_columns = columns[columns != class_column]

X = data[feature_columns].to_numpy()
print(f"{X.shape[0]} data points with {X.shape[1]} features")
Y = data[[class_column]].to_numpy().reshape(-1)
n_components = np.unique(Y).size

gmm: GaussianMixture = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)

predictions: np.ndarray = gmm.predict(X)

labelledness = np.zeros(X.shape[0])
similarity, pairings = check_similarity(X, Y, predictions)
total_labeled = int(labelledness.sum())
print(f"Iteration 00 | Labeled Percentage: {total_labeled/len(labelledness) * 100:.2f} | Similarity Score: {similarity:.4f}")

Y_aligned = convert_truth_to_cluster_indices(pairings, Y.copy(), n_components)

correction_count_per_iteration = 5
iteration = 1
while similarity < 1:

    # Simulate human correct classification of a subset of data
    unlabelled_indices = np.where(labelledness == 0)[0]
    if len(unlabelled_indices) > 0:
        new_labeled_indices = np.random.choice(
            unlabelled_indices, 
            size=min(correction_count_per_iteration, len(unlabelled_indices)), 
            replace=False
        )
        labelledness[new_labeled_indices] = 1

    r = np.zeros((X.shape[0], n_components))
    for j in range(n_components):
        likelihood = multivariate_normal.pdf(X, mean=gmm.means_[j], cov=gmm.covariances_[j])
        r[:, j] = gmm.weights_[j] * likelihood
    row_sums = r.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12 # Prevents division by zero
    r = r / row_sums
    
    labeled_rows = np.where(labelledness == 1)[0]
    for idx in labeled_rows:
        true_cluster = int(Y_aligned[idx])
        r[idx, :] = 0.0
        r[idx, true_cluster] = 1.0


    gmm.means_ = update_means(X, r)
    gmm.covariances_ = update_covariances(X, r)
    gmm.weights_ = update_weights(r)
    
    predictions = np.argmax(r, axis=1)
    
    similarity, pairings = check_similarity(X, Y_aligned, predictions)
    total_labeled = int(labelledness.sum())
    print(f"Iteration {iteration:02d} | Labeled Percentage: {total_labeled/len(labelledness) * 100:.2f} | Similarity Score: {similarity:.4f}")

    iteration += 1
