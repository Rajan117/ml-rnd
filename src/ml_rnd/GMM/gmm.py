import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import pandas as pd
from similarity import check_similarity
from update_gmm import update_means, update_covariances, update_weights
from iteration import convert_truth_to_cluster_indices, perform_corrections

wine: pd.DataFrame = datasets.load_wine(as_frame=True).frame
wine_simple: pd.DataFrame = wine[["alcohol", "color_intensity", "target"]]

gmm: GaussianMixture = GaussianMixture(n_components=3)
X = wine_simple[["alcohol", "color_intensity"]].to_numpy()
gmm.fit(X)

predictions: np.ndarray = gmm.predict(X)
Y = wine_simple[["target"]].to_numpy().reshape(-1)

similarity, pairings = check_similarity(X, Y, predictions)
print("Unsupervised Similarity: {0}".format(similarity))
print("Unsupervised Pairings: {0}".format(pairings))
print("")

r = gmm.predict_proba(X)

supervised_iteration_count = 10
iteration_correction_count = 5
for i in range(supervised_iteration_count):
    iteration = i + 1

    Y = convert_truth_to_cluster_indices(pairings, Y, 3)
    r = perform_corrections(X, Y, r, iteration_correction_count, 3)
    new_means = update_means(X, r)
    new_covariances = update_covariances(X, r)
    new_weights = update_weights(r)

    gmm.means_ = new_means
    gmm.covariances_ = new_covariances
    gmm.weights_ = new_weights

    gmm: GaussianMixture = GaussianMixture(n_components=3, covariance_type="full")
    gmm.precisions_cholesky_ = _compute_precision_cholesky(new_covariances, 'full')
    gmm.fit(X)
    predictions: np.ndarray = gmm.predict(X)

    similarity, pairings = check_similarity(X, Y, predictions)
    print("Iteration {0} Similarity: {1}".format(iteration, similarity))
    print("Iteration {0} Pairings: {1}".format(iteration, pairings))
    print("")
