import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import pandas as pd
from similarity import calculate_hungarian_similarity
from iteration import calculate_new_cluster_mean, calculate_new_cluster_covariance, calculate_new_cluster_weight

wine: pd.DataFrame = datasets.load_wine(as_frame=True).frame
wine_simple: pd.DataFrame = wine[["alcohol", "color_intensity", "target"]]

gmm: GaussianMixture = GaussianMixture(n_components=3)
training = wine_simple[["alcohol", "color_intensity"]].to_numpy()
gmm.fit(training)

predictions: np.ndarray = gmm.predict(training)


wine_simple["prediction"] = predictions
actual_sets = [[], [], []]
predicted_sets = [[], [], []]
for _, row in wine_simple.iterrows():
    train_cols = row[["alcohol", "color_intensity"]]

    target = int(row["target"])
    prediction = int(row["prediction"])
    
    actual_sets[target].append(train_cols)
    predicted_sets[prediction].append(train_cols)

similarity, pairings = calculate_hungarian_similarity(actual_sets, predicted_sets)
print("Unsupervised Similarity: {0}".format(similarity))
print("Unsupervised Pairings: {0}".format(pairings))

responsibilities = gmm.predict_proba(training)

print(gmm.means_)
print(calculate_new_cluster_mean(training, responsibilities[:, 0]))
print(calculate_new_cluster_mean(training, responsibilities[:, 1]))
print(calculate_new_cluster_mean(training, responsibilities[:, 2]))
print("")
print(gmm.covariances_)
print(calculate_new_cluster_covariance(training, responsibilities[:, 0]))
print(calculate_new_cluster_covariance(training, responsibilities[:, 1]))
print(calculate_new_cluster_covariance(training, responsibilities[:, 2]))
print("")
print(gmm.weights_)
print(calculate_new_cluster_weight(responsibilities[:, 0]))
print(calculate_new_cluster_weight(responsibilities[:, 1]))
print(calculate_new_cluster_weight(responsibilities[:, 2]))

supervised_iteration_count = 10
iteration_correction_count = 5
for i in range(supervised_iteration_count):
    iteration = i + 1
