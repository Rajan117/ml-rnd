import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import pandas as pd


wine: pd.DataFrame = datasets.load_wine(as_frame=True).frame
wine_simple: pd.DataFrame = wine[["alcohol", "color_intensity", "target"]]

gmm: GaussianMixture = GaussianMixture(n_components=3)
gmm.fit(wine_simple[["alcohol", "color_intensity"]])

predictions: np.ndarray = gmm.predict(wine_simple[["alcohol", "color_intensity"]])

plt.figure(figsize=(7,5))
plt.scatter(wine_simple[["alcohol"]], wine_simple[["color_intensity"]], c=predictions, cmap="tab10", s=30, alpha=0.8)
plt.xlabel("alcohol"); plt.ylabel("color_intensity")
plt.title("GMM Cluster Assignments for Simplified Wine Dataset")
means: np.ndarray = gmm.means_
plt.scatter(means[:,0], means[:,1], c="red", marker="x", s=100)
x: np.ndarray = np.linspace(wine_simple["alcohol"].min()-1, wine_simple["alcohol"].max()+1, 200)
y: np.ndarray = np.linspace(wine_simple["color_intensity"].min()-1, wine_simple["color_intensity"].max()+1, 200)
X: np.ndarray
Y: np.ndarray
X, Y = np.meshgrid(x, y)
grid: np.ndarray = np.column_stack([X.ravel(), Y.ravel()])
Z: np.ndarray = np.exp(gmm.score_samples(grid)).reshape(X.shape)
plt.contour(X, Y, Z, levels=10, cmap="viridis", linewidths=1)
plt.show()