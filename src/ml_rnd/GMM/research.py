import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import pandas as pd


def hungarian_accuracy(predictions: np.ndarray, target: np.ndarray, n_clusters: int) -> float:
    """Calculate clustering accuracy using Hungarian algorithm for optimal label matching.
    
    Args:
        predictions: Predicted cluster labels
        target: True class labels
        n_clusters: Number of clusters
        
    Returns:
        Accuracy score after optimal cluster-to-class mapping
    """
    confusion_matrix: np.ndarray = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            confusion_matrix[i, j] = np.sum((predictions == i) & (target == j))
    
    # Find optimal mapping using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)  # Negative for maximization
    # Calculate accuracy with optimal mapping
    cluster_to_class: dict = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    matched_correct: int = sum(confusion_matrix[i, cluster_to_class[i]] for i in range(n_clusters))
    accuracy: float = matched_correct / len(predictions)
    
    return accuracy


wine: pd.DataFrame = datasets.load_wine(as_frame=True).frame
target: np.ndarray = wine["target"].values
features: pd.DataFrame = wine.drop(columns=["target"])

# Fit GMM on all features
gmm: GaussianMixture = GaussianMixture(n_components=3)
gmm.fit(features)
predictions: np.ndarray = gmm.predict(features)

# Use PCA to reduce to 2D for visualization
pca: PCA = PCA(n_components=2)
wine_pca: np.ndarray = pca.fit_transform(features)

# Calculate Hungarian algorithm matching between clusters and classes
accuracy: float = hungarian_accuracy(predictions, target, n_clusters=3)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First subplot: GMM predictions
for cluster in np.unique(predictions):
    mask = predictions == cluster
    ax1.scatter(wine_pca[mask, 0], wine_pca[mask, 1], 
                label=f"Cluster {cluster}", s=30, alpha=0.8)
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
ax1.set_title(f"GMM Cluster Assignments (All Features), Hungarian Accuracy: {accuracy:.2%}")
ax1.legend()

# Second subplot: actual target classes
for target_class in np.unique(target):
    mask = target == target_class
    ax2.scatter(wine_pca[mask, 0], wine_pca[mask, 1], 
                label=f"Class {target_class}", s=30, alpha=0.8)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
ax2.set_title("Actual Wine Classes (All Features)")
ax2.legend()

plt.tight_layout()
plt.show()