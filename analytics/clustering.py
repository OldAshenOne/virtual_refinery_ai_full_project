
import pandas as pd
import numpy as np
def _simple_kmeans(X, k=3, iters=100, seed=42):
    np.random.seed(seed)
    perc = np.linspace(0, 100, k+2)[1:-1]
    idx = [np.argmin(np.sum((X - np.percentile(X, p, axis=0))**2, axis=1)) for p in perc]
    centers = X[idx, :].astype(float)
    for _ in range(iters):
        dists = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.vstack([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels
def cluster_units(df_units: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    use_cols = ["total_input_kgco2e", "total_output_qty"]
    X = df_units[use_cols].fillna(0.0).values.astype(float)
    labels = _simple_kmeans(X, k=n_clusters)
    out = df_units.copy()
    out["cluster"] = labels
    return out
