
import pandas as pd
import numpy as np
def detect_outliers(df_units: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    x = df_units['total_input_kgco2e'].astype(float).values
    mu = np.mean(x); sigma = np.std(x) if np.std(x)>0 else 1.0
    z = (x - mu) / sigma
    out = df_units.copy()
    out['z_energy'] = z
    out['anomaly'] = np.where(np.abs(z) > z_thresh, -1, 1)
    return out
