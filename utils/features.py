"""Feature extraction from time-series accelerometer data."""

import pandas as pd
import numpy as np


def extract_features(group):
    """Extract statistical features from time-series data for a single maneuver.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing time-series measurements for one maneuver
        with columns: measurement_x, measurement_y, measurement_z

    Returns
    -------
    pd.Series
        Feature vector with 46 extracted features
    """
    features = {}

    # Basic statistics for each measurement
    for col in ['measurement_x', 'measurement_y', 'measurement_z']:
        features[f'{col}_mean'] = float(group[col].mean())
        features[f'{col}_std'] = float(group[col].std() or 0)
        features[f'{col}_min'] = float(group[col].min())
        features[f'{col}_max'] = float(group[col].max())
        features[f'{col}_median'] = float(group[col].median())
        features[f'{col}_q25'] = float(group[col].quantile(0.25))
        features[f'{col}_q75'] = float(group[col].quantile(0.75))
        features[f'{col}_range'] = float(group[col].max() - group[col].min())
        features[f'{col}_skew'] = float(group[col].skew() or 0)
        features[f'{col}_kurtosis'] = float(group[col].kurtosis() or 0)

    # Magnitude of acceleration vector
    group['magnitude'] = np.sqrt(
        group['measurement_x']**2 +
        group['measurement_y']**2 +
        group['measurement_z']**2
    )
    features['magnitude_mean'] = float(group['magnitude'].mean())
    features['magnitude_std'] = float(group['magnitude'].std() or 0)
    features['magnitude_max'] = float(group['magnitude'].max())

    # Pairwise correlations (safe handling of NaN)
    try:
        corr_matrix = group[['measurement_x', 'measurement_y', 'measurement_z']].corr()
        features['corr_xy'] = float(corr_matrix.loc['measurement_x', 'measurement_y'] or 0)
        features['corr_xz'] = float(corr_matrix.loc['measurement_x', 'measurement_z'] or 0)
        features['corr_yz'] = float(corr_matrix.loc['measurement_y', 'measurement_z'] or 0)
    except Exception:
        features['corr_xy'] = 0.0
        features['corr_xz'] = 0.0
        features['corr_yz'] = 0.0

    # Rate of change (derivatives)
    for col in ['measurement_x', 'measurement_y', 'measurement_z']:
        diff = group[col].diff().dropna()
        if len(diff) > 0:
            features[f'{col}_diff_mean'] = float(diff.mean())
            features[f'{col}_diff_std'] = float(diff.std() or 0)
            features[f'{col}_diff_max'] = float(diff.abs().max())
        else:
            features[f'{col}_diff_mean'] = 0.0
            features[f'{col}_diff_std'] = 0.0
            features[f'{col}_diff_max'] = 0.0

    # Number of observations
    features['n_observations'] = float(len(group))

    return pd.Series(features)
