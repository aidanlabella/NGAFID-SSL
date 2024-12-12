import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def mask_transform(X,masking_ratio=0.15, mean_mask_length=3, mode='separate', distribution='geometric', exclude_feats=None):
    mask = noise_mask(X, masking_ratio, mean_mask_length, mode, distribution, exclude_feats)  # (seq_length, feat_dim) boolean array
    X = torch.from_numpy(X)
    mask = torch.from_numpy(mask)
    transformed_X = X * mask
    return X,transformed_X

# from: https://github.dev/gzerveas/mvts_transformer
def noise_mask(X, masking_ratio, lm, mode, distribution, exclude_feats):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    return mask

# from: https://github.dev/gzerveas/mvts_transformer
def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def noise_transform(X, loc = 0, range = (0.1, 0.5)):
    # normalize X column wise
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    deviation = np.random.uniform(range[0], range[1])
    noise = np.random.normal(loc, deviation, X.shape)
    X_transformed = X_standardized + noise
    return X_standardized, X_transformed

class TransformationDataset(Dataset):
    def __init__(self, flight_id_topath, transformation_method=noise_transform):
        self.flight_id_topath = flight_id_topath.reset_index()
        self.transformation = transformation_method
    
    def __len__(self):
        return len(self.flight_id_topath)
    
    def __getitem__(self, index):
        path = self.flight_id_topath.loc[index, "file_path"]
        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN '])
        flight_T = flight.T
        flight_T.ffill(inplace= True, axis=0)
        flight_T.bfill(inplace= True, axis=0)
        flight = flight_T.T
        flight = flight.to_numpy()
        flight, flight_transformed = self.transformation(flight)
        flight_transformed = torch.tensor(flight_transformed, dtype=torch.float32)
        flight = torch.tensor(flight, dtype=torch.float32)
        pos_pair = (flight.unsqueeze(dim=0), flight_transformed.unsqueeze(dim=0))
        return pos_pair

