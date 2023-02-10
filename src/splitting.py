from typing import Iterator, List, Tuple

import numpy as np



class _BaseCV:
    """Base class to perform cross validation.

    Implementations must define `_create_split_masks` or `_create_split_indices`.
    """
    def split(self, samples: np.array, groups: np.array = None) -> Iterator[Tuple[np.array]]:
        """Returns train, val and test indices.

        Args:
            samples (np.array): Numpy array with indices to split.
            groups (np.array, optional): Numpy array with groupd indices. Defaults to None.

        Yields:
            Iterator[Tuple[np.array]]: Train, val and test indices.
        """
        n_samples = samples.shape[0]
        indices = np.arange(n_samples)
        for val_mask, test_mask in self._create_split_masks(samples, groups):
            train_ids = indices[np.logical_and(np.logical_not(val_mask), np.logical_not(test_mask))]
            val_ids = indices[val_mask]
            test_ids = indices[test_mask]
            assert len(np.intersect1d(train_ids, test_ids)) == 0
            assert len(np.intersect1d(train_ids, val_ids)) == 0
            assert len(np.intersect1d(val_ids, test_ids)) == 0
            yield train_ids, val_ids, test_ids


    def _create_split_masks(self, samples: np.array, groups: np.array = None) -> Iterator[Tuple[np.array]]:
        """Returns masks for val and test sets.

        Args:
            samples (np.array): Numpy array with indices to split.
            groups (np.array, optional): Numpy array with groupd indices. Defaults to None.

        Yields:
            Iterator[Tuple[np.array]]: Train, val and test indices.
        """
        n_samples = samples.shape[0]
        for val_ids, test_ids in self._create_split_indices(samples, groups):
            val_mask = np.zeros(n_samples, dtype=bool)
            test_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_ids] = True
            test_mask[test_ids] = True
            yield val_mask, test_mask


    def _create_split_indices(self, samples: np.array, groups: np.array = None) -> Iterator[Tuple[np.array]]:
        """Splits the data and returns val and test indices.

        Args:
            samples (np.array): Numpy array with indices to split.
            groups (np.array, optional): Numpy array with groupd indices. Defaults to None.

        Yields:
            Iterator[Tuple[np.array]]: _description_
        """
        raise NotImplementedError

    def get_n_splits(self) -> int:
        """Returns number of splits"""
        raise NotImplementedError

    def get_splits(self) -> List[List[np.array]]:
        """Returns indices of all splits."""
        raise NotImplementedError


class MonteCarloCV(_BaseCV):
    """Monte Carlo Cross-Validation 

    Performs repeated 3-way (train, val, test) holdout validation. 
    There can be overlap between different splits of train, val and
    test splits due to random splitting.

    Args:
        val_size (float): Proportion of validation samples.
        test_size (float): Proportaion of test samples. 
        n_repeats (int, optional): Number of repetitions. Defaults to 10.
        shuffle (bool, optional): Whether to shuffle samples. Defaults to True.
        seed (int, optional): Random seed. Defaults to None.
    """
    def __init__(
        self, 
        val_size: float, 
        test_size: float, 
        n_repeats: int = 10, 
        shuffle: bool = True,
        seed: int = None) -> None:
        self.val_size = val_size
        self.test_size = test_size
        self.n_repeats = n_repeats
        self.shuffle = shuffle
        self.seed = seed 
    

    def split(self, samples, groups = None):
        for train, val, test in super().split(samples, groups):
            yield train, val, test


    def _create_split_indices(self, samples, groups = None):
        """Splits the data randomly into train, val and test split according to split sizes."""
        n_samples = samples.shape[0]
        n_val = np.ceil(n_samples * self.val_size).astype(int)
        n_test = np.ceil(n_samples * self.test_size).astype(int)

        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            indices = rng.permutation(indices, axis=0)

        for i in range(self.n_repeats):
            test_ids = rng.choice(indices, size=n_val+n_test, replace=False)
            val_ids = test_ids[:n_val]
            test_ids = test_ids[n_val:]
            yield val_ids, test_ids


    def get_n_splits(self):
        return self.n_repeats


    def get_splits(self):
        return self.splits



class KfoldCV(_BaseCV):
    """K-Fold Cross-Validation

    Performs k-fold cross-validation using a 3-way (train, val, test) holdout method. 
    The val and test folds do not overlap between runs. 
        
    Args:
        n_folds (int, optional): Number of folds. Defaults to 5.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        seed (int, optional): Random seed. Defaults to None.
    """
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = False,
        seed: int = None) -> None:
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.seed = seed


    def split(self, samples, groups = None):
        n_samples = samples.shape[0]

        if self.n_folds > n_samples:
            raise ValueError('Number of folds n_folds={} cannot be larger than number \
                            of samples n_samples={}.'.format(self.n_folds, n_samples))

        for train, val, test in super().split(samples, groups):
            yield train, val, test


    def _create_split_indices(self, samples, groups):
        """Splits the data randomly into k-folds and creates train, val and test splits accordingly for each run."""
        n_samples = samples.shape[0]

        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)

        if self.shuffle:
            indices = rng.permutation(indices, axis=0)

        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds, dtype=int)
        fold_sizes[: n_samples % self.n_folds] += 1

        current = 0
        for fold_size in fold_sizes:
            val_start, val_stop = current, current + fold_size
            val_ids = indices[val_start:val_stop]
            if val_stop == n_samples:
                val_stop = 0
            test_start, test_stop = val_stop, val_stop + fold_size
            test_ids = indices[test_start:test_stop]
            yield val_ids, test_ids 
            current = val_stop

    
    def get_n_splits(self):
        return self.n_folds


    def get_splits(self):
        return self.splits















        



        

        

