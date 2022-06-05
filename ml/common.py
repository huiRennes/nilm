"""
Various utilities and common modules.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128, },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128},
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512},
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536},
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000}
    }

def find_test_filename(test_dir, appliance, test_type) -> str:
    """TBA"""
    for filename in os.listdir(os.path.join(test_dir, appliance)):
        if test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break
    return test_filename

def load_dataset(file_name, crop=None) -> np.array:
    """Load CSV file, convert to np and return mains and appliance samples."""
    df = pd.read_csv(file_name, nrows=crop)

    df_np = np.array(df, dtype=np.float32)

    return df_np[:, 0], df_np[:, 1]

class WindowGenerator(tf.keras.utils.Sequence):
    """ Generates windowed timeseries samples and targets as a Keras Sequence.

    This is a subclass of a Keras Sequence class. 
    
    Attributes:
        dataset: input samples, targets timeseries data.
        batch_size: mini batch size used in training model.
        window_length: number of samples in a window of timeseries data.
        train: if True returns samples and targets else just samples.
        shuffle: if True shuffles dataset initially and every epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size=1000,
        window_length=599,
        train=True,
        shuffle=True) -> None:
        """Inits WindowGenerator."""

        self.X, self.y = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_length = window_length
        self.train = train

        # Total number of samples in dataset.
        self.total_samples=self.X.size

        # Number of samples from end of window to center.
        self.offset = int(0.5 * (window_length - 1.0))

        # Number of input samples adjusted for windowing.
        # This prevents partial window generation.
        self.num_samples = self.total_samples - 2 * self.offset

        # Indices of adjusted input sample array.
        self.indices = np.arange(self.num_samples)

        self.rng = np.random.default_rng()

        # Initial shuffle. 
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def on_epoch_end(self) -> None:
        """Shuffle at end of each epoch.""" 
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        """Returns number batches in an epoch."""
        return(int(np.ceil(self.num_samples / self.batch_size)))

    def __getitem__(self, index) -> np.ndarray:
        """Returns windowed samples and targets."""
        # Row indices for current batch. 
        rows = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Create a batch of windowed samples.
        samples = np.array(
            [self.X[row:row + 2 * self.offset + 1] for row in rows])

        # Reshape samples to match model's input tensor format.
        # Starting shape = (batch_size, window_length)
        # Desired shape = (batch_size, 1, window_length)
        samples = samples[:, np.newaxis, :]

        if self.train:
            # Create batch of single point targets offset from window start.
            targets = np.array([self.y[row + self.offset] for row in rows])
            return samples, targets
        else:
            # Return only samples if in test mode.
            return samples