import os

import h5py
import numpy as np

import torch
import torch.utils.data
from torchvision.datasets.mnist import download_and_extract_archive

from typing import Any, Callable, List, Optional, Tuple


class EventsToDense(torch.nn.Module):
    """Convert spike times to a dense matrix of zeros and ones."""

    def __init__(self, time_step, n_time_steps=None, n_units=None):
        """Initialize the conversion of spike times to a dense matrix of zeros and ones.

        time_step    -- binning interval in seconds
        n_time_steps -- number of bins along time axis (calculate from data, if `size is None`)
        n_units      -- number of units
        """

        super().__init__()

        self._time_step = time_step
        self._n_time_steps = n_time_steps
        self._n_units = n_units

    def forward(self, x):
        """Convert spike times to dense matrix of zeros and ones.
        """

        bins = (x[:, 0] / self._time_step).long()
        units = x[:, 1].long()

        if self._n_time_steps is not None:
            n_time_steps = self._n_time_steps
            n_time_steps_tmp = max(n_time_steps, int(bins.max()) + 1)
        else:
            n_time_steps = n_time_steps_tmp = int(bins.max()) + 1
        if self._n_units is not None:
            n_units = self._n_units
        else:
            n_units = int(units.max() + 1)

        dense = torch.zeros((n_time_steps_tmp, n_units))
        dense[bins, units] = 1

        return dense[:n_time_steps, :]


class Threshold(torch.nn.Module):
    def __init__(self, threshold, margin=0):
        super().__init__()

        self.threshold = threshold
        self.margin = margin

    def forward(self, x):
        onset = np.min(np.where(np.sum(x.numpy(), axis=1) >= self.threshold)[0])
        onset -= self.margin
        onset = max(onset, 0)

        x = x[onset:, :]
        x = torch.nn.functional.pad(x, (0, 0, 0, onset), mode="constant", value=0)

        return x


class SHD(torch.utils.data.Dataset):
    resources = [
            ("https://compneuro.net/datasets/shd_test.h5.gz", "3062a80ec0c5719404d5b02e166543b1"),
            ("https://compneuro.net/datasets/shd_train.h5.gz", "d47c9825dee33347913e8ce0f2be08b0"),
            ]

    training_file = 'shd_train.h5'
    test_file = 'shd_test.h5'

    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = False,
            transform: Optional[Callable] = None
    ) -> None:
        super(SHD, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data = h5py.File(os.path.join(self.data_folder, data_file))

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.data_folder, self.training_file)) and
                os.path.exists(os.path.join(self.data_folder, self.test_file)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, labels) where labels is index of the label class.
        """

        spikes = torch.stack([
            torch.from_numpy(self.data["spikes/times"][index].astype(np.float)),
            torch.from_numpy(self.data["spikes/units"][index].astype(np.float))
            ]).T
        label = self.data["labels"][index]

        if self.transform is not None:
            spikes = self.transform(spikes)

        return spikes, int(label)

    def __len__(self) -> int:
        return self.data["labels"].size

    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def classes(self) -> str:
        return list(self.data["extra/keys"][:])

    def download(self):
        """Download and rescale the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.data_folder, filename=filename, md5=md5)
