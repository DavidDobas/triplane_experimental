# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random
from .augmentations import AddGaussianNoise, ElasticTransform
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _file_ext(self, path):
        """Returns the lowercase file extension of the given path."""
        return os.path.splitext(path)[1].lower()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        # Convert NumPy array to PIL Image
        if image.shape[0] == 1:
            pil_mode = 'L'  # Grayscale
            image = image.squeeze(0)  # Remove channel dimension
        else:
            pil_mode = 'RGB'
            image = image.transpose(1, 2, 0)  # CHW to HWC

        pil_image = PIL.Image.fromarray(image, mode=pil_mode)

        # Apply transformations
        if hasattr(self, 'transform') and self.transform is not None:
            image = self.transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        return image, self.get_label(idx)

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        augment         = False, # Enable data augmentation
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        # Define augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=(0, 0))], p=0.0),  # No rotation
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))],
                    p=0.5
                ),
                # No RandomHorizontalFlip as per requirement
                transforms.RandomResizedCrop(
                    size=raw_shape[2:], scale=(0.9, 1.0), ratio=(1.0, 1.0)
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                transforms.ToTensor(),  # Convert PIL Image to Tensor at the end
                AddGaussianNoise(mean=0., std=0.02),
                ElasticTransform(alpha=1, sigma=50),  # Ensure this is tensor-compatible
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read(), format='RGB')
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class PairwiseImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, size=None):
        """Create a pairwise dataset from a base ImageFolderDataset.
        
        Args:
            dataset: Base ImageFolderDataset where each item is (image, camera_params)
        """
        self.dataset = dataset
        self.n = len(dataset)
        
        # Pre-compute all possible pairs of indices
        self.pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        # If size is not None, randomly sample a subset of the pairs
        if size is not None:
            self.pairs = random.sample(self.pairs, size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        
        # Get items from base dataset
        img_i, cam_i = self.dataset[i]
        img_j, cam_j = self.dataset[j]
        
        # Return (image_i, camera_diff) as input, image_j as target
        # Adjust based on your model's requirements
        return (img_i, cam_i, cam_j), img_j

#----------------------------------------------------------------------------

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dir, pairwise_dataset_size=None, augment=False):
        """Create a combined dataset from multiple zip datasets in a directory.
        
        Args:
            datasets_dir: Directory containing zip datasets
            pairwise_dataset_size: If not None, sample this many pairs from each dataset
            augment: Enable data augmentation
        """
        self.datasets = []
        
        # Get all zip files in directory
        for filename in os.listdir(datasets_dir):
            if filename.endswith('.zip'):
                dataset_path = os.path.join(datasets_dir, filename)
                
                # Create ImageFolderDataset for this zip file with augmentation
                base_dataset = ImageFolderDataset(dataset_path, resolution=None, augment=augment)
                
                # Wrap in PairwiseImageDataset
                pairwise_dataset = PairwiseImageDataset(
                    base_dataset,
                    size=pairwise_dataset_size
                )
                
                self.datasets.append(pairwise_dataset)

        # Total length is sum of all dataset lengths
        self.length = sum(len(dataset) for dataset in self.datasets)
        
        # Create index mapping
        self.dataset_indices = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_len = len(dataset)
            self.dataset_indices.extend([(dataset_idx, i) for i in range(dataset_len)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx, item_idx = self.dataset_indices[idx]
        dataset = self.datasets[dataset_idx]
        return dataset[item_idx]