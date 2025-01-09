import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py  # or zarr/hdf5 or any other format for your 3D arrays
from scipy.ndimage import gaussian_filter

radii = [ 6,
          6,
          9,
          15,
          13,
          14 ]

class UNetDataset(Dataset):
    def __init__(self, path, in_mem=True, train=False, val=False, fold=0, sigma_factor=2):
        """
        Args:
            file_paths (list): List of tuples, where each tuple contains 
                                ('source_filename', 'target_filename').
            transform (callable, optional): Optional transform to be applied
                                             on a sample.
            train_split (float): Proportion of data for training.
        """
        self.path = path
        self.in_mem = in_mem
        self.sigma_factor = sigma_factor
        sample_names = os.listdir(os.path.join(path, "source/"))
        val_experiment = ['TS_5_4', 'TS_6_4', 'TS_6_6', 'TS_69_2', 'TS_73_6', 'TS_86_3', 'TS_99_9'][fold]
        if train:
            sample_filenames = [sample for sample in sample_names if not sample.startswith(val_experiment)]
        elif val:
            sample_filenames = [sample for sample in sample_names if sample.startswith(val_experiment)]
        else:
            sample_filenames = sample_names
        
        self.samples = []
        if in_mem:
            for name in sample_filenames:
                source = np.load(os.path.join(self.path, 'source', name))
                target = np.load(os.path.join(self.path, 'target', name))
                self.samples.append(
                    {
                    "source": source.astype(np.float32),
                    "target": target.astype(np.int8)
                    }
                )
        else: self.samples = sample_filenames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.in_mem:
            sample["target"] = np.squeeze(self.create_gaussian_heatmap(sample["target"]))
            
            sample["source"] = torch.tensor(sample["source"], dtype=torch.float32)
            sample["target"] = torch.tensor(sample["target"], dtype=torch.float32)
            return sample

        # Load the 3D arrays (source and target)
        source = np.load(os.path.join(self.path, 'source', sample))
        target = np.load(os.path.join(self.path, 'target', sample))

        # Convert to heatmap
        target = self.create_gaussian_heatmap(target)
        
        # Convert to tensors
        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        sample = {'source': source, 'target': target}

        return sample
    
    def create_gaussian_heatmap(self, pt_mask, n_class = 6):
        sigma = [r / self.sigma_factor for r in radii]
        heatmaps = np.zeros((n_class, *pt_mask.shape), dtype=np.float32)

        for class_id in range(1, n_class + 1):
            # Create a binary mask for the current class
            class_mask = (pt_mask == class_id).astype(np.float32)
            
            # Apply Gaussian filter to the binary mask
            heatmaps[class_id - 1] = gaussian_filter(class_mask, sigma=sigma[class_id - 1], axes=(0,1,2))
            
        heatmaps /= (heatmaps.max(axis=(1, 2, 3), keepdims=True) + 1e-8)
        background = 1 - heatmaps.sum(axis=0, keepdims=True)
        background[background < 0] = 0
        heatmaps = np.concatenate((background, heatmaps), axis=0)
        print(heatmaps.shape)
        return heatmaps
        


# Collate function for padding and batching
def collate_fn(batch):
    """
    Custom collate function to handle 3D arrays and batch them together.
    """
    source = torch.stack([item['source'] for item in batch], dim=0)
    target = torch.stack([item['target'] for item in batch], dim=0)
    print(target.shape)
    return {'src': source, 'tgt': target}


# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
