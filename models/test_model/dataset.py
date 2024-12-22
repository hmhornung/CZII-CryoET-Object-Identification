import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py  # or zarr/hdf5 or any other format for your 3D arrays


class UNetDataset(Dataset):
    def __init__(self, path, train=False, val=False):
        """
        Args:
            file_paths (list): List of tuples, where each tuple contains 
                                ('source_filename', 'target_filename').
            transform (callable, optional): Optional transform to be applied
                                             on a sample.
            train_split (float): Proportion of data for training.
        """
        self.path = path
        self.samples = os.listdir(os.path.join(path, "source/"))
        if train:
            self.samples = [sample for sample in self.samples if not sample.startswith("TS_5_4")]
        if val:
            self.samples = [sample for sample in self.samples if sample.startswith("TS_5_4")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the file paths for source and target
        sample = self.samples[idx]

        # Load the 3D arrays (source and target)
        source = np.load(os.path.join(self.path, 'source', sample))
        target = np.load(os.path.join(self.path, 'target', sample))

        # Convert to tensors
        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int8)

        sample = {'source': source, 'target': target}

        return sample
        


# Collate function for padding and batching
def collate_fn(batch):
    """
    Custom collate function to handle 3D arrays and batch them together.
    """
    source = torch.stack([item['source'] for item in batch], dim=0).unsqueeze(1)
    target = torch.stack([item['target'] for item in batch], dim=0).unsqueeze(1)
    
    return {'src': source, 'tgt': target}


# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
