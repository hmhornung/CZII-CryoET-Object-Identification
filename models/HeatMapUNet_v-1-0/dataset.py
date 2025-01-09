import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py  # or zarr/hdf5 or any other format for your 3D arrays


class UNetDataset(Dataset):
    def __init__(self, path, in_mem=True, train=False, val=False, fold=0):
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
        sample_names = os.listdir(os.path.join(path, "source/"))
        val_experiment = ['TS_5_4', 'TS_6_4', 'TS_6_6', 'TS_69_2', 'TS_73_6', 'TS_86_3', 'TS_99_9'][fold]
        if train:
            sample_names = [i for i in self.samples if not i.startswith(val_experiment)]
        if val:
            sample_names = [i for i in self.samples if i.startswith(val_experiment)]
        
        self.samples = []
        if in_mem:
            for name in sample_names:
                source = np.load(os.path.join(self.path, 'source', name))
                target = np.load(os.path.join(self.path, 'target', name))
                self.samples.append(
                    {
                    "source": torch.tensor(source, dtype=torch.float32),
                    "target": torch.tensor(target, dtype=torch.float32)
                    }
                )
        else: self.samples = sample_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.in_mem:
            return self.samples[idx]
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
    source = torch.stack([item['source'] for item in batch], dim=0)
    target = torch.stack([item['target'] for item in batch], dim=0)
    
    return {'src': source, 'tgt': target}


# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
