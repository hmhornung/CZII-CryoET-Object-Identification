import numpy as np
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandSpatialCropd,
    SpatialPadd,
    RandRotated
)

def random_augmentation(volume, mask, patch_size=(85, 85, 85), pad_size=(120, 120, 120), num_samples=10):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.

    Parameters:
        volume (np.ndarray): Input 3D volume (shape: (C) x D x H x W). Channel optional
        mask (np.ndarray): Input 3D mask (shape: D x H x W).
        patch_size (tuple): Size of the random crop.
        pad_size (tuple): Final size of the padded output.
        num_samples (int): Number of augmented samples to generate.

    Returns:
        list: Augmented (volume, mask) pairs.
    """
    
    if len(volume.shape) == 3:  volume = np.expand_dims(volume, axis=0)
    if len(mask.shape) == 3:  mask = np.expand_dims(mask, axis=0)
    
    sample_dict = {"input": volume, "target": mask}
    
    crop_transform = RandSpatialCropd(keys=["input","target"], roi_size=patch_size, random_center=True, random_size=False)
    
    reg_aug = Compose([
        NormalizeIntensityd(keys="input"),
        SpatialPadd(keys=["input","target"], spatial_size=pad_size, mode="constant")
    ])
    
    
    rand_aug = Compose([
            RandFlipd(keys=["input","target"], spatial_axis=[0, 1, 2], prob=0.5),
            RandRotated(keys=["input","target"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=1.0, keep_size=True, padding_mode='zeros')
        ])
    
    augmented_samples = []
    
    for n in range(num_samples):
        # Random Crop
        sample = crop_transform(sample_dict)

        # proportion of border intersecting targets (Temporary)
        # print(f'{n}: {get_border_density(sample["target"])}')
        
        # Normalize then zero-Pad
        sample = reg_aug(sample)

        # Random Flip and Rotation
        sample = rand_aug(sample)
        
        # Add to list
        augmented_samples.append(sample)
    return augmented_samples