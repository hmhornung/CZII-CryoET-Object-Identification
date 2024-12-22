import numpy as np
import monai
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandSpatialCropd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    RandRotated
)

# Will probably add scale prob / range
aug_params = {
    "patch_size": (100,100,100),
    "final_size":   (100,100,100),
    "flip_prob":  0.5,
    "rot_prob":   1.0,
    "rot_range":  np.pi / 2
}

def random_augmentation(volume, 
                        mask, 
                        num_samples=10, 
                        aug_params=aug_params,
                        save=False,
                        dest=None,
                        filename=None):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.

    Parameters:
        volume (np.ndarray): Input 3D volume (shape: (C) x D x H x W). Channel optional
        mask (np.ndarray): Input 3D mask (shape: D x H x W).
        num_samples (int): Number of augmented samples to generate.
        aug_params (dict): parameters for augmentation {patch_size, final_size, flip_prob, rot_prob, rot_range(radians)}

    Returns:
        list: Augmented (volume, mask) pairs.
    """
    
    if len(volume.shape) == 3:  volume = np.expand_dims(volume, axis=0)
    if len(mask.shape) == 3:  mask = np.expand_dims(mask, axis=0)
    
    sample_dict = {"input": volume, "target": mask}
    
    augment = Compose([
        RandSpatialCropd(
            keys=["input", "target"], 
            roi_size=aug_params["patch_size"], 
            random_center=True, 
            random_size=False
        ),
        NormalizeIntensityd(
            keys="input"
        ),
        RandFlipd(
                keys=["input","target"], 
                spatial_axis=[0, 1, 2], 
                prob=aug_params["flip_prob"]
        ),
        RandRotated(
            keys=["input","target"], 
            range_x=aug_params["rot_range"], 
            range_y=aug_params["rot_range"], 
            range_z=aug_params["rot_range"], 
            prob=aug_params["rot_prob"],  
            keep_size=True,
            padding_mode='zeros',
            mode=['bilinear', 'nearest']
        ),
        ResizeWithPadOrCropd(
            keys=["input","target"], 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    
    crop = RandSpatialCropd(
        keys=["input", "target"], 
        roi_size=aug_params["patch_size"], 
        random_center=True, 
        random_size=False
    )
    
    reg_aug = Compose([
        NormalizeIntensityd(
            keys="input"
        ),
        ResizeWithPadOrCropd(
            keys=["input","target"], 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    
    
    rand_aug = Compose([
            RandFlipd(
                keys=["input","target"], 
                spatial_axis=[0, 1, 2], 
                prob=aug_params["flip_prob"]
            ),
            RandRotated(
                keys=["input","target"], 
                range_x=aug_params["rot_range"], 
                range_y=aug_params["rot_range"], 
                range_z=aug_params["rot_range"], 
                prob=aug_params["rot_prob"],  
                keep_size=True,
                padding_mode='zeros',
                mode=['bilinear', 'nearest']
            )
        ])
    
    augmented_samples = []
    
    for n in range(num_samples):
        sample = augment(sample_dict)
        
        # Add to list
        augmented_samples.append(sample)
    
    return augmented_samples

def get_border_density(vol: np.ndarray):
    return np.mean(
                np.concatenate([
                    vol[0, :, :],  # Top slice
                    vol[-1, :, :],  # Bottom slice
                    vol[:, 0, :],  # Front slice
                    vol[:, -1, :],  # Back slice
                    vol[:, :, 0],  # Left slice
                    vol[:, :, -1],  # Right slice
                ])
            )