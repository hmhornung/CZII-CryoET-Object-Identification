import numpy as np
import zarr
import os
import monai
import torch
import load
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandSpatialCropd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    RandRotated,
    SqueezeDimd
)

# Will probably add scale prob / range
aug_params = {
    "patch_size": (100,100,100),
    "final_size":   (100,100,100),
    "flip_prob":  0.5,
    "rot_prob":   1.0,
    "rot_range":  np.pi / 2
}

radii = [ 6,
          6,
          9,
          15,
          13,
          14 ]

def random_augmentation(volume, 
                        mask,
                        points = None,
                        num_samples=10, 
                        aug_params=aug_params,
                        save=False,
                        dest=None,
                        filename=None,
                        mask_type="cont"):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.

    Parameters:
        volume (np.ndarray): source 3D volume (shape: (C) x D x H x W). Channel optional
        mask (np.ndarray): source 3D mask (shape: D x H x W).
        num_samples (int): Number of augmented samples to generate.
        aug_params (dict): parameters for augmentation {patch_size, final_size, flip_prob, rot_prob, rot_range(radians)}

    Returns:
        list: Augmented (volume, mask) pairs.
    """
    if mask_type == "cont":
        mask_transform = 'bilinear'
    else:
        mask_transform = 'nearest'
        
    if save: print(f"Generating {filename} samples")
    
    if len(volume.shape) == 3:  volume = np.expand_dims(volume, axis=0)
    if len(mask.shape) == 3:  mask = np.expand_dims(mask, axis=0)

    sample_dict = {"source": volume, "target": mask}
    keys = ["source", "target"]
    mode = ['bilinear', mask_transform]
    #add points if necessary
    if points is not None:
        keys.append("points")
        if len(points.shape) == 3:  points = np.expand_dims(points, axis=0)
        sample_dict["points"] = points
        mode.append('nearest')

    augment = Compose([
        RandSpatialCropd(
            keys=keys, 
            roi_size=aug_params["patch_size"], 
            random_center=True, 
            random_size=False
        ),
        NormalizeIntensityd(
            keys="source"
        ),
        RandFlipd(
                keys=keys, 
                spatial_axis=[0, 1, 2], 
                prob=aug_params["flip_prob"]
        ),
        RandRotated(
            keys=keys, 
            range_x=aug_params["rot_range"], 
            range_y=aug_params["rot_range"], 
            range_z=aug_params["rot_range"], 
            prob=aug_params["rot_prob"],  
            keep_size=True,
            padding_mode='zeros',
            mode=mode
        ),
        ResizeWithPadOrCropd(
            keys=keys, 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    
    augmented_samples = []
    
    
    for n in range(num_samples):
        sample = augment(sample_dict)
        
        # Add to list
        if save:
            np.save(os.path.join(dest, "source/", f"{filename}-{n}.npy"), sample["source"])
            np.save(os.path.join(dest, "target/", f"{filename}-{n}.npy"), sample["target"])
            if points is not None:
                np.save(os.path.join(dest, "points/", f"{filename}-{n}.npy"), sample["points"])
            checkpts = [0.25,0.5,0.75,0.99]
            for i in checkpts:
                if n == int(num_samples * i):
                    print(f"\t{int(i*100)}%")
        else:
            augmented_samples.append(sample)
            
    if save: print(f"{filename} samples saved\n")
    
    return augmented_samples

def random_augmentation_gpu(
    volume,
    mask,
    points=None,
    num_samples=10,
    aug_params=None,
    save=False,
    dest=None,
    filename=None,
    mask_type="cont",
    device="cuda"
):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.

    Parameters:
        volume (np.ndarray): source 3D volume (shape: (C) x D x H x W). Channel optional.
        mask (np.ndarray): source 3D mask (shape: D x H x W).
        points (np.ndarray): optional 3D points (shape: N x 3 or C x N x 3).
        num_samples (int): number of augmented samples to generate.
        aug_params (dict): parameters for augmentation {patch_size, final_size, flip_prob, rot_prob, rot_range(radians)}.
        save (bool): whether to save augmented samples to disk.
        dest (str): destination directory for saved files.
        filename (str): base filename for saved files.
        mask_type (str): "cont" for continuous, "discrete" for discrete mask.
        device (str): device to use ("cuda" or "cpu").

    Returns:
        list: Augmented (volume, mask) pairs.
    """
    if mask_type == "cont":
        mask_transform = 'bilinear'
    else:
        mask_transform = 'nearest'
        
    if save: print(f"Generating {filename} samples")

    if len(volume.shape) == 3: volume = np.expand_dims(volume, axis=0)
    if len(mask.shape) == 3: mask = np.expand_dims(mask, axis=0)

    # Convert to tensors and move to the specified device
    volume = torch.tensor(volume, dtype=torch.float32, device=device)
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    sample_dict = {"source": volume, "target": mask}

    keys = ["source", "target"]
    mode = ['bilinear', mask_transform]

    if points is not None:
        if len(points.shape) == 3: points = np.expand_dims(points, axis=0)
        points = torch.tensor(points, dtype=torch.float32, device=device)
        sample_dict["points"] = points
        keys.append("points")
        mode.append('nearest')

    augment = Compose([
        RandSpatialCropd(
            keys=keys, 
            roi_size=aug_params["patch_size"], 
            random_center=True, 
            random_size=False
        ),
        NormalizeIntensityd(keys="source"),
        RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=aug_params["flip_prob"]),
        RandRotated(
            keys=keys, 
            range_x=aug_params["rot_range"], 
            range_y=aug_params["rot_range"], 
            range_z=aug_params["rot_range"], 
            prob=aug_params["rot_prob"],  
            keep_size=True,
            padding_mode='zeros',
            mode=mode
        ),
        ResizeWithPadOrCropd(
            keys=keys, 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    
    augmented_samples = []

    for n in range(num_samples):
        sample = augment(sample_dict)
        
        if save:
            # Save outputs as numpy arrays (transfer back to CPU first)
            np.save(os.path.join(dest, "source/", f"{filename}-{n}.npy"), sample["source"].cpu().numpy())
            np.save(os.path.join(dest, "target/", f"{filename}-{n}.npy"), sample["target"].cpu().numpy())
            if points is not None:
                np.save(os.path.join(dest, "points/", f"{filename}-{n}.npy"), sample["points"].cpu().numpy())
            checkpts = [0.25, 0.5, 0.75, 0.99]
            for i in checkpts:
                if n == int(num_samples * i):
                    print(f"\t{int(i*100)}%")
        else:
            augmented_samples.append(sample)
            
    if save: print(f"{filename} samples saved\n")
    
    return augmented_samples

def random_augmentation_heatmap_gpu(
    volume,
    points,
    num_samples=10,
    aug_params=None,
    save=False,
    dest=None,
    filename=None,
    mask_type="cont",
    device="cuda",
    sigma_factor=2,
    save_as_pts = False
):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.

    Parameters:
        volume (np.ndarray): source 3D volume (shape: (C) x D x H x W). Channel optional.
        mask (np.ndarray): source 3D mask (shape: D x H x W).
        points (np.ndarray): optional 3D points (shape: N x 3 or C x N x 3).
        num_samples (int): number of augmented samples to generate.
        aug_params (dict): parameters for augmentation {patch_size, final_size, flip_prob, rot_prob, rot_range(radians)}.
        save (bool): whether to save augmented samples to disk.
        dest (str): destination directory for saved files.
        filename (str): base filename for saved files.
        mask_type (str): "cont" for continuous, "discrete" for discrete mask.
        device (str): device to use ("cuda" or "cpu").

    Returns:
        list: Augmented (volume, mask) pairs.
    """
        
    if save: print(f"Generating {filename} samples")

    if len(volume.shape) == 3: volume = np.expand_dims(volume, axis=0)
    if len(points.shape) == 3: points = np.expand_dims(points, axis=0)

    # Convert to tensors and move to the specified device
    volume = torch.tensor(volume, dtype=torch.float32, device=device)
    points = torch.tensor(points, dtype=torch.int32, device=device)
    sample_dict = {"source": volume, "target": points}

    keys = ["source", "target"]
    mode = ['bilinear', 'nearest']

    augment = Compose([
        RandSpatialCropd(
            keys=keys, 
            roi_size=aug_params["patch_size"], 
            random_center=True, 
            random_size=False
        ),
        NormalizeIntensityd(keys="source"),
        RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=aug_params["flip_prob"]),
        RandRotated(
            keys=keys, 
            range_x=aug_params["rot_range"], 
            range_y=aug_params["rot_range"], 
            range_z=aug_params["rot_range"], 
            prob=aug_params["rot_prob"],  
            keep_size=True,
            padding_mode='zeros',
            mode=mode
        ),
        ResizeWithPadOrCropd(
            keys=keys, 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    augmented_samples = []

    for n in range(num_samples):
        sample = augment(sample_dict)
        sample["source"] = sample["source"].cpu().numpy()
        sample["target"] = sample["target"].cpu().numpy().astype(np.int32)
        if not save_as_pts: sample["target"] = load.create_gaussian_heatmap(sample["target"].squeeze(), radii=radii, sigma_factor=sigma_factor)
        
        if save:
            # Save outputs as numpy arrays (transfer back to CPU first)
            np.save(os.path.join(dest, "source/", f"{filename}-{n}.npy"), sample["source"])
            np.save(os.path.join(dest, "target/", f"{filename}-{n}.npy"), sample["target"])
            checkpts = [0.25, 0.5, 0.75, 0.99]
            for i in checkpts:
                if n == int(num_samples * i):
                    print(f"\t{int(i*100)}%")
        else:
            augmented_samples.append(sample)
            
    if save: print(f"{filename} samples saved\n")
    
    return augmented_samples