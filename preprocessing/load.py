import copick
import os
import json
import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
def get_root(cnfg_path="H:/Projects/Kaggle/CZII-CryoET-Object-Identification/preprocessing/copick_config.json"):
    root = copick.from_file(cnfg_path)
    return root

def get_picks_dict(root, particles_only=True):
    pick_dict = {}
    for pick in root.pickable_objects:
        pick_dict[pick.name] = {
            'is_particle': pick.is_particle,
            'label': pick.label,
            'color': pick.color,
            'pdb_id': pick.pdb_id,
            'radius': pick.radius,
            'threshold': pick.map_threshold
        }
    if particles_only:
        return {k : v for k, v in pick_dict.items() if v.get('is_particle')}
    return pick_dict

# Function to load CryoET OME-Zarr data at all resolutions
def get_run_volume_picks(root, run='TS_5_4', level=0, particles=None):
    
    zarr_path = os.path.join(root.root_static, f"ExperimentRuns/{run}/VoxelSpacing10.000/denoised.zarr")
    picks_path = os.path.join(root.root_overlay, f"ExperimentRuns/{run}/Picks")
    
    if particles is None: 
        pick_dict = get_picks_dict(root)
        particles = [k for k, v in pick_dict.items() if v.get('is_particle')]

    # Open the OME-Zarr dataset
    store = zarr.DirectoryStore(zarr_path)
    zarrs = zarr.open(store, mode='r')
    
    level_info = zarrs.attrs['multiscales'][0]['datasets'][level]
    
    scales = np.array(level_info['coordinateTransformations'][0]['scale'])
    
    # Swap scales since data has x & z switched
    scales[[0, 2]] = scales[[2, 0]]
    
    path = level_info["path"]
    
    volume = np.array(zarrs[path][:])
    
    # Load ground truth JSONs (particle coordinates)
    particle_coords = {}
    for particle in particles:
        json_path = os.path.join(picks_path, f"{particle}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                # Get json file
                json_data = json.load(f)
                pick_data = json_data['points']
                
                # Get all picks for specific particle type
                picks = []
                for pick in pick_data:
                    coords = pick['location']
                    # Swap the x and z coords since data has them swapped
                    picks.append(np.array([coords['z'], coords['y'],  coords['x']]) / scales)
                particle_coords[particle] = np.array(picks)
                
                    
        else:
            print(f"Ground truth file for {particle} not found.")
    
    return volume, particle_coords, scales

import numpy as np

def create_sphere(volume, center, radius, value):
    """
    Creates a sphere in a 3D volume array.

    Parameters:
    volume: 3D numpy array
    center: (x, y, z) coordinates of the sphere's center
    radius: radius of the sphere
    value: value to assign to the sphere's voxels

    Returns:
    3D numpy array with the sphere
    """

    x, y, z = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    volume[dist_sq <= radius ** 2] = value
    return volume

# i = 8

# volume = np.zeros((16,16,16))
# print(volume[i])

# volume = create_sphere(volume, (7,7,7), 7)

# print(volume[i])

def get_picks_mask(shape, pick_dict, coords, scale, pts = False, use_dscrt=True):
    mask = np.zeros(shape, dtype=np.int16)
    
    for particle in pick_dict:
        # print(pick_dict[particle]['radius'])
        rad = int(np.ceil(pick_dict[particle]['radius'] / scale))
        if use_dscrt: val = pick_dict[particle]['label']
        else: val = 1.0
        points = coords[particle]
        for idx in range(points.shape[0]):
            point = points[idx]
            if pts:
                mask[int(point[0]), int(point[1]), int(point[2])] = val
            else:
                mask = create_sphere(mask, point, rad, val)
    
    return mask

def create_exponential_heatmap(labels, volume_shape, points, radii):
    heatmap = np.zeros((labels, *volume_shape), dtype=np.float32)
    zz, yy, xx = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing="ij"
    )

    for label in range(labels):
        radius = radii[label]  # Get the radius for this label
        for point in points[label]:
            distances = np.sqrt((zz - point[0])**2 + (yy - point[1])**2 + (xx - point[2])**2)
            mask = distances <= radius
            decay = np.exp(-distances / radius) - np.exp(-1)  # Normalize decay to start near 0
            decay /= 1 - np.exp(-1)  # Scale to make center 1.0
            heatmap[label][mask] = np.maximum(heatmap[label][mask], decay[mask])
            
    background_channel = np.ones(volume_shape, dtype=np.float32)
    for i in range(heatmap.shape[0]):
        background_channel = background_channel - heatmap[i]
    background_channel[background_channel < 0.0] = 0.0
    return np.concatenate((np.expand_dims(background_channel, axis=0), heatmap), axis=0)

def create_exponential_heatmap_gpu(labels, volume_shape, points, radii, device="cuda"):
    heatmap = torch.zeros((labels, *volume_shape), dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(volume_shape[0], device=device),
        torch.arange(volume_shape[1], device=device),
        torch.arange(volume_shape[2], device=device),
        indexing="ij"
    )

    for label in range(labels):
        radius = radii[label]  # Get the radius for this label
        for point in points[label]:
            distances = torch.sqrt((zz - point[0])**2 + (yy - point[1])**2 + (xx - point[2])**2)
            mask = distances <= radius
            decay = torch.exp(-distances / radius) - torch.exp(torch.tensor(-1.0, device=device))
            decay /= 1 - torch.exp(torch.tensor(-1.0, device=device))  # Normalize
            heatmap[label][mask] = torch.maximum(heatmap[label][mask], decay[mask])

    background_channel = torch.ones(volume_shape, dtype=torch.float32, device=device)
    for i in range(heatmap.shape[0]):
        background_channel -= heatmap[i]
    background_channel[background_channel < 0.0] = 0.0
    
    return torch.cat((background_channel.unsqueeze(0), heatmap), dim=0)

def create_gaussian_heatmap(pt_mask, radii, sigma_factor = 1, n_class = 6):
    sigma = [r / 2 for r in radii]
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
    return heatmaps