"""
Derived from:
https://www.kaggle.com/code/metric/czi-cryoet-84969
"""

import numpy as np
import pandas as pd
import torch

from scipy.spatial import KDTree

def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn

def continuous_precision_recall(target, output, dims=(0,2,3,4)):
    score = output / (target + 1e-8)
    
    fp_idx = score >= 1.0
    fn_idx = score <  1.0
    
    tp, fp, fn = [torch.zeros(target.shape) for _ in range(3)]
    
    tp[fp_idx] = 1 / score[fp_idx]
    fp[fp_idx] = (score[fp_idx] - 1) / score[fp_idx]
    
    tp[fn_idx] = score[fn_idx]
    fn[fn_idx] = 1 - score[fn_idx]
    
    tp = tp.sum(dim=dims)
    fp = fp.sum(dim=dims)
    fn = fn.sum(dim=dims)
    
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    
    return precision, recall
    
    