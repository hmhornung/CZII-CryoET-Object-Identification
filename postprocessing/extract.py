import torch
import numpy as np
def extract_points_from_target(target: torch.Tensor):
    target = np.array(target, dtype=np.int16)
    
    label_points = {label: [] for label in range(1, 7)}  # Initialize a dict for each label

    points = np.array(target.nonzero(), dtype=np.int16).T
    for point in points:
        label_points[target[tuple(point)]].append(point)
    return label_points

def extract_points_from_prediction(prediction, threshold = 0.9):
    prediction = prediction.detach().numpy()
    
    labels = range(1,7)
    
    label_points = {label: [] for label in labels}  # Initialize a dict for each label

    for label in labels:
        channel = prediction[label]
        points = np.array(np.where(channel > threshold)).T
        for p in range(points.shape[0]):
            label_points[label].append(points[p])

    return label_points

# Example Usage:
# target = torch.randint(0, 8, (2, 104, 104, 104))  # A random target with labels between 0 and 7
# pred = torch.rand(2, 7, 104, 104, 104)  # A random prediction with confidence scores for 7 labels

# Extract ground truth points
# ground_truth_points = extract_points_from_target(target)

# Set a threshold for predicted confidence (e.g., 0.5)
# threshold = 0.5
# predicted_points = extract_points_from_predictions(pred, threshold)