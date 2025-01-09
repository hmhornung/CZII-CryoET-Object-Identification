phere[distances > radius] = 0  # Set values outside the radius to 0
        return sphere

    # Initialize the output array
    heatmaps = np.zeros((labels, *volume_shape), dtype=np.float32)

    # Generate heatmaps for each label
    for label_idx, label_points in enumerate(points):
        radius = radii[label_idx]
        for point in label_points:
            x, y, z = point
            # Add the sphere to the corresponding label's channel
            heatmaps[label_idx] += gaussian_sphere((z, y, x), radius, volume_shape)

    return heatmaps