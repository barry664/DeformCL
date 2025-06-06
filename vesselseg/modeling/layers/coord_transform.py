""" Transformation of image/mesh coordinates. """

def normalize_vertices(vertices, shape, ori_type="absolute"):
    if ori_type == 'absolute':
        new_verts = 2 * (vertices / (shape - 1) - 0.5)
    else:
        raise ValueError(f"Unknown ori_type: {ori_type}.")

    return new_verts

def unnormalize_vertices(vertices, shape, ori_type="absolute"):
    if ori_type == 'absolute':
        new_verts = (0.5 * vertices + 0.5) * (shape - 1)
    else:
        raise ValueError(f"Unknown ori_type: {ori_type}.")
    return new_verts
