import numpy as np
import torch
import os

def _save_ply(path, verts, faces):
    """Save a mesh to an ASCII PLY file (triangles).

    Args:
        path (str): Output file path.
        verts (np.ndarray): (N,3) float vertex positions.
        faces (np.ndarray): (M,3) integer triangle indices.
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces, dtype=np.int32)
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def extract_mesh_from_sdf(sdf_model,
                          bounds=((-1., -1., -1.), (1., 1., 1.)),
                          resolution=256,
                          level=0.0,
                          device=None,
                          batch_size=65536,
                          save_path=None):
    """Extract a triangle mesh from an SDF model using marching cubes.

    The function samples the model on a dense 3D grid within `bounds`,
    runs marching cubes on the scalar field, and returns (verts, faces, normals, values).

    Args:
        sdf_model: a callable PyTorch module that returns (distances, features) given (N,3) input.
        bounds: ((x_min,y_min,z_min),(x_max,y_max,z_max)) defining the AABB to sample.
        resolution: number of samples per axis (int).
        level: iso-value to extract (default 0.0).
        device: torch device or None (will use model device or cpu).
        batch_size: evaluation chunk size for model inference.
        save_path: if provided, writes an ASCII PLY to this path.

    Returns:
        verts (np.ndarray [V,3]), faces (np.ndarray [F,3]), normals (np.ndarray [V,3]), values (np.ndarray [V])
    """
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise RuntimeError('skimage is required for marching cubes (pip install scikit-image)') from e

    # prepare device
    if device is None:
        try:
            # try to infer from model parameters
            device = next(sdf_model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    device = torch.device(device)

    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    zs = np.linspace(z_min, z_max, resolution)

    # create grid with indexing='ij' so shape = (nx, ny, nz)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    N = int(np.prod(grid.shape[:-1]))
    points = grid.reshape(-1, 3)

    sdf_vals = np.zeros((N,), dtype=np.float32)

    sdf_model_was_training = getattr(sdf_model, 'training', False)
    sdf_model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            pts = points[i:i+batch_size]
            t = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
            dists, _ = sdf_model(t)
            sdf_vals[i:i+len(dists)] = dists.cpu().numpy().astype(np.float32)

    # restore training state
    if sdf_model_was_training:
        sdf_model.train()

    volume = sdf_vals.reshape((len(xs), len(ys), len(zs)))

    # spacing for marching_cubes
    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])

    verts, faces, normals, values = marching_cubes(volume, level=level, spacing=spacing)

    # marching_cubes returns vertices in index-space offset by the minimal corner
    # but because we passed spacing, verts are already in world coords relative to origin (0,0,0)
    # we need to shift by the minimal coordinate
    origin = np.array([x_min, y_min, z_min], dtype=np.float32)
    verts = verts + origin

    if save_path is not None:
        _save_ply(save_path, verts, faces)

    return verts, faces, normals, values


def _choose_checkpoint(path_or_dir):
    import os
    if os.path.isdir(path_or_dir):
        # pick newest .pth in dir
        pths = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith('.pth')]
        if not pths:
            raise FileNotFoundError(f'No .pth files found in directory: {path_or_dir}')
        pths = sorted(pths, key=lambda p: os.path.getmtime(p))
        return pths[-1]
    elif os.path.isfile(path_or_dir):
        return path_or_dir
    else:
        raise FileNotFoundError(f'Checkpoint path not found: {path_or_dir}')


def _load_sdf_model_from_checkpoint(checkpoint_path, feature_dim=8, device=None):
    import torch
    from mlp import sdf_freq_mlp
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)

    # create model (must match training feature_dim)
    sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim).to(device)

    # determine state dict
    if isinstance(ckpt, dict) and 'sdf_model_state_dict' in ckpt:
        state = ckpt['sdf_model_state_dict']
    else:
        state = ckpt

    try:
        sdf_model.load_state_dict(state)
    except Exception:
        # attempt non-strict load to allow missing keys (informative)
        sdf_model.load_state_dict(state, strict=False)

    return sdf_model, ckpt


def parse_cmdline():
    import argparse
    parser = argparse.ArgumentParser(description='Extract mesh from trained SDF model (.pth)')
    parser.add_argument('checkpoint', help='Path to .pth file or directory containing .pth files')
    parser.add_argument('--feature_dim', type=int, default=8, help='Feature dimension used for the SDF model')
    parser.add_argument('--bounds', type=float, nargs=6, default=[-1, -1, -1, 1, 1, 1],
                        help='Bounding box as x_min y_min z_min x_max y_max z_max')
    parser.add_argument('--resolution', type=int, default=256, help='Grid resolution per axis')
    parser.add_argument('--level', type=float, default=0.0, help='Iso-surface level to extract')
    parser.add_argument('--batch_size', type=int, default=65536, help='Evaluation batch size')
    parser.add_argument('--device', type=str, default=None, help='Torch device (e.g., cpu or cuda:0). Auto if omitted')
    parser.add_argument('--out', type=str, default=None, help='Output PLY path (defaults to checkpoint_dir/mesh.ply)')
    return parser.parse_args()


def main():
    args = parse_cmdline()

    checkpoint_path = _choose_checkpoint(args.checkpoint)

    # resolve device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Loading checkpoint: {checkpoint_path} on device {device}')
    sdf_model, ckpt = _load_sdf_model_from_checkpoint(checkpoint_path, feature_dim=args.feature_dim, device=device)

    bounds = ((args.bounds[0], args.bounds[1], args.bounds[2]), (args.bounds[3], args.bounds[4], args.bounds[5]))

    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(checkpoint_path) or os.getcwd()
        out_path = os.path.join(out_dir, 'mesh.ply')

    print(f'Extracting mesh with resolution={args.resolution}, level={args.level}, bounds={bounds}')
    verts, faces, normals, values = extract_mesh_from_sdf(
        sdf_model,
        bounds=bounds,
        resolution=args.resolution,
        level=args.level,
        device=device,
        batch_size=args.batch_size,
        save_path=out_path
    )

    print(f'Mesh saved to: {out_path}')
    print(f'Vertices: {len(verts)}, Faces: {len(faces)}')


if __name__ == '__main__':
    main()
