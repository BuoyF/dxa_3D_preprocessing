import numpy as np
import trimesh
import torch
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

def load_ply(path: str) -> np.ndarray:
    try:
        mesh = trimesh.load(path, process=False)
        return mesh.vertices.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load .ply file {path}: {e}")

def load_obj(path: str) -> np.ndarray:
    try:
        mesh = trimesh.load(path, process=False)
        return mesh.vertices.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load .obj file {path}: {e}")

def load_mesh(path: str) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.ply':
        return load_ply(str(path))
    elif path.suffix.lower() == '.obj':
        return load_obj(str(path))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .ply or .obj")

def fix_non_manifold(verts: np.ndarray, faces: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if faces is None:
        return verts
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.merge_vertices()
        mesh.remove_degenerate_faces()
        if len(mesh.edges_unique) > 0:
            mesh.remove_duplicate_faces()
        return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64)
    except Exception as e:
        warnings.warn(f"Failed to fix non-manifold mesh: {e}")
        return verts, faces

def make_watertight(verts: np.ndarray, faces: Optional[np.ndarray] = None, method: str = 'fill') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if faces is None:
        return verts
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if mesh.is_watertight:
            return verts, faces
        if method == 'fill':
            boundary_edges = mesh.outline()
            if boundary_edges is not None:
                mesh.fill_holes()
        return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64)
    except Exception as e:
        warnings.warn(f"Failed to make watertight mesh: {e}")
        return verts, faces

def downsample_vertices(verts: np.ndarray, n: int = 100000, seed: int = 0) -> np.ndarray:
    if verts.shape[0] <= n:
        return verts
    rng = np.random.default_rng(seed)
    idx = rng.choice(verts.shape[0], n, replace=False)
    return verts[idx]

def upsample_vertices(verts: np.ndarray, n: int = 100000, seed: int = 0) -> np.ndarray:
    if verts.shape[0] >= n:
        return verts
    rng = np.random.default_rng(seed)
    idx = rng.choice(verts.shape[0], n, replace=True)
    return verts[idx]

def farthest_point_sampling(verts: np.ndarray, n: int = 100000) -> np.ndarray:
    if verts.shape[0] <= n:
        return verts
    N = verts.shape[0]
    sampled_idx = np.zeros(n, dtype=np.int32)
    sampled_idx[0] = np.random.randint(N)
    distances = np.full(N, np.inf)
    
    for i in range(1, n):
        last_point = verts[sampled_idx[i-1]]
        dists = np.sum((verts - last_point) ** 2, axis=1)
        distances = np.minimum(distances, dists)
        sampled_idx[i] = np.argmax(distances)
    return verts[sampled_idx]

def downsample_voxel(verts: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
    if verts.shape[0] == 0:
        return verts
    verts_min, verts_max = verts.min(axis=0), verts.max(axis=0)
    verts_norm = (verts - verts_min) / (verts_max - verts_min + 1e-6)
    voxel_indices = (verts_norm / voxel_size).astype(np.int32)
    
    # Spatial hashing
    voxel_hash = voxel_indices[:, 0] * 73856093 ^ voxel_indices[:, 1] * 19349663 ^ voxel_indices[:, 2] * 83492791
    unique_voxels, inverse_indices = np.unique(voxel_hash, return_inverse=True)
    
    downsampled_verts = []
    for voxel_id in range(len(unique_voxels)):
        mask = inverse_indices == voxel_id
        downsampled_verts.append(verts[mask].mean(axis=0))
    return np.array(downsampled_verts, dtype=np.float32)

def downsample_poisson_disk(verts: np.ndarray, radius: float = 0.05, max_samples: int = None) -> np.ndarray:
    if verts.shape[0] == 0 or (max_samples and verts.shape[0] <= max_samples):
        return verts
    
    verts_min, verts_max = verts.min(axis=0), verts.max(axis=0)
    verts_norm = (verts - verts_min) / (verts_max - verts_min + 1e-6)
    
    grid, result = {}, []
    idx_order = np.arange(len(verts_norm))
    np.random.shuffle(idx_order)
    
    for idx in idx_order:
        point = verts_norm[idx]
        cell_id = tuple((point / radius).astype(int))
        
        is_valid = True
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_id = (cell_id[0]+dx, cell_id[1]+dy, cell_id[2]+dz)
                    if neighbor_id in grid:
                        if np.linalg.norm(point - grid[neighbor_id]) < radius:
                            is_valid = False; break
                if not is_valid: break
            if not is_valid: break
        
        if is_valid:
            grid[cell_id] = point
            result.append(verts[idx])
            if max_samples and len(result) >= max_samples: break
            
    return np.array(result, dtype=np.float32) if result else verts[:1]

def downsample_quadric(verts: np.ndarray, target_count: int = 100000, target_reduction: Optional[float] = None) -> np.ndarray:
    if verts.shape[0] <= target_count:
        return verts
    if target_reduction is not None:
        target_count = max(1, int(verts.shape[0] * (1 - target_reduction)))
    
    try:
        mesh = trimesh.Trimesh(vertices=verts, process=True)
        target_fraction = np.clip(target_count / len(mesh.vertices), 0.01, 1.0)
        mesh.simplify_mesh(target_fraction=target_fraction, update=True)
        return mesh.vertices.astype(np.float32)
    except Exception as e:
        warnings.warn(f"Quadric simplification failed, using FPS: {e}")
        return farthest_point_sampling(verts, target_count)

def downsample_grid(verts: np.ndarray, grid_size: int = 100) -> np.ndarray:
    if verts.shape[0] == 0:
        return verts
    verts_min, verts_max = verts.min(axis=0), verts.max(axis=0)
    verts_norm = (verts - verts_min) / (verts_max - verts_min + 1e-6) * (grid_size - 1)
    cell_indices = np.clip(verts_norm.astype(np.int32), 0, grid_size - 1)
    cell_hash = cell_indices[:, 0] * (grid_size**2) + cell_indices[:, 1] * grid_size + cell_indices[:, 2]
    
    result = {}
    for i, cell_id in enumerate(cell_hash):
        if cell_id not in result: result[cell_id] = []
        result[cell_id].append(verts[i])
    
    return np.array([np.mean(v, axis=0) for v in result.values()], dtype=np.float32)

def standardize_vertices(verts: np.ndarray, n: int = 100000, method: str = 'random', seed: int = 0) -> np.ndarray:
    if verts.shape[0] == n:
        return verts
    if verts.shape[0] > n:
        if method == 'fps': return farthest_point_sampling(verts, n)
        if method == 'voxel':
            avg_dim = np.mean(verts.max(axis=0) - verts.min(axis=0))
            return downsample_voxel(verts, voxel_size=avg_dim / (n ** (1/3)))
        if method == 'poisson':
            avg_dim = np.mean(verts.max(axis=0) - verts.min(axis=0))
            return downsample_poisson_disk(verts, radius=avg_dim / (n ** (1/3)), max_samples=n)
        if method == 'quadric': return downsample_quadric(verts, target_count=n)
        if method == 'grid': return downsample_grid(verts, grid_size=int(np.ceil(n ** (1/3))))
        return downsample_vertices(verts, n, seed)
    return upsample_vertices(verts, n, seed)

def normalize_vertices(verts: np.ndarray, method: str = 'unit_sphere') -> Tuple[np.ndarray, dict]:
    centroid = verts.mean(axis=0)
    verts_centered = verts - centroid
    
    if method == 'unit_sphere':
        scale = np.max(np.linalg.norm(verts_centered, axis=1))
        verts_normalized = verts_centered / scale if scale > 0 else verts_centered
    elif method == 'unit_cube':
        bbox_min, bbox_max = verts_centered.min(axis=0), verts_centered.max(axis=0)
        scale = np.max(bbox_max - bbox_min)
        verts_normalized = verts_centered / (scale / 2) if scale > 0 else verts_centered
    elif method == 'center':
        verts_normalized, scale = verts_centered, 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return verts_normalized, {'centroid': centroid, 'scale': scale, 'method': method}

def denormalize_vertices(verts: np.ndarray, norm_params: dict) -> np.ndarray:
    return (verts * norm_params['scale']) + norm_params['centroid']

def compute_normals(verts: np.ndarray, faces: Optional[np.ndarray] = None) -> np.ndarray:
    if faces is None: return None
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return mesh.vertex_normals.astype(np.float32)
    except:
        return None

def augment_vertices(verts: np.ndarray, rotation_y: float = 10.0, rotation_xz: float = 2.0,
                     scaling: Tuple[float, float] = (0.95, 1.05), translation: float = 0.05,
                     jitter_std: float = 0.01, dropout: float = 0.05,
                     random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
    rs = random_state or np.random.RandomState()
    v_aug = verts.copy()
    
    # Apply Rotations
    if rotation_y > 0:
        a = rs.uniform(-rotation_y, rotation_y) * np.pi / 180
        rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)
        v_aug = v_aug @ rot.T
    
    if rotation_xz > 0:
        ax, az = rs.uniform(-rotation_xz, rotation_xz, 2) * np.pi / 180
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]], dtype=np.float32)
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]], dtype=np.float32)
        v_aug = v_aug @ rx.T @ rz.T
    
    # Scale, Translate, Jitter
    v_aug *= rs.uniform(scaling[0], scaling[1])
    if translation > 0:
        v_aug += rs.uniform(-translation, translation, size=3) * (v_aug.max(axis=0) - v_aug.min(axis=0))
    if jitter_std > 0:
        v_aug += rs.normal(0, jitter_std, size=v_aug.shape).astype(np.float32)
    if dropout > 0:
        v_aug = v_aug[rs.choice(len(v_aug), int(len(v_aug) * (1 - dropout)), replace=False)]
        
    return v_aug

def batch_vertices(verts_list: list, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(v) for v in verts_list], dtype=torch.long)
    batched = torch.full((len(verts_list), max(lengths), 3), pad_value, dtype=torch.float32)
    for i, verts in enumerate(verts_list):
        batched[i, :lengths[i]] = torch.from_numpy(verts)
    return batched, lengths

def preprocess_scan(scan_path: str, target_vertices: int = 100000, normalization: str = 'unit_sphere',
                    downsample_method: str = 'random', fix_non_manifold_mesh: bool = False,
                    make_watertight_mesh: bool = False, augment: bool = False,
                    augment_params: Optional[dict] = None, seed: int = 0) -> Tuple[np.ndarray, dict]:
    verts = load_mesh(scan_path)
    metadata = {'scan_path': str(scan_path), 'original_n_verts': len(verts)}
    
    try:
        mesh = trimesh.load(scan_path, process=False)
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            if fix_non_manifold_mesh: verts, _ = fix_non_manifold(verts, mesh.faces)
            if make_watertight_mesh: verts, _ = make_watertight(verts, mesh.faces)
    except Exception as e:
        warnings.warn(f"Mesh cleaning failed: {e}")
    
    verts = standardize_vertices(verts, target_vertices, method=downsample_method, seed=seed)
    verts, norm_params = normalize_vertices(verts, method=normalization)
    
    if augment and augment_params:
        verts = augment_vertices(verts, random_state=np.random.RandomState(seed), **augment_params)
    
    metadata.update({'target_n_verts': target_vertices, 'norm_params': norm_params, 'augmented': augment})
    return verts, metadata

if __name__ == "__main__":
    print("Running Tests...")
    test_verts = np.random.randn(5000, 3).astype(np.float32)
    std_verts = standardize_vertices(test_verts, n=2000, method='fps')
    print(f"FPS Downsampling: {test_verts.shape} -> {std_verts.shape}")
    
    norm_verts, params = normalize_vertices(std_verts, method='unit_sphere')
    print(f"Normalization (Sphere) Max Dist: {np.linalg.norm(norm_verts, axis=1).max():.4f}")
    
    batched, lengths = batch_vertices([norm_verts, norm_verts[:1000]])
    print(f"Batching successful: {batched.shape}")
