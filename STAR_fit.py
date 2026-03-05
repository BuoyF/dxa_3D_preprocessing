import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree


# CORE STAR FUNCTIONS


def rodrigues_torch(r):
    """Batched Rodrigues rotation: Axis-angle (B,3) -> rotation matrix (B,3,3)"""
    theta = torch.norm(r + 1e-12, dim=1, keepdim=True)
    k = r / theta
    K = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    
    I = torch.eye(3, device=r.device).unsqueeze(0).expand(r.shape[0], -1, -1)
    return I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * torch.matmul(K, K)

def lbs_torch(v_shaped, J_regressor, weights, kintree, pose_aa, posedirs=None):
    """Linear Blend Skinning (LBS) implementation in PyTorch"""
    V, K = v_shaped.shape[0], weights.shape[1]
    device = v_shaped.device
    
    J = torch.matmul(J_regressor, v_shaped)
    R = rodrigues_torch(pose_aa.reshape(K, 3))
    
    v_posed = v_shaped.clone()
    if posedirs is not None:
        pose_feat = (R - torch.eye(3, device=device)).reshape(-1)
        v_posed += torch.einsum('vcp,p->vc', posedirs, pose_feat)

    # Forward Kinematics
    parents = kintree[0].long()
    A = []
    A_rest = []
    I4 = torch.eye(4, device=device)

    for j in range(K):
        p = parents[j].item()
        rel_t = J[j] if j == 0 else J[j] - J[p]
        
        T_local = torch.eye(4, device=device); T_local[:3, :3] = R[j]; T_local[:3, 3] = rel_t
        T_rest = torch.eye(4, device=device); T_rest[:3, 3] = rel_t
        
        if j == 0:
            A.append(T_local); A_rest.append(T_rest)
        else:
            A.append(torch.matmul(A[p], T_local)); A_rest.append(torch.matmul(A_rest[p], T_rest))

    T = torch.stack([torch.matmul(A[i], torch.inverse(A_rest[i])) for i in range(K)])
    
    v_h = torch.cat([v_posed, torch.ones(V, 1, device=device)], dim=1)
    v_tf = torch.einsum('kij,vj->kvi', T, v_h)
    return torch.einsum('vk,kvi->vi', weights, v_tf)[:, :3]


# STAR MODEL CLASS


class STARModel(torch.nn.Module):
    def __init__(self, model_path, device='cpu'):
        super().__init__()
        data = np.load(model_path, allow_pickle=True)
        self.register_buffer('v_template', torch.tensor(data['v_template'], dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(data['shapedirs'][:, :, :10], dtype=torch.float32))
        
        # Robust handling of posedirs shapes
        pd = data['posedirs']
        V = self.v_template.shape[0]
        if pd.ndim == 2: pd = pd.reshape(V, 3, -1)
        elif pd.shape[0] != V: pd = np.transpose(pd, (1, 2, 0))
        self.register_buffer('posedirs', torch.tensor(pd, dtype=torch.float32))
        
        self.register_buffer('J_regressor', torch.tensor(data['J_regressor'], dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(data['weights'], dtype=torch.float32))
        self.register_buffer('kintree', torch.tensor(data['kintree_table'], dtype=torch.long))
        self.faces = data['f']
        self.to(device)

    def get_apose(self, angle=-0.85):
        """Generates A-pose (arms down) axis-angle parameters"""
        pose = torch.zeros(self.weights.shape[1], 3, device=self.v_template.device)
        pose[16, 2], pose[17, 2] = angle, -angle # Shoulders
        return pose.reshape(-1)

    def forward(self, betas, pose=None):
        v_shaped = self.v_template + torch.einsum('vci,i->vc', self.shapedirs, betas)
        return lbs_torch(v_shaped, self.J_regressor, self.weights, self.kintree, 
                         pose if pose is not None else self.get_apose(), self.posedirs)


# BARIATRIC FITTING LOGIC


def get_bariatric_weights(vertices):
    """Assigns higher importance to the torso/belly for bariatric fitting"""
    weights = torch.ones(len(vertices), device=vertices.device)
    y, z, x = vertices[:, 1], vertices[:, 2], vertices[:, 0]
    y_min, y_max = y.min(), y.max()
    
    # Torso: Central height, front facing
    torso = (y > y_min + 0.25 * (y_max-y_min)) & (y < y_min + 0.75 * (y_max-y_min)) & (z > 0)
    weights[torso] = 3.0
    weights[torch.abs(x) > 0.1] = 2.0  # Limbs
    weights[y > y_max - 0.15 * (y_max-y_min)] = 0.5  # Head
    
    return weights / weights.mean()

def fit_to_scan(star, scan_path, target_height_cm=None, iters=300):
    device = star.v_template.device
    scan = trimesh.load(scan_path, process=False)
    v_scan = torch.tensor(scan.vertices, dtype=torch.float32, device=device)
    v_scan -= v_scan.mean(0) # Center scan

    betas = torch.zeros(10, device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)
    trans = torch.zeros(3, device=device, requires_grad=True)
    opt = torch.optim.Adam([betas, scale, trans], lr=0.01)
    
    pose = star.get_apose()
    weights = None

    for i in range(iters):
        opt.zero_grad()
        v_star = star(betas, pose) * scale + trans
        v_star_centered = v_star - v_star.mean(0)
        
        if weights is None: weights = get_bariatric_weights(v_star_centered)
        
        # Weighted Chamfer Distance
        dists = torch.cdist(v_star_centered, v_scan)
        loss_chamfer = (weights * dists.min(1)[0]).mean() + dists.min(0)[0].mean()
        
        loss_prior = 0.001 * (betas**2).sum()
        loss = loss_chamfer + loss_prior
        
        if target_height_cm:
            h_star = (v_star[:, 1].max() - v_star[:, 1].min()) * 100
            loss += 10.0 * ((h_star - target_height_cm)/100)**2

        loss.backward(); opt.step()
        if i % 50 == 0: print(f"Iter {i}: Loss {loss.item():.4f}")

    return v_star.detach().cpu().numpy(), betas.detach().cpu().numpy()


if __name__ == "__main__":
    # Example Usage
    MODEL_PATH = "path/to/star/model.npz"
    SCAN_PATH = "path/to/scan.obj"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    star = STARModel(MODEL_PATH, device)
    
    fitted_verts, final_betas = fit_to_scan(star, SCAN_PATH, target_height_cm=152.4)
    
    # Save Output
    trimesh.Trimesh(fitted_verts, star.faces).export("fitted_result.obj")
    print("✓ Fitting Complete. Result saved to fitted_result.obj")
