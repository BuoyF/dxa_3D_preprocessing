import os
import numpy as np
import torch
import trimesh
from tqdm import tqdm
from smplx import SMPL
from sklearn.neighbors import NearestNeighbors

# ---------------- YOUR PATHS ----------------
MODEL_PATH = "SMPL_python_v.1.1.0/smpl/models/SMPL_FEMALE.pkl" #use male/neutral for other subjets
MESH_PATH  = "human/scans/ba/1_1_3.ply" #change to your own path
GENDER     = "female"           # try 'neutral' for heavy shapes

OUT_BETAS      = "human/scans/ba/betas_1_1_3.npy"
OUT_OBJ_INPOSE = "human/scans/ba/recon_1_1_3_inpose.obj"
OUT_OBJ_APOSE  = "human/scans/ba/recon_1_1_3_apose.obj"

# If you ran transfer_labels_to_scan.py, point this to the produced file:
SCAN_LABELS_NPY = "human/scans/ba/labels_1_1_3/scan_labels.npy"   # set to None to disable gating

# ---------------- PREPROCESS ----------------
PRE_CLEAN = dict(
    remove_degenerate=True,
    remove_duplicate=True,
    remove_infinite=True,
    remove_unreferenced=True,
    min_component_faces=200,    # drop tiny islands
    taubin_iterations=20,       # volume-preserving smooth
    taubin_lambda=0.5,
    taubin_mu=-0.53,
    laplacian_iterations=0,     # extra smoothing (0 to disable)
    laplacian_lambda=0.5
)

# ---------------- FIT SCHEDULE -------------
# staged: torso -> unlock arms -> unlock legs
ITERS_STAGE = (300, 250, 200)
LR = 3e-2

# A-POSE initialization
# change here for other poses
APOSE_DEG  = 50.0
APOSE_AXIS = 'z'                # y is up

# Priors / weights  (looser to allow thickness; stronger smoothing)
BETAS_PRIOR_W = 8e-5
POSE_PRIOR_W  = 1.5e-3
GO_PRIOR_W    = 1e-4
OFFSETS_W     = 2e-4
SMOOTH_W      = 3.0e-3

# ICP warm-up per stage
WARMUP_ICP_STEPS = 12
ICP_INNER_ITERS  = 5
ICP_SUBSAMPLE    = 5000
WITH_SCALING     = True

# ----------------- UTILS -------------------
def smpl_vertex_labels_from_weights(smpl):
    """
    6-class labels for SMPL vertices via LBS weights.
    0 torso, 1 head, 2 arm_L, 3 arm_R, 4 leg_L, 5 leg_R
    """
    W = smpl.lbs_weights.detach().cpu().numpy()  # (V,24)
    HEAD  = [15]
    ARM_L = [16,18,20]      # L shoulder/elbow/wrist
    ARM_R = [17,19,21]
    LEG_L = [1,4,7]         # L hip/knee/ankle
    LEG_R = [2,5,8]
    J_all = set(range(W.shape[1]))
    TORSO = list(J_all - set(HEAD + ARM_L + ARM_R + LEG_L + LEG_R))

    groups = [(0,TORSO),(1,HEAD),(2,ARM_L),(3,ARM_R),(4,LEG_L),(5,LEG_R)]
    lab = np.zeros(W.shape[0], dtype=np.int64)
    for v in range(W.shape[0]):
        gid_best, val_best = 0, -1.0
        for gid, J in groups:
            val = W[v, J].sum()
            if val > val_best: gid_best, val_best = gid, val
        lab[v] = gid_best
    return lab

def model_dir_from_path(p: str) -> str:
    return os.path.dirname(p) if p.lower().endswith(".pkl") else p

def downsample(P: np.ndarray, n: int = 12000, seed: int = 0):
    """
    Returns (P_ds, idx_ds). idx_ds are indices into P.
    """
    if len(P) <= n:
        idx = np.arange(len(P), dtype=np.int64)
        return P, idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=n, replace=False)
    return P[idx], idx

def umeyama(A: np.ndarray, B: np.ndarray, with_scaling: bool = True):
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    C = (A0.T @ B0) / A.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = 1.0
    if with_scaling:
        varA = (A0**2).sum() / max(A.shape[0], 1)
        s = S.sum() / max(varA, 1e-12)
    t = muB - s * (muA @ R)
    return s, R, t

def rigid_align_icp(X: np.ndarray, Y: np.ndarray, iters: int = 10, with_scaling: bool = True):
    X_aligned = X.copy()
    s = 1.0; R = np.eye(3); t = np.zeros(3)
    nn = NearestNeighbors(n_neighbors=1).fit(Y)
    for _ in range(iters):
        idx = nn.kneighbors(X_aligned, return_distance=False)[:, 0]
        s, R, t = umeyama(X_aligned, Y[idx], with_scaling=with_scaling)
        X_aligned = s * (X_aligned @ R) + t
    return X_aligned, s, R, t

def make_body_pose_apose(deg: float = 50.0, axis: str = 'z', device='cpu'):
    pose = torch.zeros(1, 69, device=device)
    L_SH, R_SH = 16, 17
    L_CL, R_CL = 13, 14
    def set_j(j, d, ax):
        off = (j - 1) * 3
        a = {'x':0, 'y':1, 'z':2}[ax]
        pose[0, off + a] = float(np.deg2rad(d))
    set_j(L_SH, -deg, axis); set_j(R_SH, +deg, axis)
    set_j(L_CL, -0.3*deg, axis); set_j(R_CL, +0.3*deg, axis)
    return pose

def build_mask(joints, device):
    m = torch.zeros(1, 69, device=device)
    for j in joints:
        off = (j - 1) * 3
        m[0, off:off+3] = 1.0
    return m

ARMS_JOINTS = [13,14,16,17,18,19,20,21]     # collars, shoulders, elbows, wrists
LEGS_JOINTS = [1,2,4,5,6,7,8,9]             # hips, knees, ankles
ELBOWS = [18, 19]  # L/R elbows
KNEES  = [6, 7]    # L/R knees

def save_obj(path: str, V: np.ndarray, F: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in F:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def faces_to_edges(faces_np: np.ndarray) -> np.ndarray:
    E = set()
    for a,b,c in faces_np:
        E.add(tuple(sorted((a,b)))); E.add(tuple(sorted((b,c)))); E.add(tuple(sorted((c,a))))
    return np.array(list(E), dtype=np.int64)

def torso_weights(points: np.ndarray, torso_q=(0.18,0.88), lateral_shrink=0.75):
    y = points[:, 1]
    ylo, yhi = np.quantile(y, torso_q[0]), np.quantile(y, torso_q[1])
    in_band = (y >= ylo) & (y <= yhi)
    r = np.linalg.norm(points[:, [0,2]], axis=1)
    r_med = np.median(r[in_band]) if np.any(in_band) else np.median(r)
    w = np.where(in_band, 1.9, 0.7).astype(np.float32)
    w = np.where(in_band & (r > lateral_shrink * r_med), 0.95, w).astype(np.float32)
    return w

def classify_scan_points(points, torso_q=(0.18, 0.88), lateral_q=0.70):
    y = points[:,1]
    ylo, yhi = np.quantile(y, torso_q[0]), np.quantile(y, torso_q[1])
    in_band = (y >= ylo) & (y <= yhi)
    r = np.linalg.norm(points[:,[0,2]], axis=1)
    r_med = np.median(r[in_band]) if np.any(in_band) else np.median(r)
    r_thr = max(np.quantile(r[in_band], lateral_q), 1e-6) if np.any(in_band) else r_med
    core = in_band & (r <= 0.9*r_thr)
    lateral = in_band & (r >= r_thr)
    return core.astype(np.float32), lateral.astype(np.float32)

def model_vertex_weights(Valign, stage: int):
    with torch.no_grad():
        r = torch.linalg.norm(Valign[:,[0,2]], dim=1)
        r_thr = torch.quantile(r, 0.70)
        core_mask    = (r <= 0.9*r_thr)
        lateral_mask = (r >= r_thr)
        w = torch.ones_like(r)
        if stage == 1:
            w[core_mask]    = 1.5
            w[lateral_mask] = 0.5
        elif stage == 2:
            w[core_mask]    = 1.2
            w[lateral_mask] = 0.9
        else:
            w[:] = 1.0
    return w

def joint_limit_prior(body_pose, joints, axis='x', max_deg=160.0, w=5e-4):
    ax = {'x':0, 'y':1, 'z':2}[axis]
    loss = 0.0
    for j in joints:
        off = (j-1)*3 + ax
        ang = torch.rad2deg(body_pose[0, off])
        over = torch.clamp(torch.abs(ang) - max_deg, min=0.0)
        loss = loss + (over**2)
    return w * loss

def side_y_bands(points_np):
    """
    For normalized points (y-up), return:
      - y_norm in [0,1]
      - side: +1 for x>=0 (body-left), -1 for x<0 (body-right)
    """
    y = points_np[:,1]
    ymin, ymax = float(y.min()), float(y.max())
    h = max(ymax - ymin, 1e-8)
    y_norm = (y - ymin) / h
    side = np.where(points_np[:,0] >= 0.0, 1, -1).astype(np.int8)
    return y_norm.astype(np.float32), side

# ---------------- PREPROCESS SCAN ----------------
def preprocess_scan(path: str, cfg=PRE_CLEAN) -> trimesh.Trimesh:
    m = trimesh.load(path, process=True)
    if not isinstance(m, trimesh.Trimesh):
        raise RuntimeError(f"Not a triangle mesh: {path}")

    if cfg.get("remove_infinite", True):
        try: m.remove_infinite_values()
        except Exception: pass

    if cfg.get("remove_degenerate", True):
        try: trimesh.repair.fix_inversion(m, multibody=True)
        except Exception: pass
        try: trimesh.repair.fix_normals(m)
        except Exception: pass

    if cfg.get("remove_duplicate", True):
        try: m.update_faces(m.unique_faces())  # replacement for deprecated remove_duplicate_faces
        except Exception: pass
        try: m.remove_unreferenced_vertices()
        except Exception: pass

    if cfg.get("remove_unreferenced", True):
        try: m.remove_unreferenced_vertices()
        except Exception: pass

    min_faces = int(cfg.get("min_component_faces", 0) or 0)
    if min_faces > 0:
        comps = m.split(only_watertight=False)
        comps = [c for c in comps if len(c.faces) >= min_faces]
        if len(comps) > 0:
            m = trimesh.util.concatenate(comps)

    it_t = int(cfg.get("taubin_iterations", 0) or 0)
    if it_t > 0:
        try:
            trimesh.smoothing.filter_taubin(
                m, lamb=float(cfg.get("taubin_lambda", 0.5)),
                nu=float(cfg.get("taubin_mu", -0.53)), iterations=it_t
            )
        except Exception: pass

    it_l = int(cfg.get("laplacian_iterations", 0) or 0)
    if it_l > 0:
        try:
            trimesh.smoothing.filter_laplacian(
                m, lamb=float(cfg.get("laplacian_lambda", 0.5)), iterations=it_l
            )
        except Exception: pass

    try:
        m.rezero(); m.remove_unreferenced_vertices()
    except Exception: pass

    m.vertices = m.vertices.astype(np.float32)
    return m

# --------------- CORE FITTER ----------------
def fit_contact_offsets(mesh_path: str, model_dir: str, gender="female",
                        beta_dim_requested=10, iters_stage=(300,250,200), lr=3e-2,
                        apose_deg=50.0, apose_axis='z'):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- preprocess + normalize (y-up) ---
    scan = preprocess_scan(mesh_path, PRE_CLEAN)
    P0 = np.asarray(scan.vertices, dtype=np.float32)
    mean0 = P0.mean(0, keepdims=True)
    P  = P0 - mean0
    hY = P[:,1].max() - P[:,1].min()
    scaleY = hY if hY > 0 else 1.0
    Pn = P / scaleY
    Pn_ds, idx_ds = downsample(Pn, n=12000)

    # coarse spatial bands/sides for extra gating
    yP_norm_np, sideP_np = side_y_bands(Pn_ds)
    yP_norm_t = torch.tensor(yP_norm_np, dtype=torch.float32, device=device)
    sideP_t   = torch.tensor(sideP_np,   dtype=torch.int8,    device=device)

    # Optional: load per-vertex scan labels and map to downsample
    labels_P_ds = None
    if SCAN_LABELS_NPY and os.path.isfile(SCAN_LABELS_NPY):
        try:
            labels_full = np.load(SCAN_LABELS_NPY).astype(np.int64)  # length = len(original scan vertices)
            nn_map = NearestNeighbors(n_neighbors=1).fit(Pn)  # Pn is normalized full verts
            jmap = nn_map.kneighbors(Pn_ds, return_distance=False)[:,0]
            labels_P_ds = labels_full[jmap]  # (N_ds,)
        except Exception as e:
            print(f"[warn] Failed to load labels from {SCAN_LABELS_NPY}: {e}")

    # scan weights & partitions
    core_np, lat_np = classify_scan_points(Pn_ds)
    w_core_t = torch.tensor(core_np, dtype=torch.float32, device=device)
    w_lat_t  = torch.tensor(lat_np,  dtype=torch.float32, device=device)
    w_scan_base = torso_weights(Pn_ds)
    w_scan_base_t = torch.tensor(w_scan_base, dtype=torch.float32, device=device)
    P_t = torch.tensor(Pn_ds, dtype=torch.float32, device=device)

    # SMPL
    smpl = SMPL(model_path=model_dir, gender=gender).to(device).eval()
    try:
        shape_dim_available = int(smpl.shapedirs.shape[-1])
    except Exception:
        shape_dim_available = 10
    beta_dim = min(beta_dim_requested, shape_dim_available)

    # SMPL vertex labels (for gating)
    labels_V = smpl_vertex_labels_from_weights(smpl)
    labels_V_t = torch.tensor(labels_V, device=device, dtype=torch.long)  # (M,) but M unknown until forward

    # parameters
    betas = torch.zeros(beta_dim, dtype=torch.float32, device=device, requires_grad=True)
    body_base = make_body_pose_apose(apose_deg, apose_axis, device=device)
    pose_delta = torch.zeros_like(body_base, requires_grad=True)
    go = torch.zeros(1,3, dtype=torch.float32, device=device, requires_grad=True)

    offsets = None         # (M,3) will init lazily after first forward
    edges_t = None         # (E,2) for smoothness

    # stage masks (ON THE RIGHT DEVICE!)
    mask_none = torch.zeros_like(pose_delta, device=device)
    mask_arms = build_mask(ARMS_JOINTS, device)
    mask_legs = build_mask(LEGS_JOINTS, device)
    mask_full = torch.clamp(mask_arms + mask_legs, 0, 1)

    # ICP state in normalized space
    s_icp = 1.0
    R_icp = np.eye(3, dtype=np.float32)
    t_icp = np.zeros(3, dtype=np.float32)

    # prepare scan labels tensor if available
    labels_P_t = torch.tensor(labels_P_ds, device=device, dtype=torch.long) if labels_P_ds is not None else None

    def stage_opt(mask, n_iters, lr, stage_idx: int):
        nonlocal s_icp, R_icp, t_icp, offsets, edges_t, labels_V_t
        params = [betas, pose_delta, go]  # offsets added lazily
        opt = None

        for it in tqdm(range(n_iters), desc=f"Stage {stage_idx}"):
            # forward once to know vertex count
            body_pose = body_base + (pose_delta * mask)
            out = smpl(betas=betas.unsqueeze(0), body_pose=body_pose, global_orient=go)
            V = out.vertices[0]
            M = V.shape[0]

            if offsets is None or offsets.shape[0] != M:
                offsets = torch.zeros(M, 3, dtype=torch.float32, device=device, requires_grad=True)
                edges = faces_to_edges(smpl.faces.astype(np.int64))
                edges_t = torch.tensor(edges, dtype=torch.long, device=device)
                labels_V_t = labels_V_t[:M]  # ensure device tensor matches current M
                params = [betas, pose_delta, go, offsets]
                opt = torch.optim.Adam(params, lr=lr)
            if opt is None:
                opt = torch.optim.Adam(params, lr=lr)

            opt.zero_grad()

            # recompute with grad on current params
            body_pose = body_base + (pose_delta * mask)
            out = smpl(betas=betas.unsqueeze(0), body_pose=body_pose, global_orient=go)
            V = out.vertices[0] + offsets  # apply offsets in posed space

            # ICP warm-up at the start of each stage
            if it < WARMUP_ICP_STEPS:
                V_np = V.detach().cpu().numpy()
                if len(V_np) > ICP_SUBSAMPLE:
                    V_sub = V_np[np.random.choice(len(V_np), ICP_SUBSAMPLE, replace=False)]
                else:
                    V_sub = V_np
                _, s_icp, R_icp, t_icp = rigid_align_icp(V_sub, Pn_ds, iters=ICP_INNER_ITERS, with_scaling=WITH_SCALING)

            # apply similarity (torch)
            R_t = torch.tensor(R_icp, dtype=torch.float32, device=device)
            t_t = torch.tensor(t_icp, dtype=torch.float32, device=device)
            s_t = torch.tensor(s_icp, dtype=torch.float32, device=device)
            Valign = s_t * (V @ R_t) + t_t  # (M,3)

            # Distances
            Pa = P_t.unsqueeze(1)     # [N,1,3]
            Va = Valign.unsqueeze(0)  # [1,M,3]
            d2 = torch.sum((Pa - Va) ** 2, dim=2)  # [N,M]

            # ---- STRICT GATING: part + y-band + left/right side ----
            if labels_P_t is not None and labels_P_t.shape[0] == P_t.shape[0]:
                # model coarse bands/sides (recompute each iter)
                V_np2 = Valign.detach().cpu().numpy()
                yV_norm_np, sideV_np = side_y_bands(V_np2)
                yV_norm_t = torch.tensor(yV_norm_np, dtype=torch.float32, device=device)
                sideV_t   = torch.tensor(sideV_np,   dtype=torch.int8,    device=device)

                # part match
                compat = (labels_P_t[:, None] == labels_V_t[None, :])  # [N,M] bool

                # y-band match: legs <=0.35, torso 0.35-0.85, head >=0.85
                bandP_leg  = (yP_norm_t <= 0.35)[:, None]
                bandP_mid  = ((yP_norm_t > 0.35) & (yP_norm_t < 0.85))[:, None]
                bandP_head = (yP_norm_t >= 0.85)[:, None]
                bandV_leg  = (yV_norm_t <= 0.35)[None, :]
                bandV_mid  = ((yV_norm_t > 0.35) & (yV_norm_t < 0.85))[None, :]
                bandV_head = (yV_norm_t >= 0.85)[None, :]
                compat_band = (bandP_leg & bandV_leg) | (bandP_mid & bandV_mid) | (bandP_head & bandV_head)

                # side match for limbs; torso/head exempt
                limb_P = (labels_P_t >= 2)
                limb_V = (labels_V_t >= 2)
                expect_side_P = torch.where((labels_P_t==2)|(labels_P_t==4), torch.ones_like(labels_P_t, dtype=torch.int8), torch.full_like(labels_P_t, -1, dtype=torch.int8))
                expect_side_V = torch.where((labels_V_t==2)|(labels_V_t==4), torch.ones_like(labels_V_t, dtype=torch.int8), torch.full_like(labels_V_t, -1, dtype=torch.int8))
                side_ok = ( ( ~limb_P[:,None]) | ( ~limb_V[None,:]) |
                            ( (sideP_t[:,None] == expect_side_P[:,None]) & (sideV_t[None,:] == expect_side_V[None,:]) ) )

                compat_all = compat & compat_band & side_ok
                big = torch.full_like(d2, 1e6)
                d2 = torch.where(compat_all, d2, big)

            dist = torch.sqrt(d2 + 1e-8)

            # ---- Scan->Model TRIMMED + weights ----
            if stage_idx == 1:
                w_scan = 1.6*w_core_t + 0.3*w_lat_t + 0.7*w_scan_base_t
            elif stage_idx == 2:
                w_scan = 1.2*w_core_t + 0.8*w_lat_t + 0.8*w_scan_base_t
            else:
                w_scan = w_scan_base_t

            s2m_all = dist.min(dim=1).values  # (N,)
            q = torch.quantile(s2m_all.detach(), 0.85)  # keep 85%
            keep_s = (s2m_all <= q)
            s2m = (s2m_all[keep_s] * w_scan[keep_s]).sum() / (w_scan[keep_s].sum() + 1e-9)

            # ---- Model->Scan TRIMMED + model-side weights ----
            m2s_all = dist.min(dim=0).values  # (M,)
            w_model = model_vertex_weights(Valign.detach(), stage_idx).to(m2s_all.device)
            qm = torch.quantile(m2s_all.detach(), 0.85)
            keep_m = (m2s_all <= qm)
            m2s = (m2s_all[keep_m] * w_model[keep_m]).sum() / (w_model[keep_m].sum() + 1e-9)

            chamfer = 0.5 * (s2m + m2s)

            # priors
            prior_betas = BETAS_PRIOR_W * torch.sum(betas ** 2)
            prior_pose  = POSE_PRIOR_W  * torch.sum((pose_delta * mask) ** 2)
            prior_go    = GO_PRIOR_W    * torch.sum(go ** 2)

            # offsets magnitude + smoothness
            mag = OFFSETS_W * torch.sum(offsets ** 2)
            i, j = edges_t[:,0], edges_t[:,1]
            smooth = SMOOTH_W * torch.sum((offsets[i] - offsets[j]) ** 2)

            # soft joint limits for elbows/knees (x-axis hinge approx)
            limit_loss = joint_limit_prior(body_pose, ELBOWS, axis='x', max_deg=160.0, w=5e-4) \
                       + joint_limit_prior(body_pose, KNEES,  axis='x', max_deg=165.0, w=5e-4)

            loss = chamfer + prior_betas + prior_pose + prior_go + mag + smooth + limit_loss
            loss.backward()
            # freeze unmasked pose grads
            pose_delta.grad = pose_delta.grad * mask
            opt.step()

    # --------- RUN STAGES (device-safe masks) ---------
    stage_opt(mask_none, ITERS_STAGE[0], LR, stage_idx=1)          # torso only
    stage_opt(mask_arms, ITERS_STAGE[1], LR, stage_idx=2)          # unlock arms
    stage_opt(mask_full, ITERS_STAGE[2], LR, stage_idx=3)          # unlock legs (full)

    out = {
        "betas": betas.detach().cpu().numpy(),
        "body_pose": (make_body_pose_apose(apose_deg, apose_axis, device=device) + pose_delta).detach().cpu().numpy(),
        "global_orient": go.detach().cpu().numpy(),
        "beta_dim": beta_dim,
        "s_icp": float(s_icp), "R_icp": R_icp, "t_icp": t_icp,
        "scaleY": float(scaleY), "scan_mean": mean0[0].astype(np.float32)
    }
    return out, smpl, offsets.detach().cpu().numpy()

# ---------------- MAIN ---------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = model_dir_from_path(MODEL_PATH)

    results, smpl, final_offsets = fit_contact_offsets(
        mesh_path=MESH_PATH,
        model_dir=model_dir,
        gender=GENDER,
        beta_dim_requested=10,
        iters_stage=ITERS_STAGE,
        lr=LR,
        apose_deg=APOSE_DEG,
        apose_axis=APOSE_AXIS
    )

    betas = results["betas"]
    body_pose = torch.tensor(results["body_pose"], dtype=torch.float32, device=device)
    go = torch.tensor(results["global_orient"], dtype=torch.float32, device=device)

    np.save(OUT_BETAS, betas.astype(np.float32))
    print(f"Saved betas (len={results['beta_dim']}): {OUT_BETAS}\n{betas}")

    # IN-POSE reconstruction (with offsets), back to original coords
    with torch.no_grad():
        V_in = smpl(
            betas=torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0),
            body_pose=body_pose,
            global_orient=go
        ).vertices[0].detach().cpu().numpy()
    V_in += final_offsets
    Vn = results["s_icp"] * (V_in @ results["R_icp"]) + results["t_icp"]
    V_world = Vn * results["scaleY"] + results["scan_mean"]
    save_obj(OUT_OBJ_INPOSE, V_world, smpl.faces)
    print("Wrote IN-POSE fitted mesh:", OUT_OBJ_INPOSE)

    # A-POSE reconstruction (betas only; keep canonical clean)
    apose = make_body_pose_apose(APOSE_DEG, APOSE_AXIS, device=device)
    with torch.no_grad():
        V_a = smpl(
            betas=torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0),
            body_pose=apose,
            global_orient=torch.zeros(1,3,device=device)
        ).vertices[0].detach().cpu().numpy()
    V_a = results["s_icp"] * (V_a @ results["R_icp"]) + results["t_icp"]
    V_a = V_a * results["scaleY"] + results["scan_mean"]
    save_obj(OUT_OBJ_APOSE, V_a, smpl.faces)
    print("Wrote A-POSE (canonical) mesh:", OUT_OBJ_APOSE)
