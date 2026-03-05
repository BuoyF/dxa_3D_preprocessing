import torch
from smplx import SMPL

MODEL_PATH = "SMPL_python_v.1.1.0/smpl/models"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SMPL(
    model_path=MODEL_PATH,
    gender="female"
).to(device)

betas = torch.zeros(1,10).to(device)
body_pose = torch.zeros(1,69).to(device)
global_orient = torch.zeros(1,3).to(device)

output = model(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient
)

vertices = output.vertices
faces = model.faces

print(vertices.shape)
print(faces.shape)
