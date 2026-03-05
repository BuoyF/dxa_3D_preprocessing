import torch
from star.pytorch.star import STAR

model = STAR(
    model_path="STAR_FEMALE.npz"
)

betas = torch.zeros(1,10)
pose = torch.zeros(1,72)

out = model(pose, betas)

vertices = out.vertices
faces = model.faces

print(vertices.shape)
