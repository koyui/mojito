import torch
import numpy as np
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
)
import os

# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def fake_cameras(render_width):
    # Set fake_cameras to check
    R = np.eye(3).reshape((1, 3, 3))
    R[0][0][0] = -1
    R[0][1][1] = -1
    R[0][2][2] = -1
    T = np.array([0, 0, 3]).reshape((1, 3))
    K = [3000, 0, 1900, 0, 3000, 1100, 0, 0, 1]
    K = np.array(K).reshape((3, 3))

    resolution_ratio = 3840 / render_width  # raw images are 4K
    K = K / resolution_ratio
    K[2, 2] = 1

    return R, T, K


def make_renderer(R, T, K, render_height, render_width):
    # define rasterization setting
    cameras = cameras_from_opencv_projection(
        R,
        T,
        K,
        torch.tensor([render_height, render_width]).unsqueeze(0)
    )

    # define rasterization setting
    raster_settings = RasterizationSettings(
        image_size=[render_height, render_width],
        blur_radius=0.0,
        faces_per_pixel=10,
        max_faces_per_bin=100000
    )

    # define shader
    bp = None
    shader = SoftPhongShader(
        device=device,
        cameras=cameras,
        blend_params=bp
    )

    # create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader
    )

    return renderer

def smpl2obj(vertices, path):
    fs = torch.load(r'prepare/pre_process/smpl_faces.pt')
    
    bs = 1
    if len(vertices.shape) == 3:
        bs = vertices.shape[0]
    else:
        vertices = vertices.unsqueeze(0)
    for i in range(bs):
        with open(path, "w") as f:
            for v in vertices[i]:
                f.write(( 'v %f %f %f\n' % ( v[0], v[1], v[2]) ))
            for face in fs:
                f.write(( 'f %d %d %d\n' % ( face[0], face[1], face[2]) ))

def foot_detect(positions: np.ndarray, thres: float):
    fid_l, fid_r,  = [7-1, 10-1], [8-1, 11-1],  # leftfoot, rightfoot
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r