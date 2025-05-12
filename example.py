# script adapted from transpose
import os
import torch

import numpy as np
import pandas as pd
import os.path as osp
from scipy.spatial.transform import Rotation as R

from prepare.pre_process import BaseProcessor
from mojito.config import parse_args
from mojito.losses.utils import repr2smpl
from mojito.models.build_model import build_model
from mojito.data.imu_tokenizer_dataset import MojitoIMUTokenizerDataset

class IMUProcessor(BaseProcessor):
    def __init__(self, data_root):
        super().__init__(data_root, 10)
        self.fps = 60
    
    def normalize_tensor(self, x: torch.Tensor, dim=-1, return_norm=False):
        r"""
        Normalize a tensor in a specific dimension to unit norm. (torch)

        :param x: Tensor in any shape.
        :param dim: The dimension to be normalized.
        :param return_norm: If True, norm(length) tensor will also be returned.
        :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
                will also be returned (keepdim=True).
        """
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        return normalized_x if not return_norm else (normalized_x, norm)
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor):
        r"""
        Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

        :param q: Quaternion tensor that can reshape to [batch_size, 4].
        :return: Rotation matrix tensor of shape [batch_size, 3, 3].
        """
        q = self.normalize_tensor(q.reshape(-1, 4))
        a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                    2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                    2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
        return r.view(-1, 3, 3)
    
    def get_mean(self, q, a):
        return q.mean(dim=0), a.mean(dim=0)
    
    def prepare_imu(self):
        quat_stack = []
        acc_stack = []
        for file in sorted(os.listdir(self.data_root)):
            if file.endswith(".csv"):
                df = pd.read_csv(osp.join(self.data_root, file), skiprows=6)
                df = df[["Quat_W", "Quat_X", "Quat_Y", "Quat_Z", "Acc_X", "Acc_Y", "Acc_Z"]]
                qa = df.to_numpy() # (frames, 7)
                quat = qa[:, :4]
                acc = qa[:, 4:]
                quat_stack.append(quat)
                acc_stack.append(acc)

        min_length = min([quat.shape[0] for quat in quat_stack])
        quat_stack = [quat[:min_length] for quat in quat_stack]
        acc_stack = [acc[:min_length] for acc in acc_stack]
        
        quat_stack = torch.from_numpy(np.stack(quat_stack, axis=0).transpose(1, 0, 2)).float() # (frames, 6, 4)
        acc_stack = torch.from_numpy(np.stack(acc_stack, axis=0).transpose(1, 0, 2)).float()
        
        q, a = quat_stack[5:180], acc_stack[5:180]
        oris = self.get_mean(q, a)[0][-1]
        i2s = self.quaternion_to_rotation_matrix(oris).view(3, 3).t() # [3, 3]

        # T-pose
        oris, accs = self.get_mean(q, a) 
        oris = self.quaternion_to_rotation_matrix(oris)
        d2b = i2s.matmul(oris).transpose(1, 2).matmul(torch.eye(3).float())
        acc_offsets = i2s.matmul(accs.unsqueeze(-1))
        
        # calibration
        ori_raw = self.quaternion_to_rotation_matrix(quat_stack).view(-1, 6, 3, 3)
        acc_cal = (i2s.matmul(acc_stack.view(-1, 6, 3, 1)) - acc_offsets).view(-1, 6, 3).transpose(0, 1) # [6, frames, 3]
        ori_cal = i2s.matmul(ori_raw).matmul(d2b).transpose(0, 1) # [6, frames, 3, 3]

        # scale
        acc_scale = 1 / 30
        acc_cal *= acc_scale

        # normalization
        first_global_mat_inv = ori_cal[-1, 0].transpose(0, 1).numpy()

        ori = np.einsum("ij,ntjk->ntik", first_global_mat_inv, ori_cal)
        acc = np.einsum("ij,ntj->nti", first_global_mat_inv, acc_cal)

        T = ori.shape[1]
        imu_agv = []
        for i in range(6):
            imu_agv_i = np.zeros((T - 1,3))
            for t in range(T - 1):
                delta_rotation = R.from_matrix(np.dot(ori[i, t + 1], ori[i, t].T))
                imu_agv_i[t] = delta_rotation.as_rotvec() * self.fps  # 60 fps
            imu_agv_i = np.insert(imu_agv_i, 0, [0, 0, 0], axis=0)
            imu_agv.append(imu_agv_i)
        angular_velocity = np.array(imu_agv)

        res = {
            "ori": ori,
            "acc": acc,
            "agv": angular_velocity
        }
        
        return self.flatten_imu(res)    # (T, 72)
    
if __name__ == '__main__': 
    # Directory for storing raw IMU data. CSV files should be in the same order as IMUs.
    csv_path = 'demo/20240917_163635'
    
    # IMU data from raw csv files
    imu_proc = IMUProcessor(csv_path)
    imu_data = imu_proc.prepare_imu()
    
    # model
    cfg = parse_args(phase="test")  
    model = build_model(cfg, phase="test")
    model = model.cuda()
    model.eval()
    
    ckpt_path = cfg.TEST.CHECKPOINTS
    
    # dataloader, used solely for stats and (de)normalization. Pass an empty dataset list. No data loading required.
    cfg.DATASET.NAME_LIST = []
    dataloader = MojitoIMUTokenizerDataset(phase="test", cfg=cfg)

    # load checkpoint
    milestone = torch.load(ckpt_path, map_location="cpu")
    state_dict = milestone["model"]
    model.load_state_dict(state_dict, strict=True)
    
    # statistics
    motion_mean, motion_std = milestone["motion_mean"], milestone["motion_std"]
    imu_mean, imu_std = milestone["imu_mean"], milestone["imu_std"]
    dataloader.set_statistics(motion_mean, motion_std, imu_mean, imu_std)

    # decode motion from inertial tokens
    motion_eu_mask = dataloader.motion_eu_mask
    imu_quantized = model.val_imu_vae_forward({"imu": torch.from_numpy(imu_data).cuda().float()})
    mocap_rst = model.motion_vae.decoder(imu_quantized.permute(0, 2, 1)).permute(0, 2, 1)
    mocap_rst[0, :, motion_eu_mask] = dataloader.denormalize_motion(mocap_rst[0, :, motion_eu_mask])
    
    # smooth post-processing
    mocap_rst = mocap_rst.squeeze(0)
    assert mocap_rst.shape[0] >= 32, "At least 32 frames needed for smoothing."
        
    smoothed_smpl = repr2smpl(mocap_rst)
    with torch.no_grad():
        smoothed_smpl = model.smooth_postprocess(smoothed_smpl)
        
    np.save(osp.join(r'tmp/example', f'smoothed_smpl.npy'), smoothed_smpl.cpu())
    
    
    
    
    

    