import numpy as np
import torch
import os
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.transforms import *

from .utils import *
from body_model.body_model import BodyModel

class BaseProcessor():
    def __init__(
        self,
        root_path: str,
        smpl_shape_dim: int
    ):
        # meta information
        self.dataset_name = None
        self.data_root = root_path
        self.device = self.set_device()
        self.smpl_shape_dim = smpl_shape_dim
            
        # imu configuration
        self.k_chains = [
            [0, 3, 6, 9, 13, 16, 18],   # imu-0: left wrist
            [0, 3, 6, 9, 14, 17, 19],   # imu-1: right wrist
            [0, 1, 4],                  # imu-2: left knee
            [0, 2, 5],                  # imu-3: right knee
            [0, 3, 6, 9, 12, 15],       # imu-4: head
            [0]                         # imu-5: root
        ]
        self.verts_with_imu = [1961, 5424, 1176, 4662, 411, 3021]

        # smpl model
        self.joint_num = 22
        self.bm = BodyModel(
            bm_path=r'body_model/smplh/neutral/model.npz',
            num_betas=smpl_shape_dim,
            batch_size=1,
            num_expressions=None,
            model_type="smplh"
        ).to(self.device)
    
    def set_device(self):
        if torch.cuda.is_available(): 
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        return device

    def process(self):
        raise NotImplementedError

    def normalize(self, to_norm):
        """
        Normalize raw motion data

        params:
            to_norm: Raw moton data with shape (Frames, 75). The first 3 values are global translation, 
                the next 3 values are global orientation.
        Returns:
            new_pose: Normalized motion data with the same shape as input.
        """
        assert to_norm.shape[1] == 75, "Raw Motion Data should in size (Frames x 75)"

        to_norm = torch.from_numpy(to_norm).float()
        first_global_t = to_norm[0, :3].view(1, 3)
        first_global_o = to_norm[0, 3:6].view(1, 3)
        first_global_mat_inv = axis_angle_to_matrix(first_global_o).transpose(1, 2)

        new_pose = to_norm.clone()
        new_pose[:, :3] -= first_global_t
        new_pose[:, :3] = torch.einsum("mn,tn->tm", first_global_mat_inv[0], new_pose[:, :3])
        global_rotmat = axis_angle_to_matrix(to_norm[:, 3:6])
        new_pose[:, 3:6] = matrix_to_axis_angle(torch.einsum("mn,tnl->tml", first_global_mat_inv[0], global_rotmat))

        return new_pose.numpy()
    
    def normalize_imu(self, motion_data, imu_data):
        # N, T, _, _ = imu_data['ori'].shape
        first_global_o = motion_data[0, 3:6]
        first_global_o_tensor = torch.from_numpy(first_global_o).view(1, 3)
        first_global_mat_inv = axis_angle_to_matrix(first_global_o_tensor).transpose(1, 2).numpy()

        imu_ori_normalized = np.einsum("ij,ntjk->ntik", first_global_mat_inv[0], imu_data["ori"])
        imu_acc_normalized = np.einsum("ij,ntj->nti", first_global_mat_inv[0], imu_data["acc"])

        return imu_ori_normalized, imu_acc_normalized
    
    def get_smpl_output(self, smpl_motion: dict):
        smpl_pose = torch.from_numpy(smpl_motion).float().to(self.device)
        T = smpl_pose.shape[0]
        smpl_output = self.bm(
            root_orient=smpl_pose[:, 3:6],
            pose_body=smpl_pose[:, 6:69],
            pose_hand=torch.zeros(T, 90).to(self.device),
            betas= torch.zeros(T, self.smpl_shape_dim).to(self.device),   # smpl_shape.view(1, -1).repeat(T, 1)
            trans=smpl_pose[:, :3],
            return_dict=False
        )

        return smpl_output
    
    def to_humor_representation(self, smpl_motion: np.ndarray):
        """
        Convert SMPL motion to HuMoR representation

        params:
            smpl_motion: SMPL motion data with shape (Frames, 75)
        output:
            (Frames - 1, 271)
        """
        # root translation [0:3]
        root_trans = smpl_motion[:, :3]

        # root linear velocity [3:6]
        root_linear_velo = smpl_motion[1:, :3] - smpl_motion[:-1, :3]

        # root orientation [6:12]
        root_orient_mat = axis_angle_to_matrix(torch.from_numpy(smpl_motion[:, 3:6]).float())
        root_orient_6d = matrix_to_rotation_6d(root_orient_mat).numpy()

        # root angular velocity [12:15]
        root_angular_velo = torch.einsum("tmn,tnl->tml", root_orient_mat[1:], root_orient_mat[:-1].transpose(1, 2))
        root_angular_velo = matrix_to_axis_angle(root_angular_velo).numpy()

        # local joint rotation [15:141]
        local_joint_rot = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(smpl_motion[:, 6:69]).view(-1, 21, 3).float())).view(-1, 21*6).numpy()

        # local joint position [141:204]
        T, _ = smpl_motion.shape
        B = 15000
        if T <= B:
            smpl_output = self.get_smpl_output(smpl_motion)
            smpl_joints = smpl_output.Jtr.cpu().numpy()
        else: # issue: CUDA out of memory
            num_batches = (T + B - 1) // B
            smpl_motion_list = [smpl_motion[B*t:B*(t+1), :] for t in range(num_batches)]
            res_list = []
            for pose in smpl_motion_list:
                batch_output = self.get_smpl_output(pose)
                batch_Jtr = batch_output.Jtr.cpu().numpy()
                res_list.append(batch_Jtr)
            smpl_joints = np.concatenate(res_list, axis=0)
        assert smpl_joints.shape[0] == T
        local_joint_pos = smpl_joints[:, 1:self.joint_num, :]
        local_joint_pos = (local_joint_pos - smpl_joints[:, 0, :].reshape(-1, 1, 3)).reshape(-1, 21*3)

        # local joint linear velocity
        local_joint_linear_velo = local_joint_pos[1:] - local_joint_pos[:-1]

        # foot-ground contacts
        feet_l, feet_r = foot_detect(local_joint_pos.reshape(-1, 21, 3), 0.002)

        # collection
        representation = np.concatenate(
            [
                root_trans[:-1],
                root_linear_velo,
                root_orient_6d[:-1],
                root_angular_velo,
                local_joint_rot[:-1],
                local_joint_pos[:-1],
                local_joint_linear_velo,
                feet_l,
                feet_r
            ],
            axis=1
        )
        
        return representation

    def slerp(
        self,
        t: float,
        v0: np.ndarray,
        v1: np.ndarray
    ):
        v0_norm = v0 / np.linalg.norm(v0, axis=-1, keepdims=True)
        v1_norm = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        dot = np.sum(v0_norm * v1_norm, axis=-1, keepdims=True)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        if sin_theta == 0:
            return (1.0 - t) * v0 + t * v1
        slerp_factor = (np.sin((1.0 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1

        return slerp_factor

    def upsample_slerp(
        self,
        data: np.ndarray,
        upsample_rate: int = 2
    ):
        T, F = data.shape
        upsampled_length = upsample_rate*(T-1) + 1
        upsampled_data = np.zeros((upsampled_length, F))
        upsampled_data[0::upsample_rate] = data

        for i in range(T-1):
            for k in range(1, upsample_rate):
                upsampled_data[i*upsample_rate+k] = self.slerp(1/upsample_rate, data[i], data[i + 1])
        
        return upsampled_data
    
    def plot_imu_signal(rot_data, acc_data, start, end, save_imu_dir):
        '''
        Modded Koyui implementation in IMHD2, plot IMU signals
        
        params:
            rot_data, acc_data: can be derived from imu_signal["ori"] / imu_signal["acc"] or other ways.
            start, end: the interval of the signal data to be plotted should be given.
            save_imu_dir: the directory to save multiple pictures of the graph.
        '''
        # rot_data = imu_signal['ori']; acc_data = imu_signal['acc']
        rot_ub = np.max(rot_data); rot_lb = np.min(rot_data)
        acc_ub = np.max(acc_data); acc_lb = np.min(acc_data)

        end = rot_data.shape[0] if end == -1 else end + 1
        duration = end - start + 1
        gap = duration // 64
        mark_per = gap if gap > 0 else None
        
        x = np.arange(start=start, stop=end, step=1)
        plot_rot = rot_data[start:end].T
        plot_acc = acc_data[start:end].T

        plt.figure(figsize=(16, 4))
        plt.rc('font', family='serif')
        plt.plot(1, x, plot_rot, start, end, rot_lb, rot_ub, mark_per)
        plt.title("rot")
        plt.plot(2, x, plot_acc, start, end, acc_lb, acc_ub, mark_per)
        plt.title("acc")
        plt.suptitle("Imu Signal Visualization")
        plt.savefig(
            os.path.join(save_imu_dir, 'imu.png'),
            bbox_inches='tight',
            dpi=600
        )
        plt.close()
        
    def flatten_imu(self, imu_data):
        """
        Convert IMU signal dict to T x 72 representation

        params:
            {'ori': 6 x T x 3 x 3, 'acc': 6 x T x 3, 'agv': 6 x T x 3}
        output:
            imu_data_flatten (torch.Tensor with shape T x 72)
        """
        _, seq_len, _, _ = imu_data['ori'].shape
        rotation_6d = matrix_to_rotation_6d(torch.from_numpy(imu_data["ori"])).permute(1, 0, 2).reshape(seq_len, -1)  # (6, seq_len, 3, 3) => (6, seq_len, 6) => (seq_len, 6, 6) => (seq_len, 36) 
        acc = torch.from_numpy(imu_data["acc"]).permute(1, 0, 2).reshape(seq_len, -1)  # (6, seq_len, 3) => (seq_len, 6, 3) => (seq_len, 18)
        agv = torch.from_numpy(imu_data["agv"]).permute(1, 0, 2).reshape(seq_len, -1)  # (6, seq_len, 3) => (seq_len, 6, 3) => (seq_len, 18) 
        imu_data_flatten = torch.cat([rotation_6d, acc, agv], dim=1)  # (seq_len, 72)
        assert imu_data_flatten.shape[-1] == 72
        return imu_data_flatten.numpy()
    
    def save_norm_statistics(self):
        """
        Read all motion and imu data, save statistics.pkl for the whole dataset\n
            Data format required:\n
                motion_data (dict with key 'pose_humor_repr')\n
                imu_data (torch.Tensor with shape T x 72), already flattened before\n
            Save {motion_eu_min, motion_eu_max, imu_eu_min, imu_eu_max}, statistics for the whole dataset.
        """
        meta_dir = os.path.join(self.data_root, 'pre_processed', self.dataset_name)
        with open(os.path.join(meta_dir, 'path_meta.pkl'), 'rb') as f:
            meta_data = pkl.load(f)
        motion_path_list = meta_data['motion']
        imu_path_list = meta_data['imu']
        motion_data_list = []
        imu_data_list = []
        
        motion_eu_mask = [
                0, 1, 2,    # root translation
                3, 4, 5,    # root linear velocity
                12, 13, 14 # root angular velocity
            ]
        motion_eu_mask.extend([i for i in range(141, 267)])   # local joint position and linear velocity
        
        # mask for imu
        imu_eu_mask = list(range(36, 72))  # ori (6 x 6) + {acc (6 x 3) + agv (6 x 3)}
        
        print('Start loading motion data.')
        for motion_path in tqdm(motion_path_list):
            with open(motion_path, 'rb') as f:
                motion_data = pkl.load(f)
                assert 'pose_humor_repr' in motion_data, 'No HuMoR repr in motion data'
                motion_data_list.append(torch.from_numpy(motion_data['pose_humor_repr'][:, motion_eu_mask]))
        
        print('Start loading imu data.')
        for imu_path in tqdm(imu_path_list):
            with open(imu_path, 'rb') as f:
                imu_data = pkl.load(f)
                assert imu_data.shape[-1] == 72, 'Not flattened imu data'
                imu_data_list.append(torch.from_numpy(imu_data[:, imu_eu_mask]))
                
        for i in range(len(motion_data_list)):
            assert motion_data_list[i].shape[0] + 1 == imu_data_list[i].shape[0]
        
        
        print('Calculating data.')
        all_motion = torch.cat(motion_data_list, dim=0)
        all_imu = torch.cat(imu_data_list, dim=0)
        
        
        motion_eu_min, _ = torch.min(all_motion, dim=0)
        motion_eu_max, _ = torch.max(all_motion, dim=0)
        imu_eu_min, _ = torch.min(all_imu, dim=0)
        imu_eu_max, _ = torch.max(all_imu, dim=0)
        print(motion_eu_min.shape)
        print(motion_eu_max.shape)
        print(imu_eu_min.shape)
        print(imu_eu_max.shape)
        assert motion_eu_max.shape[0] == len(motion_eu_mask)
        assert imu_eu_max.shape[0] == len(imu_eu_mask)
        
        res = {
            'motion_eu_min': motion_eu_min.numpy(),
            'motion_eu_max': motion_eu_max.numpy(),
            'imu_eu_min': imu_eu_min.numpy(),
            'imu_eu_max': imu_eu_max.numpy()
        }
        print('Now saving')
        with open(os.path.join(meta_dir, 'statistics.pkl'), 'wb') as f:
            pkl.dump(res, f)
        
        print('Done.')
        
    def save_mean_std(self):
        """
        Read all motion and imu data, save mean_std.pkl for the whole dataset\n
            Data format required:\n
                motion_data (dict with key 'pose_humor_repr')\n
                imu_data (torch.Tensor with shape T x 72), already flattened before\n
            Save {motion_mean, motion_std, imu_mean, imu_std}, statistics for the whole dataset.
        """
        meta_dir = os.path.join(self.data_root, 'pre_processed', self.dataset_name)
        with open(os.path.join(meta_dir, 'path_meta.pkl'), 'rb') as f:
            meta_data = pkl.load(f)
        motion_path_list = meta_data['motion']
        imu_path_list = meta_data['imu']
        motion_data_list = []
        imu_data_list = []
        
        motion_eu_mask = [
                0, 1, 2,    # root translation
                3, 4, 5,    # root linear velocity
                12, 13, 14 # root angular velocity
            ]
        motion_eu_mask.extend([i for i in range(141, 267)])   # local joint position and linear velocity
        
        # mask for imu
        imu_eu_mask = list(range(36, 72))  # ori (6 x 6) + {acc (6 x 3) + agv (6 x 3)}
        
        print('Start loading motion data.')
        for motion_path in tqdm(motion_path_list):
            with open(motion_path, 'rb') as f:
                motion_data = pkl.load(f)
                assert 'pose_humor_repr' in motion_data, 'No HuMoR repr in motion data'
                motion_data_list.append(torch.from_numpy(motion_data['pose_humor_repr'][:, motion_eu_mask]))
        
        print('Start loading imu data.')
        for imu_path in tqdm(imu_path_list):
            with open(imu_path, 'rb') as f:
                imu_data = pkl.load(f)
                assert imu_data.shape[-1] == 72, 'Not flattened imu data'
                imu_data_list.append(torch.from_numpy(imu_data[:, imu_eu_mask]))
                
        for i in range(len(motion_data_list)):
            assert motion_data_list[i].shape[0] + 1 == imu_data_list[i].shape[0]
            
        print('Calculating data.')
        all_motion = torch.cat(motion_data_list, dim=0)
        all_imu = torch.cat(imu_data_list, dim=0)
        motion_frames = all_motion.shape[0]
        imu_frames = all_imu.shape[0]
        print(all_motion.shape)
        print(all_imu.shape)
        
        motion_mean= torch.mean(all_motion, dim=0)
        motion_std = torch.std(all_motion, dim=0, unbiased=False)
        imu_mean = torch.mean(all_imu, dim=0)
        imu_std = torch.std(all_imu, dim=0, unbiased=False)
        print(motion_mean.shape)
        print(motion_std.shape)
        print(imu_mean.shape)
        print(imu_std.shape)
        assert motion_mean.shape[0] == len(motion_eu_mask)
        assert imu_std.shape[0] == len(imu_eu_mask)
        
        res = {
            'motion_mean': motion_mean.numpy(),
            'motion_std': motion_std.numpy(),
            'imu_mean': imu_mean.numpy(),
            'imu_std': imu_std.numpy(),
            'motion_frames': motion_frames,
            'imu_frames': imu_frames
        }
        print('Now saving')
        with open(os.path.join(meta_dir, 'mean_std.pkl'), 'wb') as f:
            pkl.dump(res, f)
        
        print('Done.')
        
    def save_fps(self):
        """
        Save fps info for the whole dataset, only if all data in this dataset share one fps.
        """
        meta_dir = os.path.join(self.data_root, 'pre_processed', self.dataset_name)
        with open(os.path.join(meta_dir, 'path_meta.pkl'), 'rb') as f:
            meta_data = pkl.load(f)
            
        fps_list = [self.fps for _ in meta_data["imu"]]
        meta_data["fps"] = fps_list
        
        with open(os.path.join(meta_dir, 'path_meta.pkl'), 'wb') as f:
            pkl.dump(meta_data, f)