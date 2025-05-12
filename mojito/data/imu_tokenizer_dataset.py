import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import os
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy

class MojitoIMUTokenizerDataset(torch.utils.data.Dataset):
    def __init__(self, phase, **kwargs):
        super().__init__()

        self.dataset_config = kwargs["cfg"]["DATASET"]
        self.data_root = self.dataset_config.ROOT
        self.all_paths = self.load_paths(self.dataset_config.NAME_LIST)
        self.phase = phase
        self.curr_iter = 0

        # indices mask for normalization
        self.motion_eu_mask = [
            0, 1, 2,    # root translation
            3, 4, 5,    # root linear velocity
            12, 13, 14  # root angular velocity
        ]
        self.motion_eu_mask.extend([i for i in range(141, 267)])
        self.imu_eu_mask = list(range(36, 72))

        # load mean-std determined on training dataset
        if self.phase == "train":
            self.load_statistics()
            self.all_data = self.load_all_data()

    def __len__(self):
        if self.phase == "train":
            return len(self.all_data)
        else:
            return len(self.all_paths["motion"])    

    def __next__(self):
        if self.curr_iter >= len(self):
            self.curr_iter = 0
            raise StopIteration()
        else:
            single_data = self.__getitem__(self.curr_iter)
            self.curr_iter += 1
        
        return single_data

    def __getitem__(self, idx):
        if self.phase == "train":
            item = deepcopy(self.all_data[idx])
        else:
            item = self.load_data(idx)

        # move data to gpu devices
        for k in item["motion"].keys():
            item["motion"][k] = item["motion"][k].to(self.device)
        item["imu"] = item["imu"].to(self.device)

        return item

    def set_device(self, device):
        self.device = device
        if self.phase == "train":
            self.motion_mean = self.motion_mean.to(device)
            self.motion_std = self.motion_std.to(device)
            self.imu_mean = self.imu_mean.to(device)
            self.imu_std = self.imu_std.to(device)

    def set_statistics(self, motion_s1, motion_s2, imu_s1, imu_s2):
        assert self.phase == "test"

        self.motion_mean = motion_s1
        self.motion_std = motion_s2
        self.imu_mean = imu_s1
        self.imu_std = imu_s2
    
    def load_statistics(self):
        assert self.phase == "train"

        motion_mean_stack, motion_std_stack, motion_frames_stack = [], [], []
        imu_mean_stack, imu_std_stack, imu_frames_stack = [], [], []
        
        for dn in self.dataset_config.NAME_LIST:
            dataset_path = os.path.join(self.data_root, dn)
            with open(os.path.join(dataset_path, 'mean_std.pkl'), 'rb') as f:
                stat_dict = pkl.load(f)
            motion_mean_stack.append(torch.from_numpy(stat_dict['motion_mean']).float().unsqueeze(0))
            motion_std_stack.append(torch.from_numpy(stat_dict['motion_std']).float().unsqueeze(0))
            motion_frames_stack.append(stat_dict['motion_frames'])
            imu_mean_stack.append(torch.from_numpy(stat_dict['imu_mean']).float().unsqueeze(0))
            imu_std_stack.append(torch.from_numpy(stat_dict['imu_std']).float().unsqueeze(0))
            imu_frames_stack.append(stat_dict['imu_frames'])
        
        motion_mean_stack = torch.cat(motion_mean_stack, dim=0)
        motion_std_stack = torch.cat(motion_std_stack, dim=0)
        motion_frames_stack = torch.tensor(motion_frames_stack).unsqueeze(-1).float()
        imu_mean_stack = torch.cat(imu_mean_stack, dim=0)
        imu_std_stack = torch.cat(imu_std_stack, dim=0)
        imu_frames_stack = torch.tensor(imu_frames_stack).unsqueeze(-1).float()
        
        assert motion_mean_stack.shape[-1] == len(self.motion_eu_mask)
        assert imu_mean_stack.shape[-1] == len(self.imu_eu_mask)
        
        self.motion_mean, self.motion_std = self.merge_mean_std(motion_mean_stack, motion_std_stack, motion_frames_stack)
        self.imu_mean, self.imu_std = self.merge_mean_std(imu_mean_stack, imu_std_stack, imu_frames_stack)
        
    def load_paths(self, name_list):
        self.dataset_meta = {}
        all_paths = {"motion": [], "imu": [], "fps": []}

        for dn in name_list:
            dataset_path = os.path.join(self.data_root, dn)
            with open(os.path.join(dataset_path, "path_meta.pkl"), "rb") as f:
                path_meta = pkl.load(f)
            self.dataset_meta[dn] = (len(all_paths["motion"]), len(path_meta["motion"]))

            for k, _ in all_paths.items():
                all_paths[k].extend(path_meta[k])
            
        assert len(all_paths["motion"]) == len(all_paths["imu"]), "Inconsistent number of paths."
        assert len(all_paths["fps"]) == len(all_paths["imu"]), "Inconsistent number of fps list."
        return all_paths
    
    def load_data(self, idx):
        # load data
        with open(self.all_paths["motion"][idx], "rb") as f:
            data = pkl.load(f)
            motion = {
                "shape": torch.from_numpy(data["shape"]).float(),
                "pose": None
            }
        motion["pose"] = torch.from_numpy(data["pose_humor_repr"]).float()
        with open(self.all_paths["imu"][idx], "rb") as f:
            imu_data_flatten = pkl.load(f)
        imu_data_flatten = torch.from_numpy(imu_data_flatten).float()
        assert motion["pose"].shape[0] == imu_data_flatten.shape[0] - 1
        imu_data_flatten = imu_data_flatten[:-1, :]
            
        # normalize data part in euclidean space
        motion["pose"][:, self.motion_eu_mask] = self.normalize_motion(motion["pose"][:, self.motion_eu_mask])
        imu_data_flatten[:, self.imu_eu_mask] = self.normalize_imu(imu_data_flatten[:, self.imu_eu_mask])
            
        # get fps for the item
        return {"motion": motion, "imu": imu_data_flatten, "fps": self.all_paths["fps"][idx]}
    
    def load_all_data(self):
        all_data = []
        for i in tqdm(range(len(self.all_paths["motion"])), desc="loading data"):
            single_data = self.load_data(i)
            motion_data = single_data["motion"]
            imu_data = single_data["imu"]
            fps_data = single_data["fps"]

            # in training vae stage, split each sequence into chunks
            chunks = motion_data["pose"].shape[0] // self.dataset_config.SEQ_LEN
            if chunks == 0:
                all_data.append(single_data)
            else:
                for x in range(chunks):
                    all_data.append(
                        {
                            "motion": {
                                "shape": motion_data["shape"],
                                "pose": motion_data["pose"][x*self.dataset_config.SEQ_LEN:(x+1)*self.dataset_config.SEQ_LEN]
                            },
                            "imu": imu_data[x*self.dataset_config.SEQ_LEN:(x+1)*self.dataset_config.SEQ_LEN, ...],
                            "fps": fps_data
                        }
                    )

        return all_data
    
    def normalize_motion(self, eu_part):
        assert eu_part.shape[1] == len(self.motion_eu_mask)
        
        return (eu_part - self.motion_mean) / self.motion_std
    
    def normalize_imu(self, eu_part):
        assert eu_part.shape[1] == len(self.imu_eu_mask)
        
        return (eu_part - self.imu_mean) / self.imu_std

    def denormalize_motion(self, normed_eu):
        return (normed_eu.cpu() * self.motion_std + self.motion_mean).to(normed_eu.device)

    def denormalize_imu(self, normed_eu):
        return (normed_eu.cpu() * self.imu_std + self.imu_mean).to(normed_eu.device)        
    
    def merge_mean_std(self, means, stds, frames):
        """
        merge mean and std across different datasets
        """
        total_frames = torch.sum(frames)
        merged_mean = torch.sum(means * frames, dim=0) / total_frames
        # Var(X) = E(X^2) - [E(X)]^2
        merged_var = torch.sum((stds ** 2 + means ** 2) * frames, dim=0) / total_frames - merged_mean ** 2
        merged_std = torch.sqrt(merged_var)
        
        return merged_mean, merged_std
