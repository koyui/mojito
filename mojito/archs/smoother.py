# script adapted from https://github.com/cure-lab/SmoothNet/blob/main/lib/visualize/visualize_smpl.py

import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import *

from mojito.config import instantiate_from_config

def slide_window_to_sequence(slide_window,window_step,window_size):
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)

    return sequence

class SmoothNetWrapper(nn.Module):

    def __init__(
        self,
        model,
        body_representation, 
        slide_window_size, 
        slide_window_step,
        use_6d_smpl
    ):
        super().__init__()

        self.model = instantiate_from_config(model)
        self.model.eval()

        self.use_6d_smpl = use_6d_smpl

        self.body_representation = body_representation

        self.slide_window_size = slide_window_size
        self.slide_window_step = slide_window_step

        if self.body_representation == 'smpl':
            if self.use_6d_smpl:
                self.input_dimension = 6 * 24
            else:
                self.input_dimension = 3 * 24

    @torch.no_grad()
    def smooth_smpl(self, data_pred):
        if self.use_6d_smpl:
            data_pred = matrix_to_rotation_6d(
                axis_angle_to_matrix(data_pred.contiguous().view(-1, 3))
            ).contiguous().view(-1, self.input_dimension)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, 
            ((data_len - self.slide_window_size) // self.slide_window_step+1, self.slide_window_size, self.input_dimension),
            (self.slide_window_step * self.input_dimension, self.input_dimension, 1), 
            storage_offset=0
        ).view(-1, self.slide_window_size, self.input_dimension)

        data_pred_window=data_pred_window.permute(0, 2, 1)
        predicted_pos= self.model(data_pred_window)
        data_pred_window=data_pred_window.permute(0, 2, 1)
        predicted_pos=predicted_pos.permute(0, 2, 1)

        predicted_pos = slide_window_to_sequence(predicted_pos, self.slide_window_step, self.slide_window_size).view(-1, self.input_dimension)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :]


        if self.use_6d_smpl:
            data_pred = matrix_to_axis_angle(rotation_6d_to_matrix(data_pred.view(-1, 6)))
            predicted_pos = matrix_to_axis_angle(rotation_6d_to_matrix(predicted_pos.view(-1, 6)))
        data_pred = data_pred.view(-1, 24*3)
        predicted_pos = predicted_pos.view(-1, 24*3)
        
        return predicted_pos