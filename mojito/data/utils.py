import torch
from torch import Tensor
import math
import torch

from configs.lm.config_base import config

def collate_fn(batch_data, dataset_config):
    # crop or extend to 60 frames, ignore sequences less than 40 frames
    T, min_t = dataset_config.SEQ_LEN, dataset_config.MIN_MOTION_LEN
    collated_motion = []
    collated_imu = []
    collated_fps = []
    for b in batch_data:
        motion_data = b["motion"]["pose"]
        imu_data = b["imu"]
        fps = b["fps"]
        device = motion_data.device
        seq_len, motion_dim = motion_data.shape
        assert seq_len <= T, "ERROR: data length should be less than 60 frames!"
        assert imu_data.shape[0] == seq_len, "Error: frame of motion should be equal to imu"

        if seq_len < min_t:
            continue
        
        collated_fps.append(torch.tensor(fps).to(device))
        if seq_len < T:
            zeros = torch.zeros(T - seq_len, motion_dim).to(device)
            collated_motion.append(torch.cat([motion_data, zeros], dim=0))
            collated_imu.append(torch.cat([imu_data, torch.zeros(T - seq_len, imu_data.shape[-1]).to(device)], dim=0))
        else:
            # seq_len == T
            collated_motion.append(motion_data)
            collated_imu.append(imu_data)
            
    collated_motion = torch.stack(collated_motion, dim=0)
    collated_imu = torch.stack(collated_imu, dim=0)
    collated_fps = torch.stack(collated_fps, dim=0)

    return {"motion": collated_motion, "imu": collated_imu, "fps": collated_fps}

def placeholder_fulfill(prompt: str, length: int, imu_string: str, text: str, question: str = ""):
    seconds = math.floor(length / config.dataset.framerate)

    prompt = (
        prompt.replace("<Caption_Placeholder>", text)
        .replace("<Question_Placeholder>", question)
        .replace("<Imu_Placeholder>", imu_string)
        .replace("<Frame_Placeholder>", f"{length}")
        .replace("<Second_Placeholder>", "%.1f" % seconds)
    )

    return prompt

def template_fulfill(task, length, imu_string, text, question, stage="test"):
    input_template = task["user"]
    output_template = task["assistant"]
    inputs = placeholder_fulfill(input_template, length, imu_string, text, question)
    outputs = placeholder_fulfill(output_template, length, imu_string, text)

    return inputs, outputs


def token_to_string(token: Tensor, length: int, token_type: str = "motion"):
    token_i = token.cpu() if token.device.type == "cuda" else token
    token_list = token_i.tolist()[:length]
    token_string = "".join([f"<{token_type}_id_{int(i)}>" for i in token_list])
    return token_string


def collate_fn_lm_pretrain(batch_data, tokenizer):
    """
    Args(batch_data):
        list of motions, texts. [{imu_token, text, question, fps}, ...]\n
        motions: motion token ids in tensor
    Output:
    """
    task = config.dataset.task

    system_token = tokenizer(
        f"<|im_start|>system\n{task['system']}:<|im_end|>\n",
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = []
    attention_masks = []
    labels = []

    for batch in batch_data:
        imu_token = batch["imu_token"]
        text = batch["text"]
        fps = batch["fps"]
        question = batch["question"]
        length = len(imu_token)

        imu_string = token_to_string(imu_token, length, token_type="imu")

        config.dataset.framerate = fps
        inputs, outputs = template_fulfill(task, length, imu_string, text, question)

        input_token = tokenizer(
            f"<|im_start|>user\n{inputs}<|im_end|>\n",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        output_token = tokenizer(
            f"<|im_start|>assistant\n{outputs}<|im_end|>",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_id = system_token.input_ids[0].tolist() + input_token.input_ids[0].tolist()
        label = [-100] * (len(input_id)) + output_token.input_ids[0].tolist()
        input_id = input_id + output_token.input_ids[0].tolist()

        attention_mask = (
            system_token.attention_mask[0].tolist()
            + input_token.attention_mask[0].tolist()
            + output_token.attention_mask[0].tolist()
        )

        if len(input_id) > config.dataset.max_token_len:
            input_id = input_id[: config.dataset.max_token_len]
            label = label[: config.dataset.max_token_len]
            attention_mask = attention_mask[: config.dataset.max_token_len]
        elif len(input_id) < config.dataset.max_token_len:
            input_id = [tokenizer.pad_token_id] * (config.dataset.max_token_len - len(input_id)) + input_id
            label = [-100] * (config.dataset.max_token_len - len(label)) + label
            attention_mask = [0] * (config.dataset.max_token_len - len(attention_mask)) + attention_mask

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels),
    }