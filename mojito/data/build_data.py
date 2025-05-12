import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

from mojito.config import instantiate_from_config
from mojito.data.utils import collate_fn

def build_data(cfg, device=torch.device('cpu'), phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}

    dataset = instantiate_from_config(data_config)
    dataset.set_device(device)
    if phase == 'test':
        return dataset
    else:
        dataloader_options = {}
        dataloader_options["batch_size"] = cfg.TRAIN.BATCH_SIZE
        dataloader_options["num_workers"] = cfg.TRAIN.NUM_WORKERS
        dataloader_options["drop_last"] = cfg.TRAIN.DROP_LAST
        dataloader_options["persistent_workers"] = cfg.TRAIN.PERSISTENT_WORKERS
        dataloader_options["collate_fn"] = lambda batch: collate_fn(batch, cfg.DATASET)
        if cfg.TRAIN.USE_DDP:
            dataloader_options["sampler"] = DistributedSampler(dataset)
        
        return DataLoader(dataset, **dataloader_options)