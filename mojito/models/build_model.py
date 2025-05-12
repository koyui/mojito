from omegaconf import OmegaConf

from mojito.config import instantiate_from_config

def build_model(cfg, phase='train'):
    model_config = OmegaConf.to_container(cfg.MODEL, resolve=True)
    model_config["params"]["cfg"] = cfg
    model_config["params"]["phase"] = phase
    return instantiate_from_config(model_config)