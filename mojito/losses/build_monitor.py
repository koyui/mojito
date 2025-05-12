import wandb

def build_monitor(cfg, rank, time_stamp: str):
    if rank != 0 or cfg.DEBUG:
        return None
    
    config={
        # meta information
        "datasets": cfg.DATASET.NAME_LIST,
        "seed": cfg.SEED_VALUE,

        # model information
        "epochs": cfg.TRAIN.END_EPOCH,
        "switch_at": cfg.TRAIN.SWITCH_EPOCH,
        "optimizer": cfg.TRAIN.OPTIM.target,
        "learning_rate": cfg.TRAIN.OPTIM.params.lr
    }
    
    run = wandb.init(
        project=(cfg.LOGGER.WANDB.params.project + time_stamp),
        name=cfg.EXP_NAME,
        config=config
    )

    return run