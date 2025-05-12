import os
import time
import logging
from omegaconf import OmegaConf

def create_logger(cfg, rank, phase='train'):
    if rank != 0:
        return None, ''
    
    # # check resume
    # if cfg.TRAIN.RESUME:
    #     assert os.path.exists(cfg.TRAIN.RESUME), "Resume path is invalid."
    #     return

    # create folder for experimental loggers
    os.makedirs(cfg.EXP_FOLDER, exist_ok=True)

    # parse experimental logger file name
    exp_name = cfg.EXP_NAME
    model_name = cfg.MODEL.target.split('.')[-2]
    logger_dir = os.path.join(cfg.EXP_FOLDER, model_name, exp_name)
    cfg.LOGGER_DIR = logger_dir

    # get timestamp
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    head = '%(asctime)-15s %(message)s'

    # configure logger
    logger = config_logger(cfg, phase, time_str, logger_dir, head)
    assert logger is not None, "Logger is not initialized successfully."

    return logger, time_str

def config_logger(
        cfg,
        phase: str,
        time_str: str,
        logger_dir: str,
        head: str
    ):

    # create experimental logger directory
    cfg.EXP_TIME = str(time_str)
    os.makedirs(logger_dir, exist_ok=True)

    # save current configuration
    config_file = '{}_{}_{}.yaml'.format('config', time_str, phase)
    exp_config_file = os.path.join(logger_dir, config_file)
    OmegaConf.save(config=cfg, f=exp_config_file)

    # setup logger
    log_file = '{}_{}_{}.log'.format('log', time_str, phase)
    log_path = os.path.join(logger_dir, log_file)
    logging.basicConfig(filename=log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(log_path, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)

    return logger

def log_info(rank, logger, info):
    if rank == 0:
        logger.info(info)