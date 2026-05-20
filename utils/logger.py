import os
import logging
from datetime import datetime
from accelerate.logging import get_logger


def set_logger(config, accelerator, prefix=None):
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.logger.addHandler(console)

    if accelerator.is_main_process:
        logging_path = os.path.join(config.ckpt_dir, config.task_name, 'logs')
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        log_file = os.path.join(logging_path, f"{f'[{prefix}]'}{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.logger.addHandler(fh)

    logger.info(accelerator.state, main_process_only=False)

    return logger