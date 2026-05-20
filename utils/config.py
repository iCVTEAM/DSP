import argparse
import logging
import yaml
import os

from .utils import Dict

def load_config():
    parser = argparse.ArgumentParser(description='F-Kakusan Argparser')
    parser.add_argument('--cfg', '--config', type=str, dest='config', default='./configs/test.yaml', required=True, help='Path to the config file.')
    parser.add_argument('-t', '--task-name', type=str, default=None, help='Name of the task.')
    parser.add_argument('-p', '--phase', type=str, choices=["base", "novel"], required=True, help='Select the phase in [base, novel].')
    parser.add_argument('-s', '--shuffle-seed', type=int, default=None, help='Shuffle seed for novel data sampling.')
    parser.add_argument('-m', '--mode', type=str, choices=["train", "infer"], required=True, help='Mode of scripts.')
    parser.add_argument('-r', '--run', type=int, default=None, help='Number of runs of the experiments.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode with small dataset and less workers.')
    parser.add_argument('-c', '--ckpt-steps', type=int, default=None, help='Overwrite ckpt steps.')
    parser.add_argument('-M', '--max_inference_size', type=int, default=None, help='Overwrite max inference size.')
    parser.add_argument('-k', '--k-shot', type=int, default=None, help='Overwrite K Shot.')

    args = parser.parse_args()
    with open(args.config) as file:
        config = Dict(yaml.safe_load(file))
    config.config = args.config
    config.task_name = args.task_name or config.task_name
    config.mode = args.mode
    config.phase = args.phase
    config.run = args.run
    config.debug = args.debug
    config.dataset.novel_settings.shuffle_seed = args.shuffle_seed or config.dataset.novel_settings.shuffle_seed
    config.dataset.novel_settings.k_shot = args.k_shot or config.dataset.novel_settings.k_shot

    if config.phase == 'base':
        config.training |= config.training.base
        config.inference |= config.inference.base
    elif config.phase == 'novel':
        config.training |= config.training.novel
        config.inference |= config.inference.novel

    # config.inference.ckpt_steps = args.ckpt_steps or config.inference.ckpt_steps

    config.inference.ckpt_steps = (
        args.ckpt_steps if args.ckpt_steps is not None else config.inference.ckpt_steps
    )

    config.inference.max_inference_size = (
        args.max_inference_size if args.max_inference_size is not None else config.inference.get('max_inference_size', None)
    )

    return config