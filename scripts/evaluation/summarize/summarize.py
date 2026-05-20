import os
import argparse
import numpy as np
from collections import defaultdict

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '--config', type=str, default='dsp-dior', help='Config name.')
parser.add_argument('-r', '--run', type=int, default=None, help='Number of runs of the experiments.')
parser.add_argument('-c', '--ckpt', type=int, default=None, help='Ckpt steps.')
parser.add_argument('-k', '--k-shot', type=int, default=None, help='K shot.')
parser.add_argument('-m', '--metrics', nargs="+", type=str)
args = parser.parse_args()

output_path = os.path.join(PROJECT_DIR, 'outputs')
base_path = os.path.join(output_path, args.cfg, 'novel', f'run-{args.run}', f'{args.k_shot}-shot')
episodes = os.listdir(base_path)
metric_dict = defaultdict(list)

metric_path = os.path.join(PROJECT_DIR, 'metrics')
os.makedirs(metric_path, exist_ok=True)
filename = f'{args.cfg}-{args.k_shot}shot-run{args.run}-ckpt{args.ckpt}-metrics-{len(episodes)}.txt'
fileop = open(os.path.join(metric_path, filename), 'w')

invalid_metrics = set()

for episode in episodes:
    metric_path = os.path.join(base_path, episode, f'checkpoint-{args.ckpt}')
    for metric in args.metrics:
        try:
            value = float(open(os.path.join(metric_path, f"{metric}.txt")).read().strip())
            metric_dict[metric].append(value)
        except:
            invalid_metrics.add(metric)
            pass

for metric in args.metrics:
    if metric not in invalid_metrics:
        metric_array = np.array(metric_dict[metric])
        mean, std, N = metric_array.mean(), metric_array.std(ddof=1), len(metric_array)
        CI = 1.96 * std / np.sqrt(N)
        print(f"{metric}: {mean:.4f} ± {CI:.6f}", file=fileop)

fileop.close()