import subprocess
import sys
import json
from pathlib import Path

# Define 6 carefully tuned iterations exploring:
# - Learning rates (1e-5, 2e-5, 3e-5)
# - Epochs (2, 3, 4)
# - Batch sizes (32, 64)
# - Extra hard negative triplet weighting (repeat 2x, 3x, 5x)
# - Base starting checkpoint: intfloat/multilingual-e5-base vs continuous training on v4

ITERATIONS = [
    {
        "iteration": 1,
        "name": "iter1_me5_base_lr2e5_b64_ep3",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 2e-5,
        "epochs": 3,
        "batch_size": 64,
        "extra_repeat": 2,
        "gpu": 0
    },
    {
        "iteration": 2,
        "name": "iter2_me5_base_lr3e5_b64_ep3",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 3e-5,
        "epochs": 3,
        "batch_size": 64,
        "extra_repeat": 3,
        "gpu": 1
    },
    {
        "iteration": 3,
        "name": "iter3_me5_base_lr2e5_b32_ep4",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 2e-5,
        "epochs": 4,
        "batch_size": 32,
        "extra_repeat": 4,
        "gpu": 2
    },
    {
        "iteration": 4,
        "name": "iter4_v4_warm_lr1e5_b64_ep2",
        "base_model": "/home/s0r0qw6/instagram-topic-v2/tuned/multilingual-e5-base-events-real-v4",
        "lr": 1e-5,
        "epochs": 2,
        "batch_size": 64,
        "extra_repeat": 4,
        "gpu": 3
    },
    {
        "iteration": 5,
        "name": "iter5_v4_warm_lr2e5_b64_ep3",
        "base_model": "/home/s0r0qw6/instagram-topic-v2/tuned/multilingual-e5-base-events-real-v4",
        "lr": 2e-5,
        "epochs": 3,
        "batch_size": 64,
        "extra_repeat": 5,
        "gpu": 0
    },
    {
        "iteration": 6,
        "name": "iter6_me5_base_lr15e5_b64_ep4_heavy_repeat",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 1.5e-5,
        "epochs": 4,
        "batch_size": 64,
        "extra_repeat": 5,
        "gpu": 1
    }
]

def run_iter(cfg):
    cmd = [
        "/home/s0r0qw6/model-py-java/.venv/bin/python",
        "train_eval_topic_iterations.py",
        f"--iteration={cfg['iteration']}",
        f"--base-model={cfg['base_model']}",
        f"--output-dir=models/{cfg['name']}",
        f"--epochs={cfg['epochs']}",
        f"--batch-size={cfg['batch_size']}",
        f"--lr={cfg['lr']}",
        f"--extra-repeat={cfg['extra_repeat']}"
    ]
    env_prefix = f"CUDA_VISIBLE_DEVICES={cfg['gpu']}"
    print(f"\n>>> Running Iteration {cfg['iteration']} on GPU {cfg['gpu']}...")
    full_cmd = f"cd /home/s0r0qw6/topic_biencoder_training && {env_prefix} {' '.join(cmd)}"
    subprocess.run(full_cmd, shell=True, check=True)

if __name__ == "__main__":
    for cfg in ITERATIONS:
        run_iter(cfg)
    print("\nAll 6 iterations completed successfully!")
