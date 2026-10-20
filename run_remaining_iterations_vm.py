import subprocess
import sys
import json
from pathlib import Path

# Run iterations 4, 5, 6, 7, 8
ITERATIONS = [
    {
        "iteration": 4,
        "name": "iter4_v4_warm_lr1e5_b64_ep2",
        "base_model": "/home/s0r0qw6/instagram-topic-v2/tuned/multilingual-e5-base-events-real-v4",
        "lr": 1e-5,
        "epochs": 2,
        "batch_size": 64,
        "extra_repeat": 4,
        "gpu": 0
    },
    {
        "iteration": 5,
        "name": "iter5_v4_warm_lr2e5_b64_ep3",
        "base_model": "/home/s0r0qw6/instagram-topic-v2/tuned/multilingual-e5-base-events-real-v4",
        "lr": 2e-5,
        "epochs": 3,
        "batch_size": 64,
        "extra_repeat": 5,
        "gpu": 1
    },
    {
        "iteration": 6,
        "name": "iter6_me5_base_lr15e5_b64_ep4_heavy_repeat",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 1.5e-5,
        "epochs": 4,
        "batch_size": 64,
        "extra_repeat": 5,
        "gpu": 2
    },
    {
        "iteration": 7,
        "name": "iter7_me5_base_lr2e5_b128_ep4_batch128",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 2e-5,
        "epochs": 4,
        "batch_size": 128,
        "extra_repeat": 4,
        "gpu": 3
    },
    {
        "iteration": 8,
        "name": "iter8_me5_base_lr25e5_b64_ep3_balanced",
        "base_model": "intfloat/multilingual-e5-base",
        "lr": 2.5e-5,
        "epochs": 3,
        "batch_size": 64,
        "extra_repeat": 4,
        "gpu": 0
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
    print(f"\n>>> Running Iteration {cfg['iteration']} ({cfg['name']}) on GPU {cfg['gpu']}...")
    full_cmd = f"cd /home/s0r0qw6/topic_biencoder_training && {env_prefix} {' '.join(cmd)}"
    subprocess.run(full_cmd, shell=True, check=True)

if __name__ == "__main__":
    for cfg in ITERATIONS:
        run_iter(cfg)
    print("\nIterations 4 through 8 completed successfully!")
