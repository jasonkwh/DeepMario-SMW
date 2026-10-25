#!/usr/bin/env python3
"""Measure emulator and policy throughput on this machine.

Answers three questions before committing to an instance size:

* how many environment steps per second one env achieves (single-core speed)
* whether throughput scales with the number of envs (core count)
* how fast the policy forward pass is, on CPU and on GPU

Usage:
    python tools/bench_env.py --envs 6 --seconds 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import (
    ENCODER,
    FRAME_SKIP,
    GAME_NAME,
    SEED,
    STATE_NAME,
)
from encoders import build_encoder, build_policy
from environment import MarioEnvironment
from helpers import get_n_envs


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def print_machine(env_count: int) -> None:
    affinity = len(os.sched_getaffinity(0))
    print("machine")
    print(f"  cpu            {cpu_model()}")
    print(f"  usable cores   {affinity}")
    print(f"  envs requested {env_count}")
    print(f"  torch          {torch.__version__}")
    print(f"  cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu            {torch.cuda.get_device_name(0)}")
    print()


def build_env(env_count: int, augment: bool):
    env = SubprocVecEnv(
        [
            lambda rank=rank: MarioEnvironment(
                GAME_NAME,
                STATE_NAME,
                monitor_filename=None,
                augment=augment,
                seed=SEED + rank,
            )
            for rank in range(env_count)
        ],
        start_method="spawn",
    )
    return build_encoder(ENCODER).wrap_vec_env(env)


def bench_env(env, seconds: float) -> tuple[float, int]:
    obs = env.reset()
    env_count = env.num_envs
    steps = 0
    start = time.perf_counter()
    elapsed = 0.0

    while elapsed < seconds:
        actions = np.stack([env.action_space.sample() for _ in range(env_count)])
        obs, _, _, _ = env.step(actions)
        steps += env_count
        elapsed = time.perf_counter() - start

    steps_per_second = steps / elapsed
    print("environment")
    print(f"  observation    {obs.shape} {obs.dtype}")
    print(f"  wall clock     {elapsed:.1f}s")
    print(f"  steps          {steps}")
    print(f"  steps/s total  {steps_per_second:.1f}")
    print(f"  steps/s per env {steps_per_second / env_count:.1f}")
    print(f"  game seconds/s {steps_per_second * FRAME_SKIP / 60:.2f}x realtime")
    print()
    return steps_per_second, env_count


def bench_policy(env, obs, iterations: int = 200) -> None:
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("policy forward")

    policy_name, policy_kwargs = build_policy(ENCODER)

    for device in devices:
        model = PPO(
            policy_name,
            env,
            device=device,
            n_steps=8,
            batch_size=8,
            policy_kwargs=policy_kwargs,
        )

        for _ in range(10):
            model.predict(obs, deterministic=False)

        start = time.perf_counter()
        for _ in range(iterations):
            model.predict(obs, deterministic=False)
        elapsed = time.perf_counter() - start

        per_call = elapsed / iterations
        print(
            f"  {device:5} {env.num_envs} envs: "
            f"{per_call * 1000:.2f} ms/predict  "
            f"{env.num_envs / per_call:.1f} steps/s"
        )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        type=int,
        default=get_n_envs(),
        help="number of parallel envs (default: usable cores)",
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="disable observation randomisation to isolate its cost",
    )
    parser.add_argument("--no-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_machine(args.envs)

    env = build_env(args.envs, augment=not args.no_augment)
    try:
        steps_per_second, _ = bench_env(env, args.seconds)
        if not args.no_policy:
            obs = env.reset()
            bench_policy(env, obs)
        print(f"summary: {steps_per_second:.1f} env steps/s with {args.envs} envs")
    finally:
        env.close()


if __name__ == "__main__":
    main()
