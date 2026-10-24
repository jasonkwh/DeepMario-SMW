# DeepMario-SMW-Vision-E2E-PPO

PPO on Super Mario World, end to end from pixels. The goal is one policy that
clears the whole game; Yoshi's Island 1 is the current stage, set by `STATE_NAME`
in `config.py`. A CNN reads stacked 84x96 grayscale frames and picks one of 19
discrete button combinations.

## Install

- Python 3.10+
- `stable-retro==1.0.1`, `stable-baselines3==2.9.0`, `gymnasium==1.3.0`
- A Super Mario World ROM in the repo root as `.sfc` (git-ignored, never committed)

```bash
make install     # pip install + patch game data
make train       # patch data.json, then train.py
make tensorboard
```

`setup.py` patches the installed game `data.json` with the RAM fields the reward
needs, so it runs before every training start.

## Commands

| Command | |
| --- | --- |
| `make` / `make train` | Patch game data, then train |
| `make bench` | Emulator and policy throughput |
| `make tensorboard` | Serve `mario_logs/tensorboard/` |
| `make clean` | Delete checkpoints, best model, videos, logs |
| `make install` | Requirements + game data |

```bash
make train TB_LOG_NAME=t4-aug              # separate TensorBoard run
make bench BENCH_ENVS=2 BENCH_SECONDS=30   # smaller machines
```

## Configuration

All knobs live in `config.py`, each with a short comment. Env count is derived
there rather than set: usable cores minus `RESERVED_CPU_CORES` (6 on 8 vCPU).

## Environment

Order is deliberate; it cannot be shuffled.

```
stable_retro.make(SuperMarioWorld-Snes-v0, state=YoshiIsland1)
  ActionWrapper             19 discrete button combos
  RewardWrapper             shaped reward
  MaxAndSkipEnv             repeat action for FRAME_SKIP frames
  GrayscaleObservation      keep_dim
  ResizeObservation         84 x 96
  ObservationAugmentation   training only: photometric + translation jitter
  TimeLimit                 MAX_EPISODE_STEPS
  RecordVideo               only when requested
  ActionHistory             3 constant planes of recent actions
  Monitor                   episode stats -> CSV
```

One observation is `(4, 84, 96)` - grayscale frame plus 3 action planes - and
`VecFrameStack` makes it `(16, 84, 96)`. Action indices are scaled to 0-255 so the
stock `CnnPolicy` works without a custom extractor.

## Network

Stock `NatureCNN`: conv 32x8x8/4, 64x4x4/2, 64x3x3/1, flatten, linear 512, then
PPO's policy (19 logits) and value heads. Input channels are read from the
observation space. Deliberately small: this is the E2E baseline.

## Reward

Values are pre-normalisation; `VecNormalize` normalises and clips them.

| Rule | Value |
| --- | --- |
| Level completed | +100 |
| Time ran out | -10 |
| Lost a life / last life | -5 / -20 |
| Coin | +0.2 |
| Other score increase | +0.05 |
| Power-up big / cape / fire | +0.1 / +0.2 / +0.2 |
| Lost a power-up | -0.1 |
| Star | +0.3 |
| Timer per step | -0.00025 (doubles below 100) |
| Progress per pixel of x | +0.005 |

Rules fire in the order listed in `wrappers/reward.py`; the first termination rule
that triggers ends the episode.

## Evaluation

Stock `EvalCallback` bookkeeping plus two metrics that answer whether the policy
gets further:

- `eval/mean_max_x` - mean furthest x per episode
- `eval/clear_rate` - share of episodes that clear the level

The final evaluation prints reward, max_x, clear rate and episode length, and
records them as `final/*`.

## Benchmark

```bash
make bench
python tools/bench_env.py --envs 6 --seconds 60 --no-augment
```

Reports steps/s per env, whether throughput scales with env count, and policy
forward latency on CPU/GPU. `--no-augment` isolates the randomisation cost.

## Outputs

- `mario_checkpoints/mario_ppo_<steps>_steps.zip` + `mario_ppo_vecnormalize_<steps>_steps.pkl`
- `mario_best_model/best_model.zip` + its VecNormalize stats
- `mario_logs/monitor-<rank>.csv`, `monitor_evaluation-<rank>.csv`
- `mario_videos/` - written on every new best model
- `mario_logs/tensorboard/<TB_LOG_NAME>/`

`RESUME_FROM` picks the checkpoint. With no checkpoint, stale best-model files are
removed first so a fresh run cannot inherit an old best. Reuse `TB_LOG_NAME` to
keep resuming runs in one TensorBoard directory.

## Game data

`setup.py` adds these fields to `data.json`: `x`, `y` (position, drives progress),
`powerup` (00 small, 01 big, 02 cape, 03 fire), `item` (item box), `timer_hundreds`
/ `timer_tens` / `timer_ones`, `level_end_timer` (non-zero once completion starts),
`star_timer`.

## Known limitations

- **Evaluation is deterministic.** stable-retro restores the savestate on reset,
  RNG included, so a deterministic policy replays the same trajectory every
  episode: episodes add no variance and `clear_rate` is effectively 0 or 1.
  Fixing it needs a randomised start or a single-episode evaluation.
- **No upward actions.** `UP` combos are commented out, so no swimming up and no
  dismounting Yoshi. Acceptable on this stage, required later.
- **The eval env carries a no-op `VecNormalize`.** Required, because SB3's
  `sync_envs_normalization` asserts matching wrapper stacks.
- **`is_dead` is unverified.** It assumes the lives counter reads -1 on the final
  death frame.

`version.yaml` is bumped by CI on every push to `main`.
