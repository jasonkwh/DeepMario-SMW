import os

from encoders.encoder import EncoderType

SEED = 40
TORCH_NUM_THREADS = 2
GAME_NAME = "SuperMarioWorld-Snes-v0"
STATE_NAME = "YoshiIsland1"

_ENCODER_NAME = os.environ.get("ENCODER", EncoderType.CNN.value)
try:
    ENCODER = EncoderType(_ENCODER_NAME)
except ValueError:
    raise ValueError(
        f"ENCODER must be one of {[choice.value for choice in EncoderType]}, "
        f"got {_ENCODER_NAME!r}"
    ) from None

_N_ENVS_OVERRIDE = os.environ.get("N_ENVS", "").strip()
if _N_ENVS_OVERRIDE:
    try:
        N_ENVS = max(1, int(_N_ENVS_OVERRIDE))
    except ValueError:
        raise ValueError(
            f"N_ENVS must be an integer, got {_N_ENVS_OVERRIDE!r}"
        ) from None
else:
    N_ENVS = max(1, len(os.sched_getaffinity(0)))  # usable cores

FRAME_STACK = 4
RUN_NAME = f"{ENCODER.value}-{FRAME_STACK}"
MODEL_NAME = f"mario_ppo_{RUN_NAME}"
TB_LOG_NAME = os.environ.get("TB_LOG_NAME", RUN_NAME)
GAME_FPS = 60
CHECKPOINT_DIR = "./mario_checkpoints/"
BEST_MODEL_SAVE_DIR = f"./mario_best_model/{RUN_NAME}/"
VIDEO_DIR = "./mario_videos/"
LOG_DIR = "./mario_logs/"
TENSORBOARD_LOG_DIR = f"{LOG_DIR}tensorboard/"
MONITOR_FILENAME = f"{LOG_DIR}monitor_{RUN_NAME}"
MONITOR_EVALUATION_FILENAME = f"{LOG_DIR}monitor_evaluation_{RUN_NAME}"
RESUME_FROM = "latest"  # or "best" or "none"

# Training parameters
FRAME_SKIP = (
    4  # Repeat each action for 4 frames to reduce decisions and speed up training
)
VIDEO_RENDER_FPS = GAME_FPS // FRAME_SKIP  # one decision per FRAME_SKIP game frames
OBSERVATION_SHAPE = (84, 96)  # Resize observations to height x width for the CNN input
LEARNING_RATE = 0.00025  # Learning rate for the optimizer
N_STEPS = 2_048  # Steps per environment per update (per policy rollout)
MAX_EPISODE_STEPS = 4_500  # Maximum number of steps per episode
VIDEO_EVERY_N_BEST = 1  # Record a video on every Nth new best model
BATCH_SIZE = 64
ENT_COEF = 0.01  # Entropy coefficient for the loss calculation
TARGET_KL = 0.02  # Early-stops epoch updates if policy moves too fast
N_EPOCHS = 10  # Optimization epochs per rollout
GAMMA = 0.99  # Discount factor for the reward calculation
GAE_LAMBDA = 0.95  # Lambda for the Generalized Advantage Estimation
CLIP_REWARD = 10.0  # Limit normalized reward magnitude
CLIP_RANGE = 0.2  # PPO surrogate clip range
VF_COEF = 0.5  # Value function loss coefficient
MAX_GRAD_NORM = 0.5  # Prevents exploding gradients
TOTAL_TIMESTEPS = 50_000
EVAL_EVERY = 25_000
EVAL_N_ENVS = 4  # Parallel processes for evaluation; use 2 on a 4-core machine
N_EVAL_EPISODES = 20  # Number of episodes to run during each evaluation
N_EVAL_EPISODES_FINAL = 20  # Number of episodes to run during the final evaluation
EVAL_DETERMINISTIC = False
CHECKPOINT_EVERY = 50_000  # Save a checkpoint every 50,000 training timesteps
