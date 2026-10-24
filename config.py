import os

SEED = 40
RESERVED_CPU_CORES = 2
TORCH_NUM_THREADS = 2
GAME_NAME = "SuperMarioWorld-Snes-v0"
STATE_NAME = "YoshiIsland1"
MODEL_NAME = "mario_ppo"
TB_LOG_NAME = os.environ.get("TB_LOG_NAME", MODEL_NAME)
GAME_FPS = 60
CHECKPOINT_DIR = "./mario_checkpoints/"
BEST_MODEL_SAVE_DIR = "./mario_best_model/"
VIDEO_DIR = "./mario_videos/"
LOG_DIR = "./mario_logs/"
TENSORBOARD_LOG_DIR = f"{LOG_DIR}tensorboard/"
MONITOR_FILENAME = f"{LOG_DIR}monitor.csv"
MONITOR_EVALUATION_FILENAME = f"{LOG_DIR}monitor_evaluation.csv"
RESUME_FROM = "latest"  # or "best" or "none"

# Training parameters
FRAME_SKIP = (
    4  # Repeat each action for 4 frames to reduce decisions and speed up training
)
VIDEO_RENDER_FPS = GAME_FPS // FRAME_SKIP  # one decision per FRAME_SKIP game frames
OBSERVATION_SHAPE = (84, 96)  # Resize observations to height x width for the CNN input
LEARNING_RATE = 0.00025  # Learning rate for the optimizer
N_STEPS = (
    2_048  # Number of steps to run for each environment per update (per policy rollout)
)
MAX_EPISODE_STEPS = 4_500  # Maximum number of steps per episode
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
FRAME_STACK = 4  # Number of frames to stack
TOTAL_TIMESTEPS = 5_000_000
EVAL_EVERY = 500_000
EVAL_N_ENVS = 4  # Parallel processes for evaluation; use 2 on a 4-core machine
N_EVAL_EPISODES = 10  # Number of episodes to run during each evaluation
N_EVAL_EPISODES_FINAL = 20  # Number of episodes to run during the final evaluation
CHECKPOINT_EVERY = 50_000  # Save a checkpoint every 50,000 training timesteps
