TB_LOG_NAME ?= mario_ppo
BENCH_ENVS ?= 6
BENCH_SECONDS ?= 60

.PHONY: default install setup train bench clean tensorboard

default: train

install:
	python -m pip install -r requirements.txt
	$(MAKE) setup

setup:
	python setup.py

train: setup
	TB_LOG_NAME=$(TB_LOG_NAME) python train.py

bench:
	python tools/bench_env.py --envs $(BENCH_ENVS) --seconds $(BENCH_SECONDS)

clean:
	rm -rf mario_checkpoints/ mario_best_model/ mario_videos/ mario_logs/

tensorboard:
	tensorboard --logdir mario_logs/tensorboard/
