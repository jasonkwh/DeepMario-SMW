ENCODER ?= cnn
TB_LOG_NAME ?= $(ENCODER)
N_ENVS ?=
BENCH_SECONDS ?= 60
VENV ?= .venv
PYTHON ?= $(shell test -x $(VENV)/bin/python && echo $(VENV)/bin/python || echo python)

.PHONY: default venv install setup train bench check clean tensorboard

default: train

venv:
	@test -x $(VENV)/bin/python || python3 -m venv --system-site-packages $(VENV)
	@echo "venv ready: $(VENV) -> $(PYTHON)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(MAKE) setup

setup:
	$(PYTHON) setup.py

train: setup
	N_ENVS=$(N_ENVS) ENCODER=$(ENCODER) TB_LOG_NAME=$(TB_LOG_NAME) $(PYTHON) train.py

bench:
	N_ENVS=$(N_ENVS) ENCODER=$(ENCODER) $(PYTHON) tools/bench_env.py --seconds $(BENCH_SECONDS)

check:
	$(PYTHON) tools/check_invariants.py

clean:
	rm -rf mario_checkpoints/ mario_best_model/ mario_videos/ mario_logs/

tensorboard:
	tensorboard --logdir mario_logs/tensorboard/
