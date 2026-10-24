.PHONY: default install setup train clean tensorboard

default: train

install:
	python -m pip install -r requirements.txt
	$(MAKE) setup

setup:
	python setup.py

train: setup
	python train.py

clean:
	rm -rf mario_checkpoints/ mario_best_model/ mario_videos/ mario_logs/

tensorboard:
	tensorboard --logdir mario_logs/tensorboard/
