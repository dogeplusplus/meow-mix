.PHONY:test
test:
	pytest tests/

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY:tensorboard
tensorboard:
	tensorboard --logdir=runs
