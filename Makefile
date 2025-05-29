# Makefile will include make test make clean make build make run

# specify desired location for adpy python binary
VENV:= /home/$(USER)/anaconda3/envs/deepsc
PYTHON:= ${VENV}/bin/python

.PHONY: clean test format build install dev precommit sync run

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".egg-info" | xargs rm -rf

activate:
	conda activate ${VENV}

precommit:
	$(PYTHON) -m pip install pre-commit && pre-commit install

test:
	PYTHONPATH=. pytest tests/

format:
	pre-commit run --all-files

sync:
	git pull
	git pull origin main
