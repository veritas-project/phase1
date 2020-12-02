#!make

SHELL = /bin/bash


# Docker configuration
DOCKER_IMG = veritas-credit-fairness:v1
WORKSPACE = /workspace
WORKSPACE_VOL = $(PWD):$(WORKSPACE)
CONDA_ENV = veritas  # this should match the name in environment.yaml


# Jupyer configuration
JUPYTER_TOKEN = my_secret_token
JUPYTER_PORT = 8083


# Adjustable Docker parameters
MEM ?= 4
CPU ?= 1
NAME ?= veritas

.PHONY: docker.build
docker.build:
	docker build --build-arg WORKSPACE=$(WORKSPACE) -t $(DOCKER_IMG) .


.PHONY: docker.jupyter
docker.jupyter: docker.build
	docker run \
	--rm -i -t \
	-p $(JUPYTER_PORT):$(JUPYTER_PORT) \
	-v $(WORKSPACE_VOL) \
	-w $(WORKSPACE) \
	-e JUPYTER_TOKEN=$(JUPYTER_TOKEN) \
	--memory $(MEM)G \
	--cpus $(CPU) \
	--name $(NAME) \
	$(DOCKER_IMG) \
	bash -c "echo Starting server at http://localhost:$(JUPYTER_PORT)/lab?token=$(JUPYTER_TOKEN);\
	jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port $(JUPYTER_PORT)"


# Run without Docker
.PHONY: jupyter
jupyter:
	@ echo Starting server at http://localhost:$(JUPYTER_PORT)/lab?token=$(JUPYTER_TOKEN)
	jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port $(JUPYTER_PORT)
