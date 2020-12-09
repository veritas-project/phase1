#!make

# #################################################################################################
# Written by Marc-Etienne Brunet,
# Element AI inc. (info@elementai.com).
#
# Copyright © 2020 Monetary Authority of Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# #################################################################################################

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
	jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port $(JUPYTER_PORT)
