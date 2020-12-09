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

FROM continuumio/miniconda3:4.8.2

ARG WORKSPACE=/workspace
WORKDIR $WORKSPACE

COPY environment.yaml .

RUN conda env update -n base -f environment.yaml && \
    conda clean -ya
