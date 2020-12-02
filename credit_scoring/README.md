# FEAT Fairness Methodology: Credit Scoring Assessment

This repository provide a Jupyter notebook and supporting code for the
credit scoring case study assessment in the FEAT Fairness Principles Assessment Case
Studies Document. Please see Section 3 of that document for the credit scoring
case study assessment itself.

This code should be considered an *alpha* pre-release version.
It comes with **absolutely no warranty**.
It is not intended for use in production,
or for assessing high risk AIDA systems under the methodology.

This work was undertaken as part of the Veritas initiative commissioned by the
Monetary Authority of Singapore, whose goal is to accelerate the adoption of
responsible Artificial Intelligence and Data Analytics (AIDA) in the financial
services industry.

## Contents

The key files and folders are the following:

- `credit_fairness.ipynb`: generates the analysis for the assessment
- `utils/`: Python code providing functionality to support the above notebook
- `data/`: Proxy dataset supplied on which the analysis is conducted

## Setup and Usage

### Unix-based OS with Docker
These instructions assume you are on a unix-based OS with `gnu-make`, `bash` and Docker installed.
You can simply run

```bash
make docker.jupyter
```
then navigate to the url output in the terminal.

### Others (usage directly with Conda)
First install
[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
or Miniconda, then run

```bash
conda env update -f environment.yaml
conda activate veritas
jupyter notebook
```
you should be redirected to a Jupyter server in your browser,
if not, navigate to the url output in the terminal.

## Copyright

Written by Marc-Etienne Brunet & Hardeep Arora, Element AI Inc.

Copyright © 2020 Monetary Authority of Singapore.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
