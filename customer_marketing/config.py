"""
Marketing "true lift" models settings.

Written by Daniel Steinberg and Lachlan McCalman,
Gradient Institute Ltd. (info@gradientinstitute.org).

Copyright © 2020 Monetary Authority of Singapore

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

# Output settings
SENSITIVE_ATTRIBUTES = ["age", "isfemale", "isforeign"]
COVARIATES_FILE = "model_inputs.csv"
SENSITIVE_FILE = "sensitive_attributes.csv"
OUTCOMES_FILE = "outcomes.csv"
TRUTH_FILE = "truth.csv"
INDEX_LABEL = "ID"


# Randomness settings
RSEED = 42


# Data size
N_PEOPLE = 100000


# Control group proportion
P_CONTROL = 0.6


# Gender
P_FEMALE = 0.4  # proportion of data that are female


# Nationality
P_ISFOREIGN = 0.3  # proportion of the data that are foreign nationals


# Age
MIN_AGE = 18
MEAN_AGE = 45
STD_AGE = 10
FOREIGN_AGE_EFFECT = -5


# Income
BASE_MALE_INCOME = 45000
BASE_FEMALE_INCOME = 35000
FOREIGN_INCOME_EFFECT = -5000
FOREIGN_INCOME_STD = 5000
STD_MALE_INCOME = 10000
STD_FEMALE_INCOME = 15000
MEAN_AGE_INCOME = 1500  # income/years age multiplier
STD_AGE_INCOME = 0  # income std. dev. age multiplier


# Existing products
MANY_PRODUCTS = 5  # number that defines "many" products (max rate)


# Responded to previous offers
PROB_RESPONDED = 0.05


# Feature transformations
FEATURE_TRANSFORMS = {
    "age": lambda x: x,
    "isfemale": lambda x: x,
    "isforeign": lambda x: x,
    "income": lambda x: x,
    "noproducts": lambda x: x,
    "didrespond": lambda x: x
}


# Environmental effects on lift model
NORM_EFFECTS = True  # standardise environmental values going into the model.
LABEL_BIASES = {
    "P": 0.0,
    "ST": 0.5,
    "LC": 1.0,
    "DND": 0.0
}

LABEL_WEIGHTS = {
    "P": {
        "age": 2.0,
        "isfemale": 1.0,
        "isforeign": 0.0,
        "income": 0.0,
        "noproducts": 1.,
        "didrespond": 10.
    },
    "ST": {
        "age": 1.0,
        "isfemale": 0.,
        "isforeign": -1.0,
        "income": 3.,
        "noproducts": 3.,
        "didrespond": 7.
    },
    "LC": {
        "age": -1.,
        "isfemale": 0.0,
        "isforeign": 1.0,
        "income": -2.,
        "noproducts": -2.,
        "didrespond": -3.
    },
    "DND": {
        "age": -2.,
        "isfemale": -1.0,
        "isforeign": 0.0,
        "income": 0.0,
        "noproducts": 1.,
        "didrespond": -10.
    }
}


# Acquired outcomes
PROB_ACQUIRED = 0.90  # The % of those who apply then acquire
FOREIGN_ACQUIRED_MOD = -0.2
MIN_INCOME_ACQUIRED = 5000  # The minimum annual income to be allowed to apply

# Long-term outcome resolutions
PROB_SUCCESS = 0.95  # The % of those who acquire have successful resolution
FEMALE_SUCCESS_MOD = -0.05
