"""
Simulation code for true lift marketing model.

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

References
----------
[1] Kane, K., Lo, V.S.Y., Zheng, J., 2014. Mining for the truly responsive
    customers and prospects using true-lift modeling: Comparison of new and
    existing methods. J Market Anal 2, 218–238.
    https://doi.org/10.1057/jma.2014.18
"""

import numpy as np
import pandas as pd
import pathlib
from os import path
from typing import Union, Tuple
from scipy.special import softmax
import firegrad.simulation.config as cfg


RANDOM_STATE = np.random.RandomState(cfg.RSEED)


#
# Simulation functions
#

def simulate_environment(rs: Union[np.random.RandomState, int, None]=None) \
        -> pd.DataFrame:
    """Simulate the environment for the uplift model."""
    rstate = _manage_random(rs)

    # Generate gender
    isfemale = rstate.binomial(n=1, p=cfg.P_FEMALE, size=cfg.N_PEOPLE)

    # Generate foreign nationals
    isforeign = rstate.binomial(n=1, p=cfg.P_ISFOREIGN, size=cfg.N_PEOPLE)

    # Generate ages
    mean_age = cfg.MEAN_AGE + cfg.FOREIGN_AGE_EFFECT * isforeign
    age = np.maximum(
        cfg.MIN_AGE,
        rstate.normal(loc=mean_age, scale=cfg.STD_AGE, size=cfg.N_PEOPLE)
    )

    # Generate income
    mean_income = cfg.BASE_FEMALE_INCOME * isfemale \
        + cfg.BASE_MALE_INCOME * (1 - isfemale) \
        + cfg.FOREIGN_INCOME_EFFECT * isforeign \
        + cfg.MEAN_AGE * (age - cfg.MIN_AGE)
    std_income = cfg.STD_FEMALE_INCOME * isfemale \
        + cfg.STD_MALE_INCOME * (1 - isfemale) \
        + cfg.FOREIGN_INCOME_STD * isforeign \
        + cfg.STD_AGE * (age - cfg.MIN_AGE)
    income = np.maximum(0., rstate.normal(loc=mean_income, scale=std_income))

    # Generate number of products
    product_rate = income * cfg.MANY_PRODUCTS / np.max(income)
    noproducts = rstate.poisson(lam=product_rate)

    # Generate previous response to campaigns
    didrespond = rstate.binomial(n=1, p=cfg.PROB_RESPONDED, size=cfg.N_PEOPLE)

    # Concatenate all data
    environment = pd.DataFrame({
        "age": age,
        "isfemale": isfemale,
        "isforeign": isforeign,
        "income": income,
        "noproducts": noproducts,
        "didrespond": didrespond
    })

    return environment


def transform_features(environment: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transforms."""
    for col in environment.columns:
        environment[col] = cfg.FEATURE_TRANSFORMS[col](environment[col])

    return environment


def simulate_truth(
        environment: pd.DataFrame,
        rs: Union[np.random.RandomState, int, None]=None
) -> pd.DataFrame:
    """Draw samples of the label."""
    rstate = _manage_random(rs)

    weights = pd.DataFrame(cfg.LABEL_WEIGHTS)
    biases = pd.Series(cfg.LABEL_BIASES)
    effects = environment.dot(weights)
    if cfg.NORM_EFFECTS:
        effects -= effects.mean(axis=0)
        effects /= effects.std(axis=0)

    factors = effects + biases
    probabilites = softmax(factors.values, axis=1)
    draws = np.vstack([rstate.multinomial(n=1, pvals=p) for p in probabilites])
    truth = pd.DataFrame(data=draws.astype(bool), columns=effects.columns)

    # Add in the real lift score
    cols = list(effects.columns)
    lift = probabilites[:, cols.index("P")] \
        - probabilites[:, cols.index("DND")]
    truth["lift"] = pd.Series(lift)

    return truth


def experimental_outcomes(
    truth: pd.DataFrame,
    rs: Union[np.random.RandomState, int, None]=None
) -> pd.DataFrame:
    """Create experimental outcomes based on the truth.

    This is using [1] to relate how the truth is expressed in randomised
    control trials.
    """
    rstate = _manage_random(rs)

    names = ["TN", "TR", "CN", "CR", "incontrol"]
    N, D = len(truth), len(names)
    outcomes = pd.DataFrame(data=np.zeros((N, D), dtype=bool), columns=names)

    # Persuadables are CN and TR
    outcomes.loc[truth["P"], "TR"] = True
    outcomes.loc[truth["P"], "CN"] = True

    # Do not disturbs are CR and TN
    outcomes.loc[truth["DND"], "TN"] = True
    outcomes.loc[truth["DND"], "CR"] = True

    # Lost causes are TN and CN
    outcomes.loc[truth["LC"], "TN"] = True
    outcomes.loc[truth["LC"], "CN"] = True

    # Sure things are TR and CR
    outcomes.loc[truth["ST"], "TR"] = True
    outcomes.loc[truth["ST"], "CR"] = True

    # Draw if someone was in the control or treatment groups
    incontrol = rstate.binomial(n=1, p=cfg.P_CONTROL, size=cfg.N_PEOPLE)
    incontrol = incontrol.astype(bool)
    outcomes["incontrol"] = incontrol

    # Now set treatment responses to false for those in control
    outcomes.loc[incontrol, "TR"] = False
    outcomes.loc[incontrol, "TN"] = False

    # Now set control responses to false for those in treatment
    outcomes.loc[~incontrol, "CR"] = False
    outcomes.loc[~incontrol, "CN"] = False

    return outcomes


def product_outcomes(
    truth: pd.DataFrame,
    environment: pd.DataFrame,
    all_selected: bool,
    rs: Union[np.random.RandomState, int, None]=None
) -> pd.DataFrame:
    """Create applied, acquired and long term resolution outcomes."""
    rstate = _manage_random(rs)

    # Who applied
    if all_selected:
        applied = np.logical_or(truth["P"], truth["ST"]).astype(int)
    else:
        applied = np.logical_or(truth["DND"], truth["ST"]).astype(int)

    # Who acquired out of those who applied
    p_acquired = applied * (cfg.PROB_ACQUIRED + cfg.FOREIGN_ACQUIRED_MOD
                            * environment.isforeign)
    acquired = rstate.binomial(n=1, p=p_acquired, size=cfg.N_PEOPLE)
    acquired[environment.income < cfg.MIN_INCOME_ACQUIRED] = 0

    # Who successfully resolved out of those who acquired
    p_success = acquired * (cfg.PROB_SUCCESS + cfg.FOREIGN_ACQUIRED_MOD
                            * environment.isfemale)
    success = rstate.binomial(n=1, p=p_success, size=cfg.N_PEOPLE)

    prefix = "s" if all_selected else "ns"
    all_outcomes = pd.DataFrame(data={
        f"{prefix}_applied": applied,
        f"{prefix}_acquired": acquired,
        f"{prefix}_success": success
    })
    return all_outcomes


def simulate() \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the simulation to generate the true lift data."""
    environment = simulate_environment()
    environment = transform_features(environment)
    truth = simulate_truth(environment)
    ex_outcomes = experimental_outcomes(truth)
    pr_outcomes_s = product_outcomes(truth, environment, True)
    pr_outcomes_ns = product_outcomes(truth, environment, False)

    # Split sensitive attributes from covariates
    sensitives = environment[cfg.SENSITIVE_ATTRIBUTES]
    covariates = environment.drop(columns=cfg.SENSITIVE_ATTRIBUTES)

    # Concat all outcomes
    all_outcomes = pd.concat(
        (ex_outcomes, pr_outcomes_s, pr_outcomes_ns),
        axis=1
    )

    return covariates, sensitives, all_outcomes, truth


#
# Module helpers
#

def _manage_random(random_state: Union[int, np.random.RandomState, None]) \
        -> np.random.RandomState:
    """Return a random state given a seed, None, or a RandomState."""
    if random_state is None:
        return RANDOM_STATE

    if isinstance(random_state, int):
        rstate = np.random.RandomState(random_state)
        return rstate

    if isinstance(random_state, np.random.RandomState):
        return random_state

    raise ValueError("random_state has to be None, and int, or a RandomState.")


#
# Make dataset files and helper functions
#

def load_data(force_run: bool=False) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the simulation data, run the simulation if no data exists."""
    datapath = pathlib.Path(__file__).parent.absolute()
    files = [cfg.COVARIATES_FILE, cfg.SENSITIVE_FILE, cfg.OUTCOMES_FILE,
             cfg.TRUTH_FILE]
    paths = [path.join(datapath, f) for f in files]

    if force_run or not all(path.isfile(f) for f in paths):
        run_simulation()

    # load
    data = tuple(pd.read_csv(f, index_col=cfg.INDEX_LABEL) for f in paths)
    return data


def outcomes2labels(outcomes: pd.DataFrame, sensitives: pd.DataFrame) \
        -> Tuple[pd.Series, pd.Series]:
    """Filter the outcomes into labels for ML algorithms.

    This returns both uplift and response/propensity labels. This also puts
    the sensitive attributes into the targets as a MultiIndex like aif360.
    """
    # Un-dummy uplift target
    uplift = outcomes[["TN", "TR", "CN", "CR"]].astype(int)
    uplift = uplift[uplift == 1].stack().reset_index()["level_1"]

    # response
    response = outcomes["s_applied"]  # you don't see DND's in a response mod.

    # Now make a multi-index with all of the relevant things
    outcome_cols = ["s_applied", "s_acquired", "s_success",
                    "ns_applied", "ns_acquired", "ns_success"]
    ids = pd.Series(outcomes.index)
    index = pd.concat((ids, sensitives, outcomes[outcome_cols]), axis=1)
    index = pd.MultiIndex.from_frame(index)
    uplift.index = index
    response.index = index

    return uplift, response


def run_simulation() -> None:
    """Simulate "true lift" data."""
    covariates, sensitives, outcomes, truth = simulate()

    print("\nCovariate statistics:")
    print(covariates.describe())

    print("\nSensitive attribute statistics:")
    print(sensitives.describe())

    print("\nOutcomes statistics:")
    print(outcomes.sum(axis=0))

    print("\nTruth statistics:")
    print(truth.drop(columns=["lift"]).sum(axis=0))

    # Look at true lift scores by label
    print("\nLift statistics:")
    print("P av. lift: {:.4f}".format(truth["lift"][truth["P"]].mean()))
    print("ST av. lift: {:.4f}".format(truth["lift"][truth["ST"]].mean()))
    print("LC av. lift: {:.4f}".format(truth["lift"][truth["LC"]].mean()))
    print("DND av. lift: {:.4f}".format(truth["lift"][truth["DND"]].mean()))

    # Save
    covariates.to_csv(cfg.COVARIATES_FILE, index_label=cfg.INDEX_LABEL)
    sensitives.to_csv(cfg.SENSITIVE_FILE, index_label=cfg.INDEX_LABEL)
    outcomes.to_csv(cfg.OUTCOMES_FILE, index_label=cfg.INDEX_LABEL)
    truth.to_csv(cfg.TRUTH_FILE, index_label=cfg.INDEX_LABEL)


if __name__ == "__main__":
    run_simulation()
