'''
Helper functions for credit scoring analysis

Written by Marc-Etienne Brunet,
Element AI inc. (info@elementai.com).

Copyright © 2020 Monetary Authority of Singapore

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
'''

from collections import namedtuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve


def load_dataset(filepath, target='TARGET', drop_columns=[]):
    df = pd.read_csv(filepath)
    # Extraneous columns
    df.drop(drop_columns, axis=1, inplace=True)
    original_len = len(df)
    # Ensure nothing missing
    df.dropna(how="any", axis=0, inplace=True)
    n_dropped = original_len - len(df)
    n_dropped != 0 and print(f"Warning - dropped {n_dropped} rows with NA data.")
    df = compress_df_mem(df)  # Need this to detect categorical variables for SMOTe
    y = np.array(df[target])
    df.drop(target, axis=1, inplace=True)
    return df, y


def compress_df_mem(df):
    """Compress memory usage of a dataframe"""
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            col_min = df[col].min()
            col_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


def train_log_reg_model(X, y, seed=0, C=1, verbose=False, upsample=True):
    if upsample:
        verbose and print('upsampling...')
        categorical_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'int8']
        smote = SMOTENC(random_state=seed, categorical_features=categorical_features)
        X, y = smote.fit_resample(X, y)

    verbose and print('scaling...')
    scaling = StandardScaler()
    X = scaling.fit_transform(X)

    verbose and print('fitting...')
    verbose and print('C:', C)
    model = LogisticRegression(random_state=seed, C=C, max_iter=4000)
    model.fit(X, y)

    verbose and print('chaining pipeline...')
    pipe = Pipeline([('scaling', scaling), ('model', model)])
    verbose and print('done.')
    return pipe


def bootstrap_conf_int(y_true, y_model, score_func, k=50):
    results = np.zeros(k)
    for i in range(k):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        results[i] = score_func(y_true[idx], y_model[idx])

    return results.mean(), 2 * results.std()


def format_uncertainty(mean_val, conf_int):
    return f"{mean_val:.5f} +/- {conf_int:.5f}"


class ModelRates:
    def __init__(self, y_true, y_prob):
        (ths, tpr, fpr, ppv, bdr, ar) = self._compute_rates(y_true, y_prob)
        self.base_default_rate = bdr
        self.tpr = interp1d(ths, tpr)
        self.fpr = interp1d(ths, fpr)
        self.ppv = interp1d(ths, ppv)
        self.accept_rate = interp1d(ths, ar)

    @staticmethod
    def _compute_rates(y_true, y_prob):
        # Vectorizable computation of rates
        fpr, tpr, ths = roc_curve(y_true, y_prob, pos_label=1)
        ths[0] = 1.0  # roc_curve sets max threshold arbitrarily above 1
        ths = np.append(ths, [0.0])  # Add endpoints for ease of interpolation
        fpr = np.append(fpr, [1.0])
        tpr = np.append(tpr, [1.0])
        base_default_rate = 1 - np.mean(y_true)
        accept_rate = (1 - base_default_rate) * tpr + base_default_rate * fpr
        prob_tp = (1 - base_default_rate) * tpr
        ppv = np.divide(prob_tp, accept_rate, out=np.zeros_like(prob_tp), where=(accept_rate != 0))
        return ths, tpr, fpr, ppv, base_default_rate, accept_rate


class FairnessAnalysis:
    Metrics = namedtuple('Metrics', 'equal_opp fpr_bal avg_odds dem_parity ppv_parity bal_acc')

    metric_names = {'equal_opp': 'Equal Opportunity',
                    'fpr_bal': 'False Positive Rate Balance',
                    'avg_odds': 'Average Odds',
                    'dem_parity': 'Demographic Parity',
                    'ppv_parity': 'Positive Predictive Parity',
                    'bal_acc': 'Balanced Accuracy'}

    def __init__(self, y_true, y_prob, group_mask):
        self.rates_a = ModelRates(y_true[group_mask], y_prob[group_mask])
        self.rates_b = ModelRates(y_true[~group_mask], y_prob[~group_mask])
        self.y_true = y_true
        self.y_prob = y_prob
        self.group_mask = group_mask
        assert np.sum(y_true == 1) + np.sum(y_true == 0) == len(y_true)  # Confirm 1, 0 labelling

    def compute(self, th_a=0.5, th_b=None):
        # Vectorizable
        if th_b is None:
            th_b = th_a

        # Fairness
        tpr_a, tpr_b = self.rates_a.tpr(th_a), self.rates_b.tpr(th_b)
        fpr_a, fpr_b = self.rates_a.fpr(th_a), self.rates_b.fpr(th_b)
        equal_opp = tpr_a - tpr_b
        fpr_bal = fpr_a - fpr_b
        avg_odds = 0.5 * (equal_opp + fpr_bal)
        dem_parity = self.rates_a.accept_rate(th_a) - self.rates_b.accept_rate(th_b)
        ppv_parity = self.rates_a.ppv(th_a) - self.rates_b.ppv(th_b)

        # Performance
        # Combine TPRs: P(R=1|Y=1) = P(R=1|Y=1,A=1)P(A=1|Y=1) + P(R=1|Y=1,A=0)P(A=0|Y=1)
        tpr = (tpr_a * np.mean(self.group_mask[self.y_true == 1]) +
               tpr_b * np.mean(~self.group_mask[self.y_true == 1]))
        # Combine FPRs: P(R=1|Y=0) = P(R=1|Y=0,A=1)P(A=1|Y=0) + P(R=1|Y=0,A=0)P(A=0|Y=0)
        fpr = (fpr_a * np.mean(self.group_mask[self.y_true == 0]) +
               fpr_b * np.mean(~self.group_mask[self.y_true == 0]))
        bal_acc = 0.5 * (tpr + 1-fpr)

        return self.Metrics(equal_opp, fpr_bal, avg_odds, dem_parity, ppv_parity, bal_acc)
