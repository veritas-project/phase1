'''
Data preprocessing script

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

# %% Imports
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


# %% Helpers
def load_raw_dataset(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'default.payment.next.month': 'TARGET',
                       'PAY_0': 'PAY_1'}, inplace=True)
    df['TARGET'] = 1 - df['TARGET']  # 0: default/bad,  1: resolve/good
    return df


_parser = argparse.ArgumentParser()
_parser.add_argument('--data-folder', type=str,
                     default='data/creditdata',
                     help='path to UCI Credit Card dataset CSV file')

# %% Main
if __name__ == '__main__':
    args = _parser.parse_args()
    df = load_raw_dataset(os.path.join(args.data_folder, 'UCI_Credit_Card.csv'))
    train, test = train_test_split(df, test_size=0.25, shuffle=True,
                                   stratify=df['TARGET'], random_state=0)

    train.to_csv(os.path.join(args.data_folder, 'creditdata_train_v2.csv'), index=False)
    test.to_csv(os.path.join(args.data_folder, 'creditdata_test_v2.csv'), index=False)
