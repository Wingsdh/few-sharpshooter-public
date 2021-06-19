# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    data_utils.py
   Description :
   Author :       Wings DH
   Time：         5/26/21 11:35 PM
-------------------------------------------------
   Change Activity:
                   5/26/21: Create
-------------------------------------------------
"""
import json


def load_data(fp, key_sentence, key_label):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)

        data = [(d[key_sentence], d[key_label]) for d in data]
        print(f'Loaded {len(data)} data from {fp}')
        return data


def load_test_data(fp):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)

        print(f'Loaded test data {len(data)} data from {fp}')
        return data
