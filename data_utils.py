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


def load_csl_keyword(fp, key_sentence, key_label,keyword):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)
        list_keyword = [''.join(da[keyword])for da in data]
        data = [(d[key_sentence]+str_keyword, d[key_label]) for d,str_keyword in zip(data,list_keyword)]
        print(f'Loaded {len(data)} data from {fp}')
    return data

def load_wsc_data(fp, key_sentence, key_label):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for idx, value in enumerate(fd):
            value = json.loads(value)
            target = value['target']
            labels = value[key_label]
            span1_index = target['span1_index']
            span1_text = target['span1_text']
            span2_index = target['span2_index']
            span2_text = target['span2_text']
            text = value[key_sentence]
            text_list = [x for x in text]
            text_list.insert(span1_index + len(span1_text), "（这是实体）")
            text_list.insert(span2_index + len(span2_text), "（这是代词）")
            text_new = "".join(text_list)
            data.append((text_new, labels))
        return data



def load_test_data(fp):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)

        print(f'Loaded test data {len(data)} data from {fp}')
        return data
