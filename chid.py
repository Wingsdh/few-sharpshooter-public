# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    chid.py
   Description :
   Author :       Wings DH
   Time：         5/23/21 10:45 PM
-------------------------------------------------
   Change Activity:
                   5/23/21: Create
-------------------------------------------------
"""
import json
import os

from tqdm import tqdm

from bert4keras.models import build_transformer_model
from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer
import numpy as np

from utils.cls_train import dump_result


class MaskLm(object):
    MAX_LEN = 256

    def init_bert(self, model_path):
        dict_path = os.path.join(model_path, 'vocab.txt')
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

        config_path = os.path.join(model_path, 'bert_config.json')
        checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_mlm=True
        )

        self.model = bert

    def __init__(self, model_path):
        self.init_bert(model_path)

    def get_prob(self, text, mask_ind):
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.MAX_LEN)
        tokens = self.tokenizer.tokenize(text, maxlen=self.MAX_LEN)
        mapping = self.tokenizer.rematch(text, tokens)
        text_ind_2_token_ind = {}

        for token_ind, text_inds in enumerate(mapping):
            for ind in text_inds:
                text_ind_2_token_ind[ind] = token_ind

        masked_text = []
        for ind in mask_ind:
            if ind >= self.MAX_LEN or ind not in text_ind_2_token_ind:
                continue

            masked_text.append(token_ids[text_ind_2_token_ind[ind]])
            token_ids[text_ind_2_token_ind[ind]] = self.tokenizer._token_mask_id

        token_ids, segment_ids = to_array([token_ids], [segment_ids])

        # 用mlm模型预测被mask掉的部分
        probas = self.model.predict([token_ids, segment_ids])[0]
        log_prob = 0
        for ind, masked_id in zip(mask_ind, masked_text):
            if ind >= self.MAX_LEN or ind not in text_ind_2_token_ind:
                continue

            prob = probas[text_ind_2_token_ind[ind]][masked_id]
            log_prob += -np.log(prob)
        return log_prob


def load_data(fp):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)
        return data


def load_test_data(fp):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)
        return data


def main():
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    mlm = MaskLm(model_path)
    test_fp = 'dataset/chid/test.json'
    data = load_test_data(test_fp)

    for d in tqdm(data):
        candidates = d['candidates']
        content = d['content']
        probs = []
        for c in candidates:
            text = content.replace('#idiom#', c)
            ind = text.index(c)
            prob = mlm.get_prob(text, [ind, ind + 1, ind + 2, ind + 3])
            probs.append(prob)

        answer = np.argmin(probs)
        # print(answer)
        d['answer'] = int(answer)
        d.pop('content')
        d.pop('candidates')
        # print(candidates[answer], content)

    outp_fn = f'chidf_predict_all.json'
    dump_result(outp_fn, data)
    outp_fn = f'chidf_predict_0.json'
    dump_result(outp_fn, data)
    outp_fn = f'chidf_predict_1.json'
    dump_result(outp_fn, data)
    outp_fn = f'chidf_predict_2.json'
    dump_result(outp_fn, data)
    outp_fn = f'chidf_predict_3.json'
    dump_result(outp_fn, data)
    outp_fn = f'chidf_predict_4.json'
    dump_result(outp_fn, data)


if __name__ == "__main__":
    main()
