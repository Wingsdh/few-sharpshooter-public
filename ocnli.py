# -*- coding:utf-8 -*-
"""
  @Time : 2021-06-20 13:47
  @Authpr : zjg
  @File : ocnli.py
"""

import sys
import os
import json
from tqdm import tqdm

from utils.cls_train import dump_result
from utils.seed import set_seed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.data_utils import load_test_data
from absl import app, flags

sys.path.append('../')
sys.path.append('./')

label_2_desc = {'neutral': '看不出来', 'entailment': '是这样的', 'contradiction': '并非如此'}
flags.DEFINE_string('c', '0', 'index of ocnli dataset')
FLAGS = flags.FLAGS
set_seed()


# set_seed()
def build_data(sent1, sent2):
    return sent1 + "所以，锟锟锟锟" + "，" + sent2


def infer(test_data, classifier):
    for d in test_data:
        sentence_1 = d.pop('sentence1')
        sentence_2 = d.pop('sentence2')
        sentence = build_data(sentence_1, sentence_2)
        label, _ = classifier.classify(sentence)
        d['label'] = label
    return test_data


def load_ocnli_data(fp, key_sentence_1, key_sentence_2, key_label):
    data = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for d in fd:
            d = json.loads(d.strip())
            sentence_1 = d[key_sentence_1]
            sentence_2 = d[key_sentence_2]
            label = d[key_label]
            sentence = build_data(sentence_1, sentence_2)
            data.append((sentence, label))
        return data


def build_ocnli_train_data(fp, key_sentence_1, key_sentence_2, key_label):
    data = []
    pairs = []
    with open(fp, 'r', encoding='utf-8') as fd:
        for d in fd:
            d = json.loads(d.strip())
            sentence_1 = d[key_sentence_1]
            sentence_2 = d[key_sentence_2]
            label = d[key_label]
            pairs.append([sentence_1, sentence_2, label])

        for sentence_1, sentence_2, label in pairs:
            # 负例
            for other_sent1, other_sent2, _ in pairs:
                if other_sent1 == sentence_1:
                    continue
                data.append([build_data(sentence_1, other_sent2), 'neutral'])
                data.append([build_data(other_sent1, sentence_2), 'neutral'])

        return data


def get_data_fp(use_index):
    train_fp = f'dataset/ocnli/train_{use_index}.json'
    dev_fp = f'dataset/ocnli/dev_{use_index}.json'
    test_fp = 'dataset/ocnli/test.json'
    my_test_fp = []
    for ind in range(5):
        if ind != use_index:
            my_test_fp.append(f'dataset/ocnli/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def eval_ocnli_model(classifier, test_fps, key_sentence_1, key_sentence_2, key_label, need_print=False):
    cnt = 0
    test_data = []
    for fp in test_fps:
        each_data = load_ocnli_data(fp, key_sentence_1, key_sentence_2, key_label)
        test_data.extend(each_data)

    for sentence, label in tqdm(test_data):
        pred_label, _ = classifier.classify(sentence)
        if label == pred_label:
            cnt += 1
        elif need_print:
            print('-----')
            print(label, pred_label)
            print(sentence)

    return cnt / len(test_data)


def main(_):
    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label'
    key_sentence_1 = 'sentence1'
    key_sentence_2 = 'sentence2'
    train_data = load_ocnli_data(train_fp, key_sentence_1, key_sentence_2, key_label)
    # extra_data = build_ocnli_train_data(train_fp, key_sentence_1, key_sentence_2, key_label)
    data = [LabelData(text, label) for text, label in train_data]
    print(f'Got train_data: {len(train_data)}')

    dev_data = load_ocnli_data(dev_fp, key_sentence_1, key_sentence_2, key_label)
    print(f'Got dev data: {len(dev_data)}')

    # 初始化encoder
    n_top = 5
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = f'../temp_ocnli_{FLAGS.c}.weights'
    prefix = ''
    mask_ind = []
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc,
                             8,
                             merge=MlmBertEncoder.CONCAT, norm=False,
                             lr=1e-4,
                             policy='attention')
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)
    rst = eval_ocnli_model(classifier, [dev_fp], key_sentence_1, key_sentence_2, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # fine tune
    best_acc = 0
    for epoch in range(20):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=n_top)

        print('Evel model')
        rst = eval_ocnli_model(classifier, [dev_fp], key_sentence_1, key_sentence_2, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

    # 加载最终模型
    encoder.load()

    classifier = RetrieverClassifier(encoder, data, n_top=n_top)

    # 自测试集测试
    rst = eval_ocnli_model(classifier, my_test_fp, key_sentence_1, key_sentence_2, key_label, need_print=False)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)
    outp_fn = f'ocnlif_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
