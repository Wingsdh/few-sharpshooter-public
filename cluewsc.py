# -*- coding:utf-8 -*-
"""
  @Time : 2021-06-19 13:43
  @Authpr : MandyL
  @File : cluewsc.py
"""
import json
import os

from tqdm import tqdm
from absl import app, flags

from utils.cls_train import dump_result
from utils.seed import set_seed

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
set_seed()

flags.DEFINE_string('c', '0', 'index of dataset')
FLAGS = flags.FLAGS


def eval_wsc_model(classifier, test_fps, key_sentence, key_label, need_print=False):
    cnt = 0
    test_data = []
    for fp in test_fps:
        each_data = load_wsc_data(fp, key_sentence, key_label)
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
            text += f'其中，{span2_text}锟锟锟是{span1_text}'
            data.append((text, labels))
        return data


def infer(test_fp, classifier):
    data = []
    with open(test_fp, 'r', encoding='utf-8') as fd:
        for l in fd:
            d = json.loads(l.strip())
            data.append(d)

    for value in data:
        target = value.pop('target')
        span1_index = target.pop('span1_index')
        span1_text = target.pop('span1_text')
        span2_index = target.pop('span2_index')
        span2_text = target.pop('span2_text')
        text = value.pop('text')
        text += f'其中，{span2_text}锟锟锟是{span1_text}'
        label, _ = classifier.classify(text)
        value['label'] = label
    return data


label_2_desc = {'true': '确实就', 'false': '不可能'}


def get_data_fp(use_index):
    train_fp = f'dataset/cluewsc/train_{use_index}.json'
    dev_fp = f'dataset/cluewsc/dev_{use_index}.json'
    test_fp = 'dataset/cluewsc/test.json'
    my_test_fp = []
    for ind in range(5):
        if str(ind) != use_index:
            my_test_fp.append(f'dataset/cluewsc/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label'
    key_sentence = 'text'
    train_data = load_wsc_data(train_fp, key_sentence, key_label)
    dev_data = load_wsc_data(dev_fp, key_sentence, key_label)
    data = [LabelData(text, label) for text, label in train_data]

    # 初始化encoder
    n_top = len(train_data) // 10
    model_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12'
    weight_path = f'../temp_cluewsc_{FLAGS.c}.weights'
    prefix = ''
    mask_ind = []
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 8,
                             merge=MlmBertEncoder.CONCAT, norm=False, lr=1e-4, policy='attention')
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)
    rst = eval_wsc_model(classifier, [dev_fp], key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # fine tune
    best_acc = 0
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=n_top)

        print('Evel model')
        rst = eval_wsc_model(classifier, [dev_fp], key_sentence, key_label)
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

        print(f'{train_fp} + {dev_fp} -> {rst}')

    # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)

    # 自测试集测试
    rst = eval_wsc_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = infer(test_fp, classifier)
    outp_fn = f'cluewscf_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
