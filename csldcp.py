# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    tnews.py
   Description :
   Author :       Wings DH
   Time：         6/16/21 10:40 PM
-------------------------------------------------
   Change Activity:
                   6/16/21: Create
-------------------------------------------------
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modeling.classifier import LabelData
from modeling.mlm_encoder import MlmBertEncoder
from modeling.retriever_classifier import RetrieverClassifier
from utils.cls_train import eval_model, dump_result

sys.path.append('../')
sys.path.append('./')

from utils.data_utils import load_data, load_test_data
from absl import flags, app

flags.DEFINE_string('c', '0', 'index of dataset')
FLAGS = flags.FLAGS


def infer(test_data, classifier):
    for d in test_data:
        sentence = d.pop('content')
        label = classifier.classify(sentence)
        d['label'] = label
    return test_data


label_2_desc = {"材料科学与工程": "材料",
                "作物学": "作物",
                "口腔医学": "口腔",
                "药学": "药学",
                "教育学": "教育",
                "水利工程": "水利",
                "理论经济学": "经济",
                "食品科学与工程": "食品",
                "畜牧学/兽医学": "兽医",
                "体育学": "体育",
                "核科学与技术": "核能",
                "力学": "力学",
                "园艺学": "园艺",
                "水产": "水产",
                "法学": "法学",
                "地质学/地质资源与地质工程": "地质",
                "石油与天然气工程": "能源",
                "农林经济管理": "农林",
                "信息与通信工程": "通信",
                "图书馆、情报与档案管理": "情报",
                "政治学": "政治",
                "电气工程": "电气",
                "海洋科学": "海洋",
                "民族学": "民族",
                "航空宇航科学与技术": "航空",
                "化学/化学工程与技术": "化工",
                "哲学": "哲学",
                "公共卫生与预防医学": "卫生",
                "艺术学": "艺术",
                "农业工程": "农业",
                "船舶与海洋工程": "船舶",
                "计算机科学与技术": "计科",
                "冶金工程": "冶金",
                "交通运输工程": "交通",
                "动力工程及工程热物理": "动力",
                "纺织科学与工程": "纺织",
                "建筑学": "建筑",
                "环境科学与工程": "环境",
                "公共管理": "管理",
                "数学": "数学",
                "物理学": "物理",
                "林学/林业工程": "林业",
                "心理学": "心理",
                "历史学": "历史",
                "工商管理": "工商",
                "应用经济学": "经济",
                "中医学/中药学": "中医",
                "天文学": "天文",
                "机械工程": "机械",
                "土木工程": "土木",
                "光学工程": "光学",
                "地理学": "地理",
                "农业资源利用": "农业",
                "生物学/生物科学与工程": "生物",
                "兵器科学与技术": "兵器",
                "矿业工程": "矿业",
                "大气科学": "大气",
                "基础医学/临床医学": "医学",
                "电子科学与技术": "电子",
                "测绘科学与技术": "测绘",
                "控制科学与工程": "控制",
                "军事学": "军事",
                "中国语言文学": "语言",
                "新闻传播学": "新闻",
                "社会学": "社会",
                "地球物理学": "地球",
                "植物保护": "植物"}


def get_data_fp(use_index):
    train_fp = f'dataset/csldcp/train_{use_index}.json'
    dev_fp = f'dataset/csldcp/dev_{use_index}.json'
    test_fp = 'dataset/csldcp/test.json'
    my_test_fp = []
    for ind in range(5):
        ind = str(ind)
        if ind != use_index:
            my_test_fp.append(f'dataset/csldcp/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label'
    key_sentence = 'content'
    train_data = load_data(train_fp, key_sentence, key_label)
    dev_data = load_data(dev_fp, key_sentence, key_label)

    # 初始化encoder
    model_path = 'pretrained_model/roberta'
    weight_path = '../temp_csldcp.weights'

    prefix = '这篇安安论文阐述了'
    mask_ind = [2, 3]
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 8)

    # fine tune
    n_top = len(train_data) // 10
    best_acc = 0
    data = [LabelData(text, label) for text, label in train_data]
    for epoch in range(5):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=n_top)

        print('Evel model')
        rst = eval_model(classifier, [dev_fp], key_sentence, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

    # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=n_top)

    # 自测试集测试
    # rst = eval_model(classifier, my_test_fp, key_sentence, key_label)
    # print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)
    outp_fn = f'csldcp_predict{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
