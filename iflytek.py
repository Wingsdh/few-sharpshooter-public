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

flags.DEFINE_string('c', '0', 'index of tnews dataset')
FLAGS = flags.FLAGS


def infer(test_data, classifier):
    for d in test_data:
        sentence = d.pop('sentence')
        label = classifier.classify(sentence)
        label_id = label_desc_id[label]
        d['label'] = label_id
    return test_data


label_desc_id = {'旅游资讯': '62',
                 '公务员': '55',
                 '免费WIFI': '2',
                 '英语': '56',
                 '婚恋社交': '29',
                 '体育竞技': '19',
                 '收款': '117',
                 '汽车咨询': '75',
                 '成人教育': '59',
                 '银行': '99',
                 '影像剪辑': '101',
                 '政务': '9',
                 '民航': '64',
                 '语言(非英语)': '61',
                 '短视频': '47',
                 'MOBA': '23',
                 '动作类': '18',
                 '视频教育': '57',
                 '音乐': '48',
                 '问答交流': '39',
                 '杂志': '41',
                 '薅羊毛': '11',
                 '辅助工具': '24',
                 '股票': '94',
                 '电影票务': '109',
                 '职考': '54',
                 '体育咨讯': '90',
                 '打车': '0',
                 '工具': '70',
                 '卡牌': '14',
                 '情侣社交': '30',
                 '驾校': '73',
                 '魔幻': '12',
                 '社区服务': '10',
                 '生活社交': '32',
                 '相机': '103',
                 '高等教育': '58',
                 '汽车交易': '76',
                 '射击游戏': '16',
                 '直播': '49',
                 '工作社交': '27',
                 '买房': '80',
                 '技术': '37',
                 '美妆美业': '87',
                 '租房': '79',
                 '艺术': '60',
                 '电商': '106',
                 'K歌': '51',
                 '铁路': '65',
                 '飞行空战': '15',
                 '绘画': '104',
                 '教辅': '38',
                 '酒店': '66',
                 '运动健身': '91',
                 '社交工具': '31',
                 '出国': '69',
                 '微博博客': '33',
                 '问诊挂号': '83',
                 '日常养车': '77',
                 '经营': '116',
                 '支付': '92',
                 '日程管理': '114',
                 '其他': '118',
                 '视频': '46',
                 '违章': '74',
                 '即时通讯': '26',
                 '团购': '107',
                 '购物咨询': '111',
                 '电子产品': '82',
                 '记账': '98',
                 '养生保健': '84',
                 '策略': '22',
                 '仙侠': '13',
                 '中小学': '53',
                 '棋牌中心': '20',
                 '小说': '36',
                 '求职': '44',
                 '二手': '105',
                 '母婴': '72',
                 '约会社交': '25',
                 '餐饮店': '89',
                 '民宿短租': '68',
                 '成人': '52',
                 '家政': '7',
                 '办公': '113',
                 '同城服务': '4',
                 '租车': '3',
                 '百科': '42',
                 '搞笑': '40',
                 '漫画': '35',
                 '理财': '96',
                 '电台': '50',
                 '兼职': '45',
                 '亲子儿童': '71',
                 '女性': '115',
                 '行车辅助': '78',
                 '彩票': '97',
                 '摄影修图': '102',
                 '经营养成': '21',
                 '行程管理': '67',
                 '论坛圈子': '28',
                 '装修家居': '81',
                 '婚庆': '6',
                 '保险': '93',
                 '快递物流': '5',
                 '外卖': '108',
                 '地图导航': '1',
                 '美颜': '100',
                 '社区超市': '110',
                 '减肥瘦身': '86',
                 '公共交通': '8',
                 '影视娱乐': '43',
                 '医疗服务': '85',
                 '借贷': '95',
                 '新闻': '34',
                 '菜谱': '88',
                 '休闲益智': '17',
                 '笔记': '112',
                 '综合预定': '63'}
label_2_desc = {'银行': '银行',
                '社区服务': '社区',
                '电商': '电商',
                '支付': '支付',
                '经营养成': '经营',
                '卡牌': '卡牌',
                '借贷': '借贷',
                '驾校': '驾校',
                '理财': '理财',
                '职考': '职考',
                '新闻': '新闻',
                '旅游资讯': '旅游',
                '公共交通': '交通',
                '魔幻': '魔幻',
                '医疗服务': '医疗',
                '影像剪辑': '影像',
                '动作类': '动作',
                '工具': '工具',
                '体育竞技': '体育',
                '小说': '小说',
                '运动健身': '运动',
                '相机': '相机',
                '辅助工具': '工具',
                '快递物流': '快递',
                '高等教育': '教育',
                '股票': '股票',
                '菜谱': '菜谱',
                '行车辅助': '行车',
                '仙侠': '仙侠',
                '亲子儿童': '亲子',
                '购物咨询': '购物',
                '射击游戏': '射击',
                '漫画': '漫画',
                '中小学': '小学',
                '同城服务': '同城',
                '成人教育': '成人',
                '求职': '求职',
                '电子产品': '电子',
                '艺术': '艺术',
                '薅羊毛': '赚钱',
                '约会社交': '约会',
                '经营': '经营',
                '兼职': '兼职',
                '短视频': '视频',
                '音乐': '音乐',
                '英语': '英语',
                '棋牌中心': '棋牌',
                '摄影修图': '摄影',
                '养生保健': '养生',
                '办公': '办公',
                '政务': '政务',
                '视频': '视频',
                '论坛圈子': '论坛',
                '彩票': '彩票',
                '直播': '直播',
                '其他': '其他',
                '休闲益智': '休闲',
                '策略': '策略',
                '即时通讯': '通讯',
                '汽车交易': '买车',
                '违章': '违章',
                '地图导航': '地图',
                '民航': '民航',
                '电台': '电台',
                '语言(非英语)': '语言',
                '搞笑': '搞笑',
                '婚恋社交': '婚恋',
                '社区超市': '超市',
                '日常养车': '养车',
                '杂志': '杂志',
                '视频教育': '在线',
                '家政': '家政',
                '影视娱乐': '影视',
                '装修家居': '装修',
                '体育咨讯': '资讯',
                '社交工具': '社交',
                '餐饮店': '餐饮',
                '美颜': '美颜',
                '问诊挂号': '挂号',
                '飞行空战': '飞行',
                '综合预定': '预定',
                '电影票务': '票务',
                '笔记': '笔记',
                '买房': '买房',
                '外卖': '外卖',
                '母婴': '母婴',
                '打车': '打车',
                '情侣社交': '情侣',
                '日程管理': '日程',
                '租车': '租车',
                '微博博客': '博客',
                '百科': '百科',
                '绘画': '绘画', '铁路': '铁路',
                '生活社交': '生活',
                '租房': '租房',
                '酒店': '酒店',
                '保险': '保险',
                '问答交流': '问答',
                '收款': '收款',
                'MOBA': '竞技',
                'K歌': '唱歌',
                '技术': '技术',
                '减肥瘦身': '减肥',
                '工作社交': '工作',
                '团购': '团购',
                '记账': '记账',
                '女性': '女性',
                '公务员': '公务',
                '二手': '二手',
                '美妆美业': '美妆',
                '汽车咨询': '汽车', '行程管理': '行程',
                '免费WIFI': '免费', '教辅': '教辅', '成人': '两性', '出国': '出国', '婚庆': '婚庆', '民宿短租': '民宿'}


def get_data_fp(use_index):
    train_fp = f'dataset/iflytek/train_{use_index}.json'
    dev_fp = f'dataset/iflytek/dev_{use_index}.json'
    test_fp = 'dataset/iflytek/test.json'
    my_test_fp = []
    for ind in range(5):
        if str(ind) != use_index:
            my_test_fp.append(f'dataset/iflytek/dev_{ind}.json')
    return train_fp, dev_fp, my_test_fp, test_fp


def main(_):
    # 参数

    # 加载数据
    train_fp, dev_fp, my_test_fp, test_fp = get_data_fp(FLAGS.c)
    key_label = 'label_des'
    key_sentence = 'sentence'
    train_data = load_data(train_fp, key_sentence, key_label)
    dev_data = load_data(dev_fp, key_sentence, key_label)

    # 初始化encoder
    model_path = 'pretrained_model/roberta'
    weight_path = '../temp_iflytek.weights'
    prefix = '做为一款游戏应用，'
    mask_ind = [4, 5]
    encoder = MlmBertEncoder(model_path, weight_path, train_data, dev_data, prefix, mask_ind, label_2_desc, 8)

    # fine tune
    data = [LabelData(text, label) for text, label in train_data]
    best_acc = 0
    for epoch in range(10):
        print(f'Training epoch {epoch}')
        encoder.train(1)
        # 加载分类器
        classifier = RetrieverClassifier(encoder, data, n_top=11)

        print('Evel model')
        rst = eval_model(classifier, [dev_fp], key_sentence, key_label)
        print(f'{train_fp} + {dev_fp} -> {rst}')
        if rst > best_acc:
            encoder.save()
            best_acc = rst
            print(f'Save for best {best_acc}')

    # # 加载最终模型
    encoder.load()
    classifier = RetrieverClassifier(encoder, data, n_top=11)

    # 自测试集测试
    rst = eval_model(classifier, my_test_fp, key_sentence, key_label)
    print(f'{train_fp} + {dev_fp} -> {rst}')

    # 官方测试集
    test_data = load_test_data(test_fp)
    test_data = infer(test_data, classifier)

    outp_fn = f'iflytekf_predict_{FLAGS.c.replace("few_all", "all")}.json'
    dump_result(outp_fn, test_data)


if __name__ == "__main__":
    app.run(main)
