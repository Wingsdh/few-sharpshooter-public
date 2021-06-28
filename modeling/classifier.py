# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    classifier.py
   Description :
   Author :       Wings DH
   Time：         2021/5/14 下午10:04
-------------------------------------------------
   Change Activity:
                   2021/5/14: Create
-------------------------------------------------
"""
from abc import ABC, abstractmethod


class LabelData(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __repr__(self):
        return f'{self.label}: {self.text}'


class BaseClassifier(ABC):

    @abstractmethod
    def classify(self, text):
        pass
