# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    base_encoder.py
   Description :
   Author :       Wings DH
   Time：         2021/5/13 下午9:56
-------------------------------------------------
   Change Activity:
                   2021/5/13: Create
-------------------------------------------------
"""
from abc import ABC, abstractmethod


class BaseEncoder(ABC):

    @abstractmethod
    def encode(self, text):
        pass

    @property
    @abstractmethod
    def dim(self):
        return


class BaseTokenEncoder(ABC):

    @abstractmethod
    def encode(self, text):
        pass

    @property
    @abstractmethod
    def dim(self):
        return
