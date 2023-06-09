# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: bert4cls_model.py
@time: 2021/1/26 10:34
"""
import os
from core.layers import *
from core.models import build_transformer_model
# from core.optimizers import Adam
from keras.models import Model
from keras.optimizers import Adam


class Bert4Classification(object):
    def __init__(self, args, nums_class):
        self.args = args
        # 模型结构参数
        self.nums_class = nums_class
        # 字典

    def build_model(self):
        if self.args.bert_model_path:
            config_path = os.path.join(self.args.bert_model_path, "bert_config.json")
            checkpoint_path = os.path.join(self.args.bert_model_path, "bert_model.ckpt")
            model = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
            output_layer = 'Transformer-%s-FeedForward-Norm' % (self.args.bert_layers - 1)
            output = model.get_layer(output_layer).output
            output = Lambda(lambda x: x[:, 0:1, :])(output)  # 获取CLS
            output = Lambda(lambda x: K.squeeze(x, axis=1))(output)
            output = Dense(self.nums_class, activation="softmax")(output)

            new_model = Model(model.input, output)
            new_model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(self.args.lr),
                              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
                              )
            return new_model
