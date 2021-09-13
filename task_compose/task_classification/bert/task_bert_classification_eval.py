# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_bert_classification_eval.py
@time: 2021/8/31 9:32
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import Model
from keras.layers import Lambda, Dense

import argparse
import csv
import itertools
import os
import numpy as np
import tensorflow as tf
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing

from core.snippets import sequence_padding, convert_to_unicode
from keras.utils import to_categorical
from keras.utils.data_utils import Sequence
import keras.backend as K

# 关闭eager模式
tf.disable_eager_execution()


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessorFunction(object):

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            label = convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_path):
        pass

    def get_dev_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "dev-%d" % (i)
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            label = convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def get_labels():
        return ["positive", "negative"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if float(line[-1]) == 0:
                    lines.append([line[0], line[1], "negative"])
                else:

                    lines.append([line[0], line[1], "positive"])
            return lines

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode):
        encoder = tokenizer.encode(sequence=example.text_a, pair=example.text_b)
        input_ids = encoder.ids
        segment_ids = encoder.type_ids
        input_mask = encoder.attention_mask
        label_id = label_encode.transform([example.label])
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def convert_examples_to_features(self, examples, tokenizer, label_encode):
        features = []
        # 这里可以采用多线程等方式提高预处理速度
        for example in examples:
            feature = self.convert_single_example(example=example, tokenizer=tokenizer, label_encode=label_encode)
            features.append(feature)

        return features


class DataSequence(Sequence):
    def __init__(self, features, token_pad_id, num_classes, batch_size):
        self.batch_size = batch_size
        self.features = features
        self.token_pad_id = token_pad_id
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        data = self.features[index * self.batch_size:(index + 1) * self.batch_size]
        return self.feature_batch_transform(data)

    def feature_batch_transform(self, features):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for features in features:
            batch_token_ids.append(features.input_ids)
            batch_segment_ids.append(features.segment_ids)
            batch_labels.append(features.label_id)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return [batch_token_ids, batch_segment_ids], to_categorical(np.array(batch_labels),
                                                                    num_classes=self.num_classes)


if __name__ == '__main__':
    # 简化参数配置，不加载bert模型的初始化权重
    config_path = ""
    model_path = ""
    data_path = ""
    batch_size = 1
    tokenizer = BertWordPieceTokenizer(os.path.join(), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=128)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    model = build_transformer_model(
        config_path=config_path,
    )
    output_layer = 'Transformer-%s-FeedForward-Norm' % (11)
    output = model.get_layer(output_layer).output
    output = Lambda(lambda x: x[:, 0:1, :])(output)  # 获取CLS
    output = Lambda(lambda x: K.squeeze(x, axis=1))(output)
    output = Dense(2, activation="softmax")(output)
    new_model = Model(model.input, output)
    new_model.load_weights(model_path)
    processor = DataProcessorFunction()
    pre_exampels = processor.get_dev_examples(data_path=data_path)
    pre_features = processor.convert_examples_to_features(pre_exampels)
    train_sequence = DataSequence(pre_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=2,
                                  batch_size=batch_size)
