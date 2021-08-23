# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task4classification.py
@time: 2021/1/25 15:06
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import csv
import itertools
import os
import pickle
import pprint
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from bert4keras.snippets import sequence_padding, convert_to_unicode
from keras.utils import to_categorical
from keras.utils.data_utils import Sequence

from core.modeling.modeling_bert_classification import Bert4Classification
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing


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


class ClassificationProcessor(object):

    def __init__(self, tokenizer):
        self.language = "zh"
        self.tokenizer = tokenizer

    def get_train_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            label = convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_path):
        pass

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = convert_to_unicode(line[0])
            if language != convert_to_unicode(self.language):
                continue
            text_a = convert_to_unicode(line[6])
            text_b = convert_to_unicode(line[7])
            label = convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def get_labels():
        return ["positive", "negative"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def batch_transform(self, features, num_classes):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for features in features:
            batch_token_ids.append(features.input_ids)
            batch_segment_ids.append(features.segment_ids)
            batch_labels.append(features.label_id)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.tokenizer.token_to_id("[PAD]"))
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return [batch_token_ids, batch_segment_ids], to_categorical(np.array(batch_labels), num_classes=num_classes)

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

    def convert_examples_features(self, examples, label_encode):
        features = []
        # 这里可以采用多线程等方式提高预处理速度
        for example in examples:
            feature = self.convert_single_example(example=example, tokenizer=self.tokenizer, label_encode=label_encode)
            features.append(feature)

        return features


class ClassificationSequence(Sequence):
    def __init__(self, features, processor, num_classes, batch_size):
        self.batch_size = batch_size
        self.features = features
        self.processor = processor
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, idx):
        data = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.processor.batch_transform(data, num_classes=self.num_classes)


class ClassificationCallBack(Callback):
    def __init__(self, validation_data):
        super(ClassificationCallBack, self).__init__()
        self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = []
        val_true = []
        for (x_val, y_val) in self.validation_data:
            val_pred_batch = self.model.predict_on_batch(x_val)
            val_pred_batch = np.argmax(np.asarray(val_pred_batch).round(), axis=1)
            val_pred.append(val_pred_batch)
            val_true.append(np.argmax(np.asarray(y_val).round(), axis=1))
        val_pred = np.asarray(list(itertools.chain.from_iterable(val_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(val_true)))

        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred, average='macro')
        _val_precision = precision_score(val_true, val_pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(sk_classification_report(val_true, val_pred, digits=4))

        return


class TaskBertClassification:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.args.bert_model_path, "vocab.txt"), pad_token="[PAD]")
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
        if not os.path.exists(self.args.output_root):
            os.mkdir(self.args.output_root)
        self.label_encode = LabelEncoder()

    @staticmethod
    def build_model(args, word_dict, nums_class):
        bert = Bert4Classification(args=args, word_dict=word_dict, nums_class=nums_class)
        model = bert.build_model()
        return model

    def train(self):
        word_dict = self.tokenizer.get_vocab()
        self.tokenizer.enable_truncation(max_length=self.args.max_length)
        processor = ClassificationProcessor(tokenizer=self.tokenizer)
        labels = processor.get_labels()
        self.label_encode.fit(labels)
        train_examples = processor.get_train_examples(self.args.train_data)
        train_features = processor.convert_examples_features(train_examples, label_encode=self.label_encode)
        train_sequence = ClassificationSequence(train_features,
                                                processor=processor,
                                                num_classes=len(self.label_encode.classes_),
                                                batch_size=self.args.batch_size)
        # for data in train_sequence:
        #     print(data)
        # call_backs = []
        # tensor_board = TensorBoard(log_dir="./logs", write_graph=False)
        # call_backs.append(tensor_board)
        # checkpoint = ModelCheckpoint('bert_cls_best.hdf5', monitor='val_acc', verbose=2, save_best_only=True,
        #                              mode='max',
        #                              save_weights_only=True)
        # call_backs.append(checkpoint)
        # early_stop = EarlyStopping('val_acc', patience=4, mode='max', verbose=2, restore_best_weights=True)
        # call_backs.append(early_stop)
        #
        model = self.build_model(self.args, word_dict, len(labels))
        model.fit(train_sequence,
                  steps_per_epoch=len(train_sequence),
                  epochs=self.args.epochs,
                  use_multiprocessing=True
                  )


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("-train_data", type=str, default="data/train.txt", help="train data path")
    parse.add_argument("-dev_data", type=str, default="data/train.txt", help="validation data path")
    parse.add_argument("-bert", type=str, default="BERT", choices=["BERT", "ALBERT"], help="bert type")
    parse.add_argument("-bert_model_path", type=str,
                       default="E:\\项目资料\\主题挖掘项目\\DataRepository\\bert_model")
    parse.add_argument("--bert_layers", type=int, default=12)
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=4, help="batch size")
    parse.add_argument("--max-length", type=int, default=128, help="max sequence length")

    parse.add_argument("-epochs", type=int, default=10, help="number of training epoch")
    args = parse.parse_args()
    task = TaskBertClassification(args=args)
    task.train()
