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
import numpy as np
import tensorflow as tf
from core.snippets import sequence_padding, convert_to_unicode
from keras.utils import to_categorical
from keras.utils.data_utils import Sequence
from tensorflow_core.python.keras.callbacks import TensorBoard
from core.modeling.modeling_bert_classification import Bert4Classification
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint)
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing

from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

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


class ClassificationReporter(Callback):
    def __init__(self, validation_data):
        super(ClassificationReporter, self).__init__()
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
            # 这里可以设置不同的阈值进行验证
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
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_model_path, "vocab.txt"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)
    label_encode = LabelEncoder()
    # 模型
    bert = Bert4Classification(args=args, nums_class=len(label_encode.classes_))
    model = bert.build_model()
    word_dict = tokenizer.get_vocab()
    tokenizer.enable_truncation(max_length=args.max_length)
    processor = DataProcessorFunction()
    # 设置类别标签
    labels = processor.get_labels()
    label_encode.fit(labels)
    train_examples = processor.get_train_examples(args.train_data)
    train_features = processor.convert_examples_to_features(train_examples,
                                                            tokenizer=tokenizer,
                                                            label_encode=label_encode)
    train_sequence = DataSequence(train_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=len(label_encode.classes_),
                                  batch_size=args.batch_size)
    # 加载验证集数据
    dev_examples = processor.get_dev_examples(args.dev_data)
    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          label_encode=label_encode)
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=tokenizer.token_to_id("[PAD]"),
                                num_classes=len(label_encode.classes_),
                                batch_size=args.batch_size)
    call_backs = []
    classification_report = ClassificationReporter(dev_sequence)
    call_backs.append(classification_report)
    tensor_board = TensorBoard(log_dir="./logs", write_graph=False)
    call_backs.append(tensor_board)
    checkpoint = ModelCheckpoint('bert_cls_best.hdf5', monitor='accuracy', verbose=2, save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('accuracy', patience=4, mode='max', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)

    model.fit(train_sequence,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              use_multiprocessing=True,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              class_weight={0: 1, 1: 1.5},
              )
