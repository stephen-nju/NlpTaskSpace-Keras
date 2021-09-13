# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_albert_ml_classification.py
@time: 2021/9/10 16:05
"""
import argparse
import csv
import json
import os
import keras.backend as K
import keras.metrics
import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras.utils.data_utils import Sequence
from tokenizers.implementations import BertWordPieceTokenizer
from core.models import build_transformer_model
from core.snippets import convert_to_unicode, sequence_padding
import pickle

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


class DataProcessorFunctions(object):

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=line[1]))
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
            label = convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    @staticmethod
    def get_labels():
        return ["男", "女", "婴幼儿", "儿童", "少年", "青年", "中年", "老年", "学生", "孕妇", "上班族"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                assert len(line) == 2
                lines.append([line[0], json.loads(line[1])])
            return lines

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode):
        encoder = tokenizer.encode(sequence=example.text_a)
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
        return [batch_token_ids, batch_segment_ids], np.squeeze(np.array(batch_labels), axis=1)


def create_model_fn(config_path, num_classes, lr, **kwargs):
    if "ckpt_path" in kwargs:
        albert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=kwargs["ckpt_path"],
            model="albert")
    else:
        albert = build_transformer_model(
            config_path=config_path,
            model="albert"
        )
    output = albert.output
    output = Lambda(lambda x: x[:, 0:1, :])(output)  # 获取CLS
    output = Lambda(lambda x: K.squeeze(x, axis=1))(output)
    output = Dense(num_classes, activation="sigmoid")(output)
    model = Model(albert.input, output)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr),
                  metrics=['accuracy', keras.metrics.AUC()]
                  )
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("-train_data", type=str, default="data/train.multi.txt", help="train data path")
    parse.add_argument("-dev_data", type=str, default="data/train.multi.txt", help="validation data path")
    parse.add_argument("-albert", type=str, default="BERT", choices=["BERT", "ALBERT"], help="bert type")
    parse.add_argument("-bert_model", type=str,
                       default="E:\\resources\\albert_tiny_zh_google")
    parse.add_argument("--bert_layers", type=int, default=11)
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=4, help="batch size")
    parse.add_argument("--max-length", type=int, default=128, help="max sequence length")
    parse.add_argument("-epochs", type=int, default=10, help="number of training epoch")
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_model, "vocab.txt"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)

    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    label_encode = MultiLabelBinarizer()

    # 模型
    word_dict = tokenizer.get_vocab()
    tokenizer.enable_truncation(max_length=args.max_length)
    processor = DataProcessorFunctions()
    # 设置类别标签
    labels = processor.get_labels()
    label_encode.fit([labels])
    # 二维列表
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
    log_dir = os.path.join(args.output_root, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=False)
    call_backs.append(tensor_board)
    checkpoint = ModelCheckpoint('albert_ml.hdf5', monitor='accuracy', verbose=2, save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('accuracy', patience=4, mode='max', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)
    albert_config = os.path.join(args.bert_model, "albert_config_tiny_g.json")
    albert_ckpt = os.path.join(args.bert_model, "albert_model.ckpt")
    model = create_model_fn(config_path=albert_config,
                            ckpt_path=albert_ckpt,
                            lr=args.lr,
                            num_classes=len(label_encode.classes_)
                            )
    model.fit(train_sequence,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              use_multiprocessing=False,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              )
