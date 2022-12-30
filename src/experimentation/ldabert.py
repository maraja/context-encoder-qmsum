from utils.experiments import get_experiments_json, get_experiments, save_results
import numpy as np
import pandas as pd
import json
import toml
from tensorflow.python import keras
from src.dataset.ldabertv3 import LDABERT3Dataset
from src.encoders.context_encoder_ldabert_2 import ContextEncoderSimple
import nltk
import tensorflow
import sentence_transformers
import transformers
import sys
import os
import config

import tensorflow as tf
import random
import string

sys.path.insert(0, config.root_path)


def get_random_hash(k):
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


class Experiment:
    def __init__(self):


def run_experiment(experiment_string: str, experiment_hash: str):
    """this will run an LDA BERT experiment

    Args:
        experiment_string (string): experiment name found in settings/experiments.json
        experiment_hash (string): experiment hash if training continuation needed
    """
    experiments_config = get_experiments_json('qmsum_ldabert2_simple_test')
    experiments_config_df = pd.DataFrame.from_dict(experiments_config)

    for experiment in experiments_config[0:3]:
        dataset_type = experiment['dataset_type']
        final_dropout = experiment['final_dropout']
        dense_neurons = experiment['dense_neurons']
        pct_data = experiment['pct_data']
        augment_pct = experiment['augment_pct']
        gamma = experiment['gamma']
        max_sentence_length = experiment['max_sentence_length']
        bert_trainable = experiment['bert_trainable']
        epochs = experiment['epochs']
        BATCH_SIZE = 32
        print("params:", experiment)
        random_hash = get_random_hash(5)
        experiment['epochs'] = epochs

        # init model
        print("initializing model...")
        model = ContextEncoderSimple(final_dropout=final_dropout,
                                     dense_neurons=dense_neurons,
                                     gamma=gamma,
                                     max_sentence_length=max_sentence_length,
                                     bert_trainable=bert_trainable)

        # print("number of params: ", sum([np.prod(keras.get_value(w).shape) for w in model.trainable_weights]))

        # init training dataset
        print("initializing dataset...")
        dataset = LDABERT3Dataset(dataset_type=dataset_type,
                                  pct_data=pct_data,
                                  max_seq_length=max_sentence_length,
                                  max_segment_length=300,
                                  augment_pct=0,
                                  split="train",
                                  artificial_segments=False)

        # process training dataset
        print("processing dataset...")
        sentences, tokenized_sentences, labels = dataset.process()

        vectors_filename = '{}_{}_as-{}_msl-{}.pkl'.format(
            dataset.pct_data, dataset.augment_pct, dataset.artificial_segments, dataset.max_segment_length)

        saved_vectors, saved_labels, saved_sentences, saved_tokenized_sentences = dataset.get_saved_vectors(
            "train", dataset.dataset_type, vectors_filename)

        if len(saved_vectors) == 0:
            saved_vectors, saved_labels, saved_sentences, saved_tokenized_sentences = dataset.create_vectors(
                "train", dataset.dataset_type, vectors_filename)

        left_input, mid_input, right_input = dataset.format_sentences_tri_input_plus(
            saved_tokenized_sentences)
        lda_left_input, lda_mid_input, lda_right_input = dataset.format_sentences_tri_input(
            saved_vectors)

        # init testing dataset
        print("initializing testing dataset...")
        testing_dataset = LDABERT3Dataset(dataset_type=dataset_type,
                                          pct_data=0.25,
                                          max_seq_length=512,
                                          max_segment_length=300,
                                          augment_pct=0,
                                          split="test",
                                          artificial_segments=False)

        # process testing dataset
        print("processing testing dataset...")
        testing_sentences, testing_tokenized_sentences, testing_labels = testing_dataset.process()

        vectors_filename = '{}_{}_testing_data.pkl'.format(
            testing_dataset.pct_data, testing_dataset.augment_pct)

        testing_saved_vectors, testing_saved_labels, testing_saved_sentences, testing_saved_tokenized_sentences = testing_dataset.get_saved_vectors(
            "test", testing_dataset.dataset_type, vectors_filename)

        if len(testing_saved_vectors) == 0:
            testing_saved_vectors, testing_saved_labels, testing_saved_sentences, testing_saved_tokenized_sentences = testing_dataset.create_vectors(
                "test", testing_dataset.dataset_type, vectors_filename)

        testing_left_input, testing_mid_input, testing_right_input = testing_dataset.format_sentences_tri_input_plus(
            testing_saved_tokenized_sentences)
        testing_lda_left_input, testing_lda_mid_input, testing_lda_right_input = testing_dataset.format_sentences_tri_input(
            testing_saved_vectors)

        # get class weight
        neg, pos = np.bincount(labels.flatten())
        initial_bias = np.log([pos/neg])

        total = len(labels)
        weight_for_0 = (1 / neg)*(total)/2.0
        weight_for_1 = (1 / pos)*(total)/2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}
        print("class weight", class_weight)

        # create checkpoint path
        checkpoint_filepath = '{}/models/LDABERT2/simple/{}-{}-{}-pct-{}-aug_{}/no-finetune/checkpoint.ckpt'.format(
            config.root_path,
            dataset.dataset_type,
            len(saved_sentences),
            dataset.pct_data,
            dataset.augment_pct,
            random_hash)

        # # continue training
        # random_hash = "SwPSB"
        # checkpoint_filepath = '{}/models/LDABERT2/simple/wiki-40561-1-pct-0-aug_{}/no-finetune/checkpoint.ckpt'.format(
        #     config.root_path,
        #     random_hash
        # )

        # model_filepath = "/".join(checkpoint_filepath.split("/")[:-1]) + "/model.h5"
        print(checkpoint_filepath)

        # get callbacks ready.
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            save_best_only=True,
            mode="auto",
            save_freq="epoch")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

        log_path = '{}/models/LDABERT2/simple/{}-{}-{}-pct-{}-aug_{}/no-finetune/training.log'.format(
            config.root_path,
            dataset.dataset_type,
            len(saved_sentences),
            dataset.pct_data,
            dataset.augment_pct,
            random_hash)

        csv_logger = tf.keras.callbacks.CSVLogger(
            log_path, separator=",", append=False)

        callbacks = [
            #     early_stopping,
            model_checkpoint_callback,
            csv_logger
        ]

        # compiling model
        print("compiling the model...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy')
        ])

        try:
            model.load_weights(checkpoint_filepath)
            print("model loaded.")
        except:
            print("No checkpoint available.")

        print("starting the training process...")
        history = model.fit([
            left_input, mid_input, right_input,
            lda_left_input, lda_mid_input, lda_right_input
        ],
            tf.convert_to_tensor(saved_labels),
            validation_data=([
                testing_left_input, testing_mid_input, testing_right_input,
                testing_lda_left_input, testing_lda_mid_input, testing_lda_right_input
            ], tf.convert_to_tensor(testing_saved_labels)),
            epochs=epochs,
            # validation_split=0.25,
            batch_size=BATCH_SIZE,
            verbose=1,
            class_weight=class_weight,
            callbacks=callbacks)

        # assigning history to experiment object for saving.
        experiment["history"] = history.history
        experiment["hash"] = random_hash

        print("saving results...")
        save_results(experiment)
