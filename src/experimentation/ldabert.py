from utils.experiments import get_experiments_json, get_experiments, save_results, print_break
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
import csv

import tensorflow as tf
import random
import string

from utils.metrics import windowdiff, pk
from collections import OrderedDict

sys.path.insert(0, config.root_path)


def get_random_hash(k):
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1
    return (TP, FP, TN, FN)


def get_precision(TP, FP, TN, FN):
    if TP == 0:
        return 0
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def get_recall(TP, FP, TN, FN):
    if TP == 0:
        return 0
    if TP + FP == 0:
        return 0
    return TP / (TP + FN)


class SimpleExperiment:
    def __init__(self, *, experiment_string: str, override_epochs: int = None, experiment_hash: str = None, msl: int = 300, batch_size: int = 32, training_lda_gamma: int = None, predictions_lda_gamma: int = 15):
        """Generate a new experiment with the following params

        Args:
            experiment_string (str): string that corresponds to the experiment id in settings/experiments.json
            override_epochs (int, optional): if overriding epochs is desired, provide here. Defaults to None.
            experiment_hash (str, optional): if continuation of training, provide this hash. Defaults to None
            msl (int, optional): max segment length for segment fabrication. Defaults to 300.
            batch_size (int, optional): batch size to train if overriding. Defaults to 32.
            training_lda_gamma (int, optional): gamma used to apply to training LDA words. Defaults to None.
            predictions_lda_gamma (int, optional): gamma used to apply to predictions LDA influence. Defaults to 15.
        """
        assert experiment_string is not None, "experiment string should be provided. E.g., 'qmsum_ldabert2_simple_test'"
        self.experiment_string = experiment_string
        self.experiment_hash = experiment_hash
        self.override_epochs = override_epochs
        self.batch_size = batch_size
        self.msl = msl
        self.training_lda_gamma = training_lda_gamma
        self.predictions_lda_gamma = predictions_lda_gamma

    def evaluate(self, random_hash):
        print("running evaluations")
        predictions_directory = "{}/predictions/{}".format(
            config.root_path, random_hash)
        predictions_path = "{}/predictions.csv".format(predictions_directory)

        predictions = []
        reader = csv.DictReader(open(predictions_path, encoding="utf8"))
        for row in reader:
            predictions.append(row)

        if type(predictions[0]) is OrderedDict:
            predictions = [dict(x) for x in predictions]
        prediction_thresholds = [
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            0.9999,
        ]

        full_preds = []
        for p in predictions:
            curr_preds = []
            for thresh in prediction_thresholds:
                curr_preds.append(1 if float(p["logit"]) > thresh else 0)
            full_preds.append([*curr_preds, p["ground_truth"]])

        columns = [*prediction_thresholds, "ground_truth"]
        data = full_preds

        df = pd.DataFrame(data, columns=columns)

        ground_truth = [int(k) for k in df["ground_truth"].tolist()]
        new_predictions = []

        for p_t in prediction_thresholds:
            new_predictions.append(df[p_t].tolist())

        df_data = []
        string_ground_truth = "".join([str(s) for s in ground_truth])

        # k values
        evaluation_windows = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]  # [2, 5, 7, 10, 14]
        for evaluation_window in evaluation_windows:
            for thresh, pred_set in zip(prediction_thresholds, new_predictions):
                string_pred_set = "".join([str(s) for s in pred_set])

                pk_result = pk(string_pred_set,
                               string_ground_truth, evaluation_window)
                window_diff_result = windowdiff(
                    string_pred_set, string_ground_truth, evaluation_window
                )

                TP, FP, TN, FN = perf_measure(ground_truth, pred_set)
                precision = get_precision(TP, FP, TN, FN)
                recall = get_recall(TP, FP, TN, FN)

                # create the dataframe
                df_data.append(
                    [
                        evaluation_window,
                        thresh,
                        TP, FP, TN, FN,
                        precision,
                        recall,
                        pk_result,
                        window_diff_result,
                    ]
                )

                # print(thresh, (TP, FP, TN, FN), precision, recall)

        df_evaluation_set = pd.DataFrame(
            df_data, columns=["K", "Threshold", "TP", "FP", "TN",
                              "FN", "precision", "recall", "Pk", "WindowDiff"]
        )
        evaluation_file = "{}/eval.csv".format(predictions_directory)

        try:
            # if not os.path.exists(os.path.dirname(evaluation_file)):
            #     os.makedirs(os.path.dirname(evaluation_file))
            with open(evaluation_file, 'w') as csvfile:
                df_evaluation_set.to_csv(evaluation_file, index=False)
        except IOError:
            print("I/O error")

        print_break("EXPERIMENT COMPLETE")

    def predict(self, model, dataset_type: str, random_hash: str):
        print("Conducting predictions - initializing testing dataset...")
        slice_size = 125
        predictions_log = []
        log_file = '{}/predictions/{}/predictions.csv'.format(
            config.root_path, random_hash)
        testing_dataset = LDABERT3Dataset(dataset_type=dataset_type,
                                          dataset_slice="testing",
                                          pct_data=1,
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

        for i in range(0, testing_saved_vectors.shape[0]//slice_size):
            start = i*slice_size
            end = start + slice_size

            sliced_tokenized_sentences = {
                'input_ids': testing_saved_tokenized_sentences['input_ids'][start:end],
                'token_type_ids': testing_saved_tokenized_sentences['token_type_ids'][start:end],
                'attention_mask': testing_saved_tokenized_sentences['attention_mask'][start:end]
            }
            sliced_vectors = testing_saved_vectors[start:end]

            left_input, mid_input, right_input = testing_dataset.format_sentences_tri_input_plus(
                sliced_tokenized_sentences)
            lda_left_input, lda_mid_input, lda_right_input = testing_dataset.format_sentences_tri_input(
                sliced_vectors)

            logits = model([
                left_input,
                mid_input,
                right_input,
                lda_left_input,
                lda_mid_input,
                lda_right_input
            ])

            logits_flattened = [float(p) for p in logits]

            pred_threshold = 0.5
            predictions = [1 if float(
                logit) >= pred_threshold else 0 for logit in logits_flattened]

            k = 14

            # # Calculate only if want to print out
            # string_predictions = "".join([str(i) for i in predictions])
            # string_ground_truth = "".join(
            #     [str(i) for i in testing_saved_labels[start:end]])

            # overall_windowdiff = windowdiff(
            #     string_predictions, string_ground_truth, k)
            # overall_pk = pk(string_predictions, string_ground_truth, k)

            for pred_idx, pred in enumerate(predictions):
                log = {
                    "logit": logits_flattened[pred_idx],
                    "prediction": pred,
                    "ground_truth": testing_saved_labels[start:end][pred_idx],
                    "text": testing_saved_sentences[start:end][pred_idx],
                    "lda_gamma": self.predictions_lda_gamma,
                    "prediction_threshold": pred_threshold,
                }
                predictions_log.append(log)

        try:
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
            mode = 'a' if os.path.exists(log_file) else 'w'
            with open(log_file, mode) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                                        "prediction", "ground_truth", "text", "lda_gamma", "prediction_threshold", "logit"])
                writer.writeheader()
                for data in predictions_log:
                    writer.writerow(data)
            print("predictions complete...")
        except IOError:
            print("I/O error")

        self.evaluate(random_hash)

    def run(self, cast_predictions: bool = True, start: int = None, end: int = None):
        """Run experiments

        Args:
            cast_predictions (bool, optional): Determines whether to run predictions after training. Defaults to True.
            start (int, optional): Overrides experiments start and end (e.g., run the first two experiments only - 0, 2). Defaults to None.
            end (int, optional): Overrides experiments start and end (e.g., run the first two experiments only - 0, 2). Defaults to None.
        """
        experiments_config = get_experiments_json(self.experiment_string)

        assert len(
            experiments_config) > 0, "There should be at least one experiment"

        if start is None:
            start = 0
        if end is None:
            end = len(experiments_config)

        for experiment in experiments_config[start:end]:
            dataset_type = experiment['dataset_type']
            final_dropout = experiment['final_dropout']
            dense_neurons = experiment['dense_neurons']
            pct_data = experiment['pct_data']
            augment_pct = experiment['augment_pct']
            lda_gamma = self.training_lda_gamma if self.training_lda_gamma is not None else experiment[
                'gamma']
            max_sentence_length = experiment['max_sentence_length']
            bert_trainable = experiment['bert_trainable']
            epochs = self.override_epochs if self.override_epochs is not None else experiment[
                'epochs']
            print("training params:", experiment)
            random_hash = get_random_hash(5)

            # init model
            print("initializing model...")
            model = ContextEncoderSimple(final_dropout=final_dropout,
                                         dense_neurons=dense_neurons,
                                         gamma=lda_gamma,
                                         max_sentence_length=max_sentence_length,
                                         bert_trainable=bert_trainable)

            # print("number of params: ", sum([np.prod(keras.get_value(w).shape) for w in model.trainable_weights]))

            # init training dataset
            print("initializing dataset...")
            dataset = LDABERT3Dataset(dataset_type=dataset_type,
                                      pct_data=pct_data,
                                      max_seq_length=max_sentence_length,
                                      max_segment_length=300,
                                      augment_pct=augment_pct,
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
            print("initializing validation dataset...")
            testing_dataset = LDABERT3Dataset(dataset_type=dataset_type,
                                              pct_data=0.25,
                                              max_seq_length=1024,
                                              max_segment_length=300,
                                              augment_pct=0,
                                              split="test",
                                              artificial_segments=False)

            # process testing dataset
            print("processing validation dataset...")
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

            # set to continue training
            if self.experiment_hash is not None:
                random_hash = self.experiment_hash
                print(f"continuing training with {self.experiment_hash}")

            # create checkpoint path
            checkpoint_filepath = '{}/models/LDABERT2/simple/{}-{}-{}-pct-{}-aug_{}/{}finetune/checkpoint.ckpt'.format(
                config.root_path,
                dataset.dataset_type,
                len(saved_sentences),
                dataset.pct_data,
                dataset.augment_pct,
                random_hash,
                '' if bert_trainable is True else 'no-')

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
                batch_size=self.batch_size,
                verbose=1,
                class_weight=class_weight,
                callbacks=callbacks)

            # assigning history to experiment object for saving.
            experiment["history"] = history.history
            experiment["hash"] = random_hash
            experiment["epochs"] = epochs

            print("saving results...")
            save_results(experiment)

            if cast_predictions:
                self.predict(model, dataset_type, random_hash)
