#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil

import logging

logger = tf.get_logger()
logger.setLevel(logging.INFO)

# List the CSV columns
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']

#Choose which column is your label
LABEL_COLUMN = 'fare_amount'

# Set the default values for each CSV column in case there is a missing value
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# Create an input function that stores your data into a dataset
# DONE: Add input function

# DONE: Create an appropriate input function read_dataset
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():  
        def decode_csv(row):
            columns = tf.io.decode_csv(row, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            features.pop('key') # not needed, since it's not a feature
            label = features.pop('fare_amount') # remove label from features and store
            return features, label

        # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
        filenames_dataset = tf.data.Dataset.list_files(filename, shuffle=False)
        # Read lines from text files
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
        # Parse text lines as comma-separated values (CSV)
        dataset = textlines_dataset.map(decode_csv)

        # Note:
        # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
        # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # loop indefinetly
            dataset = dataset.shuffle(buffer_size = 10 * batch_size, seed = 42)
        else:
            num_epochs = 1 # one run and it's over

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset
    
    return _input_fn

# Define your feature columns
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

# Create a function that will augment your feature set
def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn2():
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(INPUT_COLUMNS))
    serving_input_receiver = serving_input_fn()
    return serving_input_receiver

def serving_input_fn():
    feature_placeholders = {
        column.name: tf.compat.v1.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }

    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# Create an estimator that we are going to train and evaluate
def train_and_evaluate(args):
    tf.compat.v1.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    #tf.python.summary.FileWriterCache.clear()
    
    # DONE: Create tf.estimator.DNNRegressor train and evaluate function passing args['parsed_argument'] from task.py
    opti = tf.keras.optimizers.Adam(learning_rate = 0.01)
    estimator = tf.estimator.DNNRegressor(
        model_dir = args['output_dir'],
        feature_columns = feature_cols,
        hidden_units = args['hidden_units'])
        #, optimizer=opti,

    train_spec=tf.estimator.TrainSpec(
        input_fn = read_dataset(args['train_data_paths'], 
                                       batch_size = args['train_batch_size'],
                                       mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec=tf.estimator.EvalSpec(
        input_fn = read_dataset(args['eval_data_paths'], 
                                        batch_size = 10000,
                                        mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs = args['eval_delay_secs'], # start evaluating after N seconds
        throttle_secs = args['throttle_secs'],  # evaluate every N seconds
        exporters = exporter)
        
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
