import itertools
import numpy as np
import torch
from stockpy.neural_network import *
from stockpy.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import openml
import pandas as pd
import time
import json
import os
import argparse
from sklearn.metrics import f1_score
import pickle

early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        threshold=0.01,
        threshold_mode='rel',
        lower_is_better=True
    )

def load_dataset_and_preprocess_target(dataset_id):
    """
    Loads a dataset from OpenML and preprocesses the target variable by mapping
    categorical values to integers. Uses caching to avoid re-downloading and 
    re-processing if the dataset has already been processed.

    Parameters:
    dataset_id (int): The OpenML dataset ID.

    Returns:
    X (DataFrame): The features dataset.
    y (Series): The preprocessed target dataset.
    attribute_names (list): List of attribute names.
    """
    cache_file = f'dataset_{dataset_id}.pkl'

    # Check if the data is already cached
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            X, y, attribute_names = pickle.load(f)
    else:
        # Load the dataset
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

        # Check if the target variable is categorical and map it to integers
        if isinstance(y.dtype, pd.CategoricalDtype):
            # Get unique categories and sort them
            categories = sorted(y.unique())
            # Create a mapping from categories to integers
            mapping = {category: i for i, category in enumerate(categories)}
            # Apply the mapping
            y = y.replace(mapping)

        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump((X, y, attribute_names), f)

    return X, y, attribute_names

def train_model(X_train, y_train, X_test, y_test, model_name, **kwargs):


    predictors = {
        'mlp': MLPClassifier,
        'cnn': CNNClassifier,
        'lstm': LSTMClassifier,
        'bilstm': BiLSTMClassifier,
        'gru': GRUClassifier,
        'bigru': BiGRUClassifier,
    }

    # Extract training parameters and remove them from kwargs if present
    learning_rate = kwargs.pop('learning_rate', None)
    batch_size = kwargs.pop('batch_size', None)

    if learning_rate is None or batch_size is None:
        raise ValueError("Required training parameters 'learning_rate' or 'batch_size' not provided")
        
    # Configure the model with specific hyperparameters
    model_class = predictors[model_name]
    predictor = model_class(**kwargs)  

    # Training process
    predictor.fit(X_train, 
                  y_train, 
                  batch_size=batch_size, 
                  lr=learning_rate, 
                  optimizer=torch.optim.Adam, 
                  callbacks=[early_stopping],
                  epochs=100)

    # Example: Extracting validation loss as a metric
    completed_epochs = len(predictor.history)
    losses = [predictor.history[i]['valid_loss'] for i in range(completed_epochs)]

    # Extracting the best validation accuracy
    best_epoch_info = max(enumerate(predictor.history), key=lambda x: (x[1]['valid_acc_best'], x[1]['valid_acc']))
    best_epoch = best_epoch_info[0]
    valid_best_acc = best_epoch_info[1]['valid_acc']

    y_pred = predictor.predict(X_test)
    # Calculate average loss over all epochs
    average_loss = sum(losses) / len(losses)

    score_f1 = f1_score(y_test, y_pred, average='weighted')
    
    return predictor, average_loss, valid_best_acc, best_epoch, score_f1, losses

def main(dataset_id, name_file, model_name, **kwargs):
    X, y, attribute_names = load_dataset_and_preprocess_target(dataset_id)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)

    check_rnn = ['lstm', 'bilstm', 'gru', 'bigru']

    # Define hyperparameter search space
    hidden_layer_sizes = [2**n for n in range(8, 11)] if model_name not in check_rnn else [32]
    # learning rate - log uniform between 1e-5 and 1e-1
    learning_rates = [10 ** -i for i in range(1, 5)]
    # Batch size - uniform choice
    batch_sizes = [2 ** i for i in range(9, 12)]
    # Define hyperparameter search space
    rnn_sizes = [2**n for n in range(8, 11)]
    # sequence length
    seq_lengths = [60] # [30 * i for i in range(1, 3)]
    # number of layers
    num_layers = [1, 2]

    model_params = {
        'mlp': ['hidden_size', 'learning_rate', 'batch_size'],
        'cnn': ['hidden_size', 'learning_rate', 'batch_size'],
        'lstm': ['rnn_size', 'hidden_size', 'seq_len', 'num_layers', 'learning_rate', 'batch_size'],
        'bilstm': ['rnn_size', 'hidden_size', 'seq_len', 'num_layers', 'learning_rate', 'batch_size'],
        'gru': ['rnn_size', 'hidden_size', 'seq_len', 'num_layers', 'learning_rate', 'batch_size'],
        'bigru': ['rnn_size', 'hidden_size', 'seq_len', 'num_layers', 'learning_rate', 'batch_size'],
    }

    # Create hyperparameter grid based on model_name
    hyperparameter_combinations = {
        'mlp': itertools.product(hidden_layer_sizes, learning_rates, batch_sizes),
        'cnn': itertools.product(hidden_layer_sizes, learning_rates, batch_sizes),
        'lstm': itertools.product(rnn_sizes, hidden_layer_sizes, seq_lengths, num_layers,  learning_rates, batch_sizes),
        'bilstm': itertools.product(rnn_sizes, hidden_layer_sizes, seq_lengths, num_layers, learning_rates, batch_sizes),
        'gru': itertools.product(rnn_sizes, hidden_layer_sizes, seq_lengths, num_layers, learning_rates, batch_sizes),
        'bigru': itertools.product(rnn_sizes, hidden_layer_sizes, seq_lengths, num_layers, learning_rates, batch_sizes),
    }

    hyperparameter_grid = hyperparameter_combinations.get(model_name)

    name_file = f"{name_file}_{model_name}.json"

    with open(name_file, 'w') as log_file:
        for params in hyperparameter_grid:
            start_time = time.time()
            
            model_args = dict(zip(model_params[model_name], params))
            model, average_loss, best_val_acc, best_epoch, score_f1, losses = train_model(
                X_train, y_train, X_test, y_test, model_name, **model_args
            )

            end_time = time.time()
            time_taken = end_time - start_time

            log_data = {
                'params': model_args,
                'average_loss': average_loss,
                'best_validation_accuracy': best_val_acc,
                'best_epoch': best_epoch,
                'losses': losses,
                'f1_score': score_f1,
                'time': time_taken
            }

            log_file.write(json.dumps(log_data) + '\n')

if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train a model on a dataset from OpenML.')
    
    # Add the dataset_id argument
    parser.add_argument('dataset_id', type=int, help='The OpenML dataset ID to use.')

    # Add the name_file argument
    parser.add_argument('name_file', type=str, help='The name of the file to log the results.')

    # Add the model argument
    parser.add_argument("--model", type=str, default="mlp", help="Type of the model to train (e.g., 'mlp', 'cnn')")

    # Parse the arguments
    args = parser.parse_args()

    # Make sure model is a string
    args.model = str(args.model)

    # Make sure name_file is a string
    args.name_file = args.model + '/' + str(args.name_file)

    # Call the main function with the dataset_id argument
    main(args.dataset_id, args.name_file, args.model)