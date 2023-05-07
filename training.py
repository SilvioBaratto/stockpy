import numpy as np
import pandas as pd
import os 
import glob
import argparse
import pathlib
from tqdm.auto import tqdm
from datetime import date, datetime
from typing import Union, Tuple
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stockpy.neural_network import *
from stockpy.probabilistic import *
from stockpy.config import Config as cfg

def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='stock/AAPL.csv', help='Path to the data file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for train/test split')
    parser.add_argument('--model', type=str, default='MLPRegressor', help='Model type')
    parser.add_argument('--label', nargs='+', type=str, default='Close', help='Label for the model')

    # Common arguments
    parser.add_argument('--hidden_size', type=int, default=cfg.comm.hidden_size, help='Size of the hidden layer')
    parser.add_argument('--num_filters', type=int, default=cfg.comm.num_filters, help='Number of filters in the convolutional layer')
    parser.add_argument('--pool_size', type=int, default=cfg.comm.pool_size, help='Pooling size')
    parser.add_argument('--kernel_size', type=int, default=cfg.comm.kernel_size, help='Kernel size in the convolutional layer')
    parser.add_argument('--dropout', type=float, default=cfg.comm.dropout, help='Dropout rate')

    # NN arguments
    parser.add_argument('--num_layers', type=int, default=cfg.nn.num_layers, help='Number of layers in the neural network')

    # cfg.prob arguments
    parser.add_argument('--rnn_dim', type=int, default=cfg.prob.rnn_dim, help='RNN dimension for cfg.probabilistic model')
    parser.add_argument('--z_dim', type=int, default=cfg.prob.z_dim, help='Latent variable dimension')
    parser.add_argument('--emission_dim', type=int, default=cfg.prob.emission_dim, help='Emission model dimension')
    parser.add_argument('--transition_dim', type=int, default=cfg.prob.transition_dim, help='Transition model dimension')
    parser.add_argument('--variance', type=float, default=cfg.prob.variance, help='Model variance')

    # Training arguments
    parser.add_argument('--lr', type=float, default=cfg.training.lr, help='Learning rate')
    parser.add_argument('--betas', nargs=2, type=float, default=cfg.training.betas, help='Betas for the optimizer')
    parser.add_argument('--lrd', type=float, default=cfg.training.lrd, help='Learning rate decay')
    parser.add_argument('--clip_norm', type=float, default=cfg.training.clip_norm, help='Gradient clipping norm')
    parser.add_argument('--weight_decay', type=float, default=cfg.training.weight_decay, help='Weight decay for the optimizer')
    parser.add_argument('--eps', type=float, default=cfg.training.eps, help='Epsilon value for the optimizer')
    parser.add_argument('--amsgrad', type=bool, default=cfg.training.amsgrad, help='Use AMSGrad for the optimizer')
    parser.add_argument('--optim_args', type=float, default=cfg.training.optim_args, help='Additional optimizer arguments')
    parser.add_argument('--gamma', type=float, default=cfg.training.gamma, help='Scheduler gamma')
    parser.add_argument('--step_size', type=float, default=cfg.training.step_size, help='Scheduler step size')
    parser.add_argument('--epochs', type=int, default=cfg.training.epochs, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.training.batch_size, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=cfg.training.sequence_length, help='Sequence length for time series data')
    parser.add_argument('--num_workers', type=int, default=cfg.training.num_workers, help='Number of workers for data loading')
    parser.add_argument('--validation_cadence', type=int, default=cfg.training.validation_cadence, help='Validation cadence')
    parser.add_argument('--patience', type=int, default=cfg.training.patience, help='Patience for early stopping')
    parser.add_argument('--prediction_window', type=int, default=cfg.training.prediction_window, help='Prediction window size')
    parser.add_argument('--scheduler', type=bool, default=cfg.training.scheduler, help='Use learning rate scheduler')

    parser.add_argument('--metrics', type=bool, default=cfg.training.metrics, help='Report metrics during training')
    parser.add_argument('--pretrained', type=bool, default=cfg.training.pretrained, help='Use pretrained model if available')
    parser.add_argument('--folder', type=str, default=cfg.training.folder, help='Folder for saving/loading model and training results')

    return parser

class Trainer:
    def __init__(self):
        pass    

    def run(self, model, args, **kwargs):
        # read CSV file and drop missing values
        df = pd.read_csv("test/ParisHousing.csv").dropna(how='any')

        # split data into training and test set
        X = df.drop(columns=['cityPartRange'])
        y = df['cityPartRange']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)

        # create model instance and fit to training data
        if args.metrics:
            train_losses, train_accuracies, train_f1_scores = model.fit(X=X_train, y=y_train, **kwargs)
        
        else:
            model.fit(X=X_train, y=y_train, **kwargs)

        accuracy, f1_score, conf_matrix = model.score(X_test, y_test)
        print("Accuracy: ", accuracy)
        print("F1 Score: ", f1_score)
        print("Confusion Matrix: \n", conf_matrix)

        # Create a DataFrame with the values
        result_df = pd.DataFrame({"Accuracy": [accuracy], "F1 Score": [f1_score]})

        # Save the confusion matrix to a CSV file
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_filename = f"results/{model.name}_confusion_matrix.csv"
        conf_matrix_df.to_csv(conf_matrix_filename, index=False)

        # Save the result DataFrame to a CSV file
        result_filename = f"results/{model.name}_results.csv"
        result_df.to_csv(result_filename, index=False)

        if args.metrics:
            self.plot_training_metrics(model, train_losses, train_accuracies, train_f1_scores)

    @staticmethod
    def plot_training_metrics(predictor, train_losses, train_accuracies, train_f1_scores):
        epochs = range(1, len(train_losses) + 1)
        sns.set(style="ticks", context="talk")
        plt.style.use('dark_background')
        file = file = 'results/classifier' + predictor.__class__.__name__ + '.png'
        fig, ax = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        ax[0].plot(epochs, train_losses, label='Training Loss', linewidth=3, color='#2962FF')
        ax[1].plot(epochs, train_accuracies, label='Training Accuracy', linewidth=3, color='#FFA726')
        ax[2].plot(epochs, train_f1_scores, label='Training F1 Score', linewidth=3, color='#4CAF50')

        ax[0].set_title('Training Loss', fontsize=24, color='white')
        ax[1].set_title('Training Accuracy', fontsize=24, color='white')
        ax[2].set_title('Training F1 Score', fontsize=24, color='white')

        for i in range(3):
            ax[i].set_xlabel('Epoch', fontsize=16, color='white', labelpad=10)
            ax[i].set_ylabel('Value', fontsize=16, color='white', labelpad=10)
            ax[i].grid(color='white', linestyle='--', linewidth=0.5)
            ax[i].legend(fontsize=16, loc='upper right', edgecolor='black')

        plt.tight_layout()
        plt.savefig(file, dpi=80, facecolor='#131722')
        plt.show()

def main(args):
    # Create a dictionary with training arguments
    training_kwargs = {
        'lr': args.lr,
        'betas': args.betas,
        'lrd': args.lrd,
        'clip_norm': args.clip_norm,
        'weight_decay': args.weight_decay,
        'eps': args.eps,
        'amsgrad': args.amsgrad,
        'optim_args': args.optim_args,
        'gamma': args.gamma,
        'step_size': args.step_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'num_workers': args.num_workers,
        'validation_cadence': args.validation_cadence,
        'patience': args.patience,
        'prediction_window': args.prediction_window,
        'scheduler': args.scheduler,
        'metrics': args.metrics,
        'pretrained': args.pretrained,
        'folder': args.folder,
    }

    model_dict = {
        'GRUClassifier': GRUClassifier,
        'LSTMClassifier': LSTMClassifier,
        'CNNClassifier': CNNClassifier,
        'BiGRUClassifier': BiGRUClassifier,
        'BiLSTMClassifier': BiLSTMClassifier,
        'MLPClassifier': MLPClassifier,
        'BayesianNNClassifier': BayesianNNClassifier,
        'BayesianCNNClassifier': BayesianCNNClassifier,
        'GRURegressor': GRURegressor,
        'LSTMRegressor': LSTMRegressor,
        'CNNRegressor': CNNRegressor,
        'BiGRURegressor': BiGRURegressor,
        'BiLSTMRegressor': BiLSTMRegressor,
        'MLPRegressor': MLPRegressor,
        'BayesianNNRegressor': BayesianNNRegressor,
        'BayesianCNNRegressor': BayesianCNNRegressor,
        'DeepMarkovModelRegressor': DeepMarkovModelRegressor,
        'GaussianHMMRegressor': GaussianHMMRegressor,
    }

    # Create an instance of Trainer and CNNClassifier
    trainer = Trainer()
    model = model_dict[args.model]()

    # Call the run method on the trainer instance
    trainer.run(model, args, **training_kwargs)

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)
