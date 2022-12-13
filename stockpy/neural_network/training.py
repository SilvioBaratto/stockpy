import argparse
import datetime
import hashlib
import os
import shutil
import sys

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from StockPredictorLSTM import StockDataset, StockPredictorLSTM

class Training():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=24,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=48,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--dataset',
            help="What to dataset to feed the model.",
            action='store',
            default='LunaDataset',
        )
        parser.add_argument('--model',
            help="What to model class name to use.",
            action='store',
            default='StockPredictorLSTM',
        )
        parser.add_argument('--malignant',
            help="Train the model to classify nodules as benign or malignant.",
            action='store_true',
            default=False,
        )

        self.cli_args = parser.parse_args(sys_argv)
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model_cls = getattr(model, self.cli_args.model)
        cls_model = model_cls()

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in cls_model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")
            cls_model.load_state_dict(
                {
                    k: v for k,v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in cls_model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                cls_model = nn.DataParallel(cls_model)
            cls_model = cls_model.to(self.device)
        return cls_model

    def initOptimizer(self):
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        #return Adam(self.model.parameters(), lr=3e-4)

    def initTrainDl(self):
        ds_cls = getattr(dataset, self.cli_args.dataset)    # our custom dataset

        train_ds = ds_cls(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=1,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(  # An. off-the-shelf class
            train_ds,
            batch_size=batch_size,  # batching is done automatically
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,   # Pinned memory transfers to GPU quickly
        )

        return train_dl

    def initValDl(self):
        ds_cls = getattr(dataset, self.cli_args.dataset)

        val_ds = ds_cls(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix,
                                   self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()   # The validation data loader is very similar to training

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # TODO: this 'cls' will need to change for the malignant classifier
                self.saveModel('cls', epoch_ndx, score == best_score)


        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()


    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros( # Initilizes an empty metrics array
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate( # sets up our batch looping with time estimate
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()  # Frees any leftover gradient tensors

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                augment=True
            )

            loss_var.backward()     # Actually updates the model weights
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')


    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()   # Turns off training-time behaviour
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    augment=False
                )

        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         augment=True):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)


        if augment:
            input_g = model.augment3d(input_g)

        logits_g, probability_g = self.model(input_g)   

        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1],   # Reduction = 'none' gives the loss per sample
                                             reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                                   out=None)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g

        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:,1]

        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()    # This is the loss over the entire batch


    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'


        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())    # convert to a normal Python Integer
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float64(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0, steps=89)
        
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:]-fpr[:-1]
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(  
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, " # format string updated
                 + "{pr/recall:.4f} recall, "       # format string updated
                 + "{pr/f1_score:.4f} f1 score, "   # format string updated
                 + "{auc:.4f} auc"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + neg,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + pos,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        if not self.cli_args.malignant:
            score = metrics_dict['pr/f1_score']
        else:
            score = metrics_dict['auc']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            '..',
            '..',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                '..',
                '..',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':
    ClassificationTrainingApp().main()