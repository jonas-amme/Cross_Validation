from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import os 
import tqdm 
import sqlite3
import pickle
import wandb
import argparse

from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torchmetrics import MetricCollection, Accuracy, AUROC, F1Score, Precision, Recall

from src.model import Mitosis_Classifier



def save_splits(
    split_datasets: Tuple[Dataset],
    column_keys: Tuple[str], 
    filename: str) -> None:
    """Saves the slide indices of the splits together into one dataframe.

    Args:
        split_datasets (Tuple[Dataset]]): Train, val and test datasets.
        column_keys (Tuple[str]): Column names (e.g. train, val, test).
        filename (str): File name for the dataframe.
    """
    splits = [pd.Series(split_datasets[i].data.slide.unique()) for i in range(len(split_datasets))]
    df = pd.concat(splits, ignore_index=True, axis=1)
    df.columns = column_keys
    df.to_csv(filename)



class EarlyStopping:
    """EarlyStopping callback class

    Args:
        patience (int, optional): Number of epochs to stop after last val loss improved. Defaults to 20.
        stop_epoch (int, optional): Earliest epoch possible for stopping. Defaults to 50.
        verbose (bool, optional): If True, prints message for each val loss improvement. Defaults to False.
    """
    def __init__(
        self,
        patience: int = 20,
        stop_epoch: int = 50,
        verbose: bool = False) -> None:
   	
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(
        self,
        epoch: int,
        val_loss: torch.Tensor,
        model: nn.Module,
        ckpt_name: str = 'checkpoint.pt') -> None:
        """EarlyStopping call function. 

        Args:
            epoch (int): Current epoch.
            val_loss (torch.Tensor): Validation loss.
            model (nn.Module): Pytorch model.
            ckpt_name (str, optional): Name of checkpoint. Defaults to 'checkpoint.pt'.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0


    def save_checkpoint(
        self, 
        val_loss: torch.Tensor,
        model: nn.Module,
        ckpt_name: str = 'checkpoint.pt') -> None:
        """Save checkpoint function. 

        Saves checkpoints to `ckpt_name`. If `verbose` is True, then prints 
        a message for each validation loss improvement. 

        Args:
            val_loss (torch.Tensor): Validation loss.
            model (nn.Module): Pytorch model
            ckpt_name (str, optional): Name of checkpoint. Defaults to 'checkpoint.pt'.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def calculate_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return 1. - y_hat.float().eq(y.float()).float().mean().item()


def print_model(model: nn.Module) -> None:
    """Prints total number of params and number of trainable params.

    Args:
        model (nn.Module): Pytorch model
    """
    num_params = 0
    num_params_train = 0
    print(model)
    for param in model.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def get_optimizer(model: nn.Module, args: argparse.ArgumentParser):
    """Initializes optimizer

    Args:
        model (nn.Module): Pytorch model.
        args (argparse.ArgumentParser): Configurations

    Raises:
        NotImplementedError: If optimizer is not suppported.
    """
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def get_split_loader(
    split_dataset: Dataset,
    args: argparse.ArgumentParser,
    collante_fn: callable,
    training: bool = False) -> DataLoader:
    """Returns either training or validation dataloader.

    If `training` is True, returns a RandomSampler and a SequentialSampler otherwise.

    Args:
        split_dataset (Dataset): Pytorch dataset
        args (argparser): Configurations.
        collante_fn (callable): Collate function.
        training (bool, optional): If true, sets shuffle to True. Defaults to False.

    Returns:
        DataLoader: Dataloader
    """
    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'collate_fn': collante_fn
        }
    if training:
        kwargs.update({'shuffle': True})
        loader = DataLoader(split_dataset, **kwargs)
    else:
        kwargs.update({'shuffle': False})
        loader = DataLoader(split_dataset, **kwargs)
    return loader 


def print_summary(split: str, results: Dict) -> None:
    """Prints each result from a dictionary."""
    print('{} results:'.format(split))
    for metric, value in results.items():
        print('{}: {:.4f}'.format(metric, value))


def train(
    run_id: int, 
    datasets: Tuple[Dataset], 
    args: argparse.ArgumentParser):
    """Training function to train a single fold.

    Args:
        run_id (int): Current fold.
        datasets (Tuple[Dataset]): Train, val, test datasets.
        args (argparse.ArgumentParser): Training configurations.
    """
    print('\nTraining fold {}!'.format(run_id))
    if args.logging:
        run = wandb.init(reinit=True, project=args.project_name, name='_'.join([str(run_id).zfill(2), args.exp_code]))

    print('\nInitialize train/val/test splits ...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.result_dir, 'splits_{}'.format(run_id)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    # set loss 
    loss_fn = nn.functional.binary_cross_entropy_with_logits

    # init model
    print('\nInitialize model ...', end=' ')
    model = Mitosis_Classifier(model=args.model, weights=args.weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    print('Done!')
    print_model(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optimizer(model, args)
    print('Done!')

    print('\nInitialize dataloader ...', end=' ')
    collate_fn = train_split.collate_fn
    train_loader = get_split_loader(train_split, args, collante_fn=collate_fn, training=True)
    val_loader = get_split_loader(val_split, args, collante_fn=collate_fn, training=False)
    test_loader = get_split_loader(train_split, args, collante_fn=collate_fn, training=False)
    print('Done!')

    print('\nInitialize EarlyStopping ...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, stop_epoch=args.stop_epoch, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        # train one epoch
        train_loop(epoch, model, train_loader, optimizer, loss_fn, args.calculate_metrics, args.logging)
        
        # validate
        stop = eval_loop(run_id, epoch, model, val_loader, loss_fn, early_stopping, args.result_dir, args.calculate_metrics, args.logging)

        if stop:
            break 

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.result_dir, 's_{}_checkpoint.pt'.format(run_id))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(run_id)))

    val_metrics = summary(model, val_loader, calculate_metrics=True, logging_prefix='final/val_')
    print_summary(split='Validation', results=val_metrics)

    test_metrics = summary(model, test_loader, calculate_metrics=True, logging_prefix='final/test_')
    print_summary(split='Test', results=test_metrics)

    if args.logging:
        wandb.log(val_metrics)
        wandb.log(test_metrics)

    if args.logging:
        run.finish()
    return val_metrics, test_metrics


def train_loop(
    epoch: int, 
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.functional,
    calculate_metrics: bool = False, 
    logging: bool = True,
    reload_patches: bool = True) -> None:

    if reload_patches and epoch > 0:
        loader.dataset.resample_patches()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    if calculate_metrics:
        metrics = MetricCollection(
            [Accuracy(task='binary'), 
            AUROC(task='binary'), 
            F1Score(task='binary'),
            Precision(task='binary'), 
            Recall(task='binary')],
            prefix='train/')
        metrics = metrics.to(device)

    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (images, labels) in enumerate(loader): 
        images, labels = images.to(device), labels.to(device)

        # forward pass 
        logits, _, Y_hat = model(images)
        loss = loss_fn(logits, labels.float())
        loss_value = loss.item()
        train_loss += loss_value
        if (batch_idx + 1) % 10 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss_value))
        error = calculate_error(Y_hat, labels)
        train_error += error

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # collect metrics
        if calculate_metrics:
            metrics.update(logits, labels)

    train_loss /= len(loader)
    train_error /= len(loader)
    
    print('Epoch {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

    if logging:
        wandb.log({
            'train_loss': train_loss, 
            'train_error': train_error})
        if calculate_metrics:
            res = metrics.compute()
            wandb.log(res)


def eval_loop(
    run_id: int, 
    epoch: int, 
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.functional,
    early_stopping: EarlyStopping = None,
    result_dir: str = None,
    calculate_metrics: bool = True,
    logging: bool = True,
    logging_prefix: str = 'val/') -> bool:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if calculate_metrics:
        metrics = MetricCollection(
            [Accuracy(task='binary'), 
            AUROC(task='binary'), 
            F1Score(task='binary'),
            Precision(task='binary'), 
            Recall(task='binary')],
            prefix=logging_prefix)
        metrics = metrics.to(device)

    val_loss = 0.
    val_error = 0.

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # forward pass 
            logits, _, Y_hat = model(images)
            loss = loss_fn(logits, labels.float())
            loss_value = loss.item()
            val_loss += loss_value
            error = calculate_error(Y_hat, labels)
            val_error += error

            # collect metrics
            if calculate_metrics:
                metrics.update(logits, labels)

    val_error /= len(loader)
    val_loss /= len(loader)

    print('Val set, val_loss: {:.4f}, val_error: {:.4f}'.format(val_loss, val_error))

    if logging:
        wandb.log({
            logging_prefix + 'loss': val_loss,
            logging_prefix + 'error': val_error
        })
        if calculate_metrics:
            res = metrics.compute()
            wandb.log(res)

    if early_stopping:
        assert result_dir 
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(result_dir, 's_{}_checkpoint.pt'.format(run_id)))
        
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False


def summary(
    model: nn.Module,
    loader: DataLoader,
    calculate_metrics: bool = True,
    logging_prefix: str = 'test/') -> Dict[str, float]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if calculate_metrics:
        metrics = MetricCollection(
            [Accuracy(task='binary'), 
            AUROC(task='binary'), 
            F1Score(task='binary'),
            Precision(task='binary'), 
            Recall(task='binary')],
            prefix=logging_prefix)
        metrics = metrics.to(device)

    test_error = 0.

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # forward pass 
            logits, _, Y_hat = model(images)
            error = calculate_error(Y_hat, labels)
            test_error += error

            # collect metrics
            if calculate_metrics:
                metrics.update(logits, labels)

    test_error /= len(loader)

    if calculate_metrics:
        res = metrics.compute()
        res = {k: v.cpu().item() for k, v in res.items()}
        res.update({logging_prefix + 'error': test_error})
        return res
            
    else:
        return {logging_prefix + '_error': test_error}  


def save_pkl(filename: str, object: Any) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def load_pkl(filename: str) -> Any:
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file 



def initiate_model(
    model: str,
    ckpt_path: str
    ) -> Mitosis_Classifier:
    """Load a Mitosis Classifier with weights from ckpt_path.

    Args:
        model (str): Model type.
        ckpt_path (str): Path to weights.

    Returns:
        Mitosis_Classifier: 
    """
    # build model 
    model = Mitosis_Classifier(model=model, num_classes=1)
    # print_model(model)

    # load weights
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        ckpt_clean.update({key.replace('module.', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.eval()
    return model 


