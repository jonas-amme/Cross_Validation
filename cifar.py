import argparse
from typing import Any, Dict
import pandas as pd
import numpy as np
import os 
import torchvision.transforms as T
import pprint
import torch.nn as nn
import wandb
import pickle 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from torchmetrics import MetricCollection, Accuracy, AUROC, F1Score, Precision, Recall


from src.utils import EarlyStopping
from src.splitting import MonteCarloCV, KfoldCV
from src.model import CIFAR_Classifier



def train_loop(epoch,model,loader,optimizer,loss_fn,calculate_metrics=True,logging=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    if calculate_metrics:
        metrics = MetricCollection(
            [Accuracy(task='multiclass', num_classes=10), 
            AUROC(task='multiclass',num_classes=10), 
            F1Score(task='multiclass', num_classes=10),
            Precision(task='multiclass', num_classes=10), 
            Recall(task='multiclass', num_classes=10)],
            prefix='train/')
        metrics = metrics.to(device)

    train_loss,train_correct=0.0,0

    for images, labels in loader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        logits, Y_prob, Y_hat = model(images)
        loss = loss_fn(logits,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        train_correct += (Y_hat == labels).sum().item()

        if calculate_metrics:
            metrics.update(logits,labels)

    train_loss /= len(loader.sampler)
    train_correct /= len(loader.sampler) 
    
    print('Epoch {}, train_loss: {:.4f}, train_correct: {:.4f}'.format(epoch, train_loss, train_correct))

    if logging:
        wandb.log({
            'train_loss': train_loss, 
            'train_correct': train_correct})
        if calculate_metrics:
            res = metrics.compute()
            wandb.log(res)


def eval_loop(run_id,epoch,model,loader,loss_fn,early_stopping,result_dir,calculate_metrics=True,logging=True,logging_prefix='val/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if calculate_metrics:
        metrics = MetricCollection(
            [Accuracy(task='multiclass', num_classes=10), 
            AUROC(task='multiclass',num_classes=10), 
            F1Score(task='multiclass', num_classes=10),
            Precision(task='multiclass', num_classes=10), 
            Recall(task='multiclass', num_classes=10)],
            prefix=logging_prefix)
        metrics = metrics.to(device)

    val_loss,val_correct=0.0,0

    for images, labels in loader:

        images,labels = images.to(device),labels.to(device)
        logits, Y_prob, Y_hat = model(images)
        loss=loss_fn(logits,labels)
        val_loss+=loss.item()*images.size(0)
        val_correct+=(Y_hat == labels).sum().item()

        if calculate_metrics:
            metrics.update(logits,labels)
    
    val_correct /= len(loader.sampler) 
    val_loss /= len(loader.sampler)

    print('Val set, val_loss: {:.4f}, val_correct: {:.4f}'.format(val_loss, val_correct))

    if logging:
        wandb.log({
            logging_prefix + 'loss': val_loss,
            logging_prefix + 'correct': val_correct
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
            [Accuracy(task='multiclass', num_classes=10), 
            AUROC(task='multiclass',num_classes=10), 
            F1Score(task='multiclass', num_classes=10),
            Precision(task='multiclass', num_classes=10), 
            Recall(task='multiclass', num_classes=10)],
            prefix=logging_prefix)
        metrics = metrics.to(device)

    test_correct = 0.

    with torch.no_grad():
        for images, labels in loader:

            images,labels = images.to(device),labels.to(device)
            logits, Y_prob, Y_hat = model(images)
            test_correct +=(Y_hat == labels).sum().item()

            if calculate_metrics:
                metrics.update(logits,labels)

    test_correct /= len(loader.sampler) 

    if calculate_metrics:
        res = metrics.compute()
        res = {k: v.cpu().item() for k, v in res.items()}
        res.update({logging_prefix + 'correct': test_correct})
        return res
            
    else:
        return {logging_prefix + '_error': test_correct}  


def print_summary(split: str, results: Dict) -> None:
    """Prints each result from a dictionary."""
    print('{} results:'.format(split))
    for metric, value in results.items():
        print('{}: {:.4f}'.format(metric, value))



def train(run_id, train_loader, val_loader, test_loader, args):
    print('\nTraining fold {}!'.format(run_id))
    if args.logging:
        run = wandb.init(reinit=True, project=args.project_name, name='_'.join([str(run_id).zfill(2), args.exp_code]))
    
    print("Training on {} samples".format(len(train_loader.sampler)))
    print("Validating on {} samples".format(len(val_loader.sampler)))
    print("Testing on {} samples".format(len(test_loader.sampler)))

    model = CIFAR_Classifier(model=args.model, weights=args.weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.reg)
    loss_fn = nn.CrossEntropyLoss().to(device)


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


def save_pkl(filename: str, object: Any) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def main(args):
    # init results directory
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    # set up CV
    if args.cv == 'mccv':
        cv = MonteCarloCV(args.val_size, args.test_size, args.n_repeats, args.shuffle, args.seed)
    elif args.cv == 'kfoldcv':
        cv = KfoldCV(n_folds=args.n_folds, shuffle=args.shuffle, seed=args.seed)
    else:
        raise ValueError('Unrecognized cv type for cv={}.\
            Should be one of [mccv, kfoldcv].'.format(args.cv))


    # set up dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    if args.cifar_tiny:
        N = 1000 
        trainset.data = trainset.data[:N]
        trainset.targets = trainset.targets[:N]    
        testset.data = testset.data[:N]
        testset.targets = testset.targets[:N]                            

    dataset = ConcatDataset([trainset, testset])
    
    # collect results across folds 
    all_val_res = []
    all_test_res = []

    for run_id, (train_ids, val_ids, test_ids) in enumerate(cv.split(np.arange(len(dataset)))):
        
        # create subset sampler
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        # create loader 
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

        val_metrics, test_metrics = train(run_id, train_loader, val_loader, test_loader, args)

        all_val_res.append(pd.DataFrame(val_metrics, index=[run_id]))
        all_test_res.append(pd.DataFrame(test_metrics, index=[run_id]))

        # write test results to dir
        fn = os.path.join(args.result_dir, 'split_{}_results.pkl'.format(run_id))
        save_pkl(fn, test_metrics)

    # combine results
    final_val_df = pd.concat(all_val_res, axis=0)
    final_test_df = pd.concat(all_test_res, axis=0)

    # save results
    final_val_df.to_csv(os.path.join(args.result_dir, 'summary_val.csv'))
    final_test_df.to_csv(os.path.join(args.result_dir, 'summary_test.csv'))




# init settings
parser = argparse.ArgumentParser(description='Configuration for Cross-validation of Mitosis-Classifier')

# directory settings
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--exp_code', type=str, default='experiment_0')

# cross-validation settings
parser.add_argument('--cv', choices=['mccv', 'kfoldcv'], default='mccv')
parser.add_argument('--val_size', type=float, default=0.2)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--n_repeats', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_folds', type=int, default=10)

# model & optimizer & dataloader settings
parser.add_argument('--cifar_tiny', action='store_true')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--weights', type=str, default='DEFAULT')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

# misc settings
parser.add_argument('--project_name', type=str, default='CrossValidation')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--stop_epoch', type=int, default=50)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--calculate_metrics', action='store_true')
parser.add_argument('--logging', action='store_true')

args = parser.parse_args()

if not os.path.isdir(args.result_dir):
    os.mkdir(args.result_dir)

# create new result dir for each experiment
args.result_dir = os.path.join(args.result_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.result_dir):
    os.mkdir(args.result_dir)


# save settings
settings = vars(args)
save_pkl(os.path.join(args.result_dir, 'settings.pkl'), settings)

# print settings
print('#' * 20 + ' Settings ' + '#' * 20)
pprint.pprint(settings)

if __name__ == '__main__':
    main(args)
    print('End of script!')





