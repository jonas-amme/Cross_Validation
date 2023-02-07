import argparse
import pandas as pd
import os 
import torchvision.transforms as T
import pprint

from src.splitting import MonteCarloCV, KfoldCV
from src.utils import train, save_pkl
from src.dataset import Mitosis_Base_Dataset


# set train transforms
train_transforms = T.Compose([
    T.RandomApply(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), p=0.5),
    T.RandomApply(T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 1)), p=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply(T.RandomRotation(degrees=(90, 180, 270)), p=0.5),
    T.ToTensor(),
    T.Normalize()
])

# set val transforms
val_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize()
])


def main(args):
    # init results directory
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    # set up CV
    if args.cv == 'mccv':
        cv = MonteCarloCV(args.val_size, args.test_size, args.n_repeats, args.seed)
    elif args.cv == 'kfoldcv':
        cv = KfoldCV(n_folds=args.n_folds)
    else:
        raise ValueError('Unrecognized cv type for cv={}.\
            Should be one of [mccv, kfoldcv].'.format(args.cv))


    # set up dataset
    dataset = Mitosis_Base_Dataset(args.database, args.image_dir)
    indices = dataset.get_ids()


    # collect results across folds 
    all_val_res = []
    all_test_res = []

    # set dataset args
    train_kwargs = {
        'sqlite_file': args.sqlite_file,
        'image_dir': args.image_dir,
        'pseudo_epoch_length': args.pseudo_epoch_length,
        'patch_size': args.patchs_size,
        'mit_prob': args.mit_prob,
        'arb_prob': args.arb_prob,
        'patch_size': args.patch_size,
        'level': args.level,
        'transforms': train_transforms
    }
    val_kwargs = {
        'sqlite_file': args.sqlite_file,
        'image_dir': args.image_dir,
        'pseudo_epoch_length': args.pseudo_epoch_length,
        'patch_size': args.patchs_size,
        'level': args.level,
        'transforms': val_transforms,
        'n_random_samples': args.n_random_samples
    }

    for run_id, (train_ids, val_ids, test_ids) in enumerate(cv.split(indices)):

        # create datasets
        train_ds = dataset.return_split(indices=train_ids, **train_kwargs)
        val_ds = dataset.return_split(indices=val_ds, **val_kwargs)
        test_ds = dataset.return_split(indices=test_ds, **val_kwargs)
        datasets = (train_ds, val_ds, test_ds)
        
        # start training on the folds 
        val_metrics, test_metrics = train(run_id, datasets, args)
        all_val_res.append(val_metrics)
        all_test_res.append(test_metrics)
        
        # write test results to dir
        fn = os.path.join(args.result_dir, 'split_{}_results.pkl'.format(run_id))
        save_pkl(fn, test_metrics)

    # combine results
    final_val_df = pd.concat(all_val_res, axis=0)
    final_test_df = pd.concat(all_test_res, axis=0)

    # save results
    final_val_df.to_csv(os.path.join(args.result_dir, 'summary_val.csv'))
    final_test_df.to_csv(os.path.join(args.result_dir, 'summary_test.csv'))



parser = argparse.ArgumentParser(description='Configuration for Cross-validation of Mitosis-Classifier')

# directory settings
parser.add_argument('--sqlite_file', type=str, default='./annotations/MIDOG.sqlite')
parser.add_argument('--image_dir', type=str, default='/home/ammeling/data/images_training')
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--exp_code', type=str, default='experiment_0')

# cross-validation settings
parser.add_argument('--cv', choices=['mccv', 'kfoldcv'], default='mccv')
parser.add_argument('--val_size', type=float, default=0.2)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--n_repeats', type=int, default=10)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_folds', type=int, default=10)

# model & optimizer & dataloader settings
parser.add_argument('--model', type=str, choices=['resnet18, resnet50'], default='resnet18')
parser.add_argument('--weights', type=str, choices=['DEFAULT', 'IMAGNE1K_V1', None], default='DEFAULT')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_worker', type=int, default=4)

# dataset settings
parser.add_argument('--pseudo_epoch_length', type=int, default=1024)
parser.add_argument('--mit_prob', type=float, default=0.5)
parser.add_argument('--arb_prob', type=float, default=0.25)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--level', type=int, default=0)
parser.add_argument('--n_random_samples', type=int, default=0)

# misc settings
parser.add_argument('--project_name', type=str, default='CrossValidation')
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--stop_epoch', type=int, default=50)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# create new result dir for each experiment
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))

# save settings
settings = vars(args)
save_pkl(os.path.join(args.result_dir, 'settings.pkl'), settings)

# print settings
print('#' * 20 + ' Settings ' + '#' * 20)
pprint.pprint(settings)

if __name__ == '__main__':
    main(args)
    print('End of script!')

























