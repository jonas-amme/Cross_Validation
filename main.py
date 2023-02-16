import argparse
import pandas as pd
import os 
import torchvision.transforms as T
import pprint

from src.splitting import MonteCarloCV, KfoldCV
from src.utils import train, save_pkl
from src.dataset import Mitosis_Base_Dataset



# Use this command here to remove Human Breast Cancer from MIDOG2
MIDOG2_COMMAND = 'SELECT coordinateX as x, coordinateY as y, Annotations.agreedClass as label, Slides.filename, Annotations.slide\
            FROM Annotations_coordinates \
            INNER JOIN Annotations \
            ON Annotations_coordinates.annoId = Annotations.uid \
            INNER JOIN Slides \
            ON Annotations.slide = Slides.uid \
            WHERE Annotations.deleted=0 and Slides.uid > 150'



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

    # whether to remove hbc from midog2
    if args.use_midog2_command:
        sqlite_command = MIDOG2_COMMAND
    else: 
        sqlite_command = None

    # set up dataset
    dataset = Mitosis_Base_Dataset(args.sqlite_file, args.image_dir, sqlite_command=sqlite_command)
    indices = dataset.get_ids()


    # collect results across folds 
    all_val_res = []
    all_test_res = []

    # set dataset args
    train_kwargs = {
        'sqlite_file': args.sqlite_file,
        'image_dir': args.image_dir,
        'pseudo_epoch_length': args.pseudo_epoch_length,
        'patch_size': args.patch_size,
        'mit_prob': args.mit_prob,
        'arb_prob': args.arb_prob,
        'patch_size': args.patch_size,
        'level': args.level,
        'transforms': train_transforms
    }
    val_kwargs = {
        'sqlite_file': args.sqlite_file,
        'image_dir': args.image_dir,
        'patch_size': args.patch_size,
        'level': args.level,
        'transforms': val_transforms,
        'n_random_samples': args.n_random_samples
    }

    for run_id, (train_ids, val_ids, test_ids) in enumerate(cv.split(indices)):

        # create datasets
        train_ds = dataset.return_split(indices=train_ids, training=True, **train_kwargs)
        val_ds = dataset.return_split(indices=val_ids, training=False, **val_kwargs)
        test_ds = dataset.return_split(indices=test_ids, training=False, **val_kwargs)
        datasets = (train_ds, val_ds, test_ds)
        
        # start training on the folds 
        val_metrics, test_metrics = train(run_id, datasets, args)

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
parser.add_argument('--sqlite_file', type=str, default='./annotations/MIDOG.sqlite', help='Path to sqlite database.')
parser.add_argument('--image_dir', type=str, default='/home/ammeling/data/images_training', help='Directory with images.')
parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results.')
parser.add_argument('--exp_code', type=str, default='experiment_0', help='Name of folder under result_dir.')

# cross-validation settings
parser.add_argument('--cv', choices=['mccv', 'kfoldcv'], default='mccv', help='Which CV technique to perform.')
parser.add_argument('--val_size', type=float, default=0.2, help='Size of validation set for MCCV.')
parser.add_argument('--test_size', type=float, default=0.2, help='Size of test set for MCCV')
parser.add_argument('--n_repeats', type=int, default=10, help='Number of repeats for MCCV')
parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle data before splitting')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for KFOLD CV.')

# model & optimizer & dataloader settings
parser.add_argument('--model', type=str, default='resnet18', help='Which model architecture')
parser.add_argument('--weights', type=str, default='DEFAULT', help='Which type of weights.')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Which optimizer.')
parser.add_argument('--lr', type=float, default=0.001 ,help='Learning rate')
parser.add_argument('--reg', type=float, default=1e-5, help='Regularization parameter')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

# dataset settings
parser.add_argument('--use_midog2_command', action='store_true', help='If using MIDOG2 data, setting to true removes hBC cases.')
parser.add_argument('--pseudo_epoch_length', type=int, default=1024, help='Number of patches per epoch')
parser.add_argument('--mit_prob', type=float, default=0.5, help='Percentage of patches with mitotic figures')
parser.add_argument('--arb_prob', type=float, default=0.25, help='Percentage of random patches')
parser.add_argument('--patch_size', type=int, default=128, help='Patch size')
parser.add_argument('--level', type=int, default=0, help='Level to sample patches from image.')
parser.add_argument('--n_random_samples', type=int, default=0, help='Number of additional random samples for validation')

# misc settings
parser.add_argument('--project_name', type=str, default='CrossValidation', help='Experiment name for W&B')
parser.add_argument('--early_stopping', action='store_true', help='Whether to use early stopping.')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs.')
parser.add_argument('--patience', type=int, default=20, help='Number of epochs without improvement before training ends.')
parser.add_argument('--stop_epoch', type=int, default=30, help='Mininum number of epochs before training can end.')
parser.add_argument('--verbose', action='store_true', help='Whether to show output for saving the best model.')
parser.add_argument('--calculate_metrics', action='store_true', help='Whether to calculate metrics during training.')
parser.add_argument('--logging', action='store_true', help='Whether to use W&B for logging the training.')

args = parser.parse_args()

if not os.path.isdir(args.result_dir):
    os.mkdir(args.result_dir)

# create new result dir for each experiment
args.result_dir = os.path.join(args.result_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.result_dir):
    os.mkdir(args.result_dir)


# set train transforms
train_transforms = T.Compose([
    T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 1))], p=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(degrees=360)], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# set val transforms
val_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.model == 'vit_b_16':
    # set train transforms
    train_transforms = T.Compose([T.Resize(size=(224, 224)), train_transforms])
    val_transforms = T.Compose([T.Resize(size=(224, 224)), val_transforms])

# save settings
settings = vars(args)
save_pkl(os.path.join(args.result_dir, 'settings.pkl'), settings)

# print settings
print('#' * 20 + ' Training Settings ' + '#' * 20)
pprint.pprint(settings)

if __name__ == '__main__':
    main(args)
    print('End of script!')

























