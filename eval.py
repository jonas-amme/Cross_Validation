import argparse
import pandas as pd
import numpy as np
import os 
import torchvision.transforms as T
import pprint
import torch
import torch.nn as nn

from src.splitting import MonteCarloCV, KfoldCV
from src.utils import save_pkl, get_split_loader, initiate_model, summary, print_summary
from src.dataset import Mitosis_Validation_Dataset


# Use this command here to remove Human Breast Cancer from MIDOG2 as this was part of training 
MIDOG2_COMMAND = 'SELECT coordinateX as x, coordinateY as y, Annotations.agreedClass as label, Slides.filename, Annotations.slide\
            FROM Annotations_coordinates \
            INNER JOIN Annotations \
            ON Annotations_coordinates.annoId = Annotations.uid \
            INNER JOIN Slides \
            ON Annotations.slide = Slides.uid \
            WHERE Annotations.deleted=0 and Slides.uid > 150'


# init settings
parser = argparse.ArgumentParser(description='CV Eval Script for Mitosis-Classifier')

# directory settings
parser.add_argument('--sqlite_file', type=str, default=None)
parser.add_argument('--image_dir', type=str, default=None)
parser.add_argument('--result_dir', type=str, default='./results', help='Directory containing results from CV')
parser.add_argument('--models_exp_code', type=str, default='experiment_0', help='Directory under results_dir containing trained models')
parser.add_argument('--save_exp_code', type=str, default='./eval_results', help='Directory for to save eval results')

# misc settings
parser.add_argument('--model', type=str, default='resnet18', help='Type of model.')
parser.add_argument('--folds', type=int, default=5, help='Number of folds under models_exp_code')
parser.add_argument('--use_command', action='store_true', help='Use command to remove hBC from MIDOG2 data')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--level', type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', str(args.save_exp_code))
os.makedirs(args.save_dir, exist_ok=True)

args.models_dir = os.path.join(args.result_dir, args.models_exp_code)
assert os.path.isdir(args.models_dir)

# save settings
settings = vars(args)
save_pkl(os.path.join(args.save_dir, 'eval_experiment_{}.pkl'.format(args.save_exp_code)), settings)

# collect ckpts
folds = np.arange(args.folds)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]

# set val transforms
val_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.model == 'vit_b_16':
    # set train transforms
    val_transforms = T.Compose([T.Resize(size=(224, 224)), val_transforms])

# collect datasets
dataset_settings = {
    'sqlite_file': args.sqlite_file,
    'image_dir': args.image_dir,
    'patch_size': args.patch_size,
    'sqlite_command': MIDOG2_COMMAND if args.use_command else None,  #TODO: make option for other commands available
    'transforms': val_transforms
    }

# print settings
print('#' * 20 + ' Eval Settings ' + '#' * 20)
pprint.pprint(settings)


if __name__ == '__main__':

    # load dataset
    print('Initialize dataset ...', end=' ')
    dataset = Mitosis_Validation_Dataset(**dataset_settings)
    print('Done!')
    dataset.summarize()

    print('\nInitialize dataloader ...', end=' ')
    # set up dataloader
    dataloader = get_split_loader(dataset, args, collante_fn=dataset.collate_fn, training=False)
    print('Done!')

    all_res = []
    for idx, ckpt_path in enumerate(ckpt_paths):
        print('\nEvaluating Model {}'.format(idx))

        # load model 
        model = initiate_model(model=args.model, ckpt_path=ckpt_path)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(device)


        # collect results
        res = summary(model, dataloader, calculate_metrics=True, logging_prefix='')
        print_summary(split='Fold {}'.format(idx), results=res)
        print()

        all_res.append(pd.DataFrame(res, index=[idx]))

    # combine results
    all_res = pd.concat(all_res, axis=0)

    # save results
    all_res.to_csv(os.path.join(args.save_dir, 'summary.csv'))
    print('End of script!')


    

    

















