from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import openslide 
import pandas as pd
import torch
import sqlite3
import os

from numpy.random import randint, choice
from torch.utils.data import Dataset
from tqdm import tqdm


Coords = Tuple[int, int]


class SlideObject():
    def __init__(
        self,
        slide_path: str = None,
        annotations: pd.DataFrame = None,
        patch_size: Coords = (128, 128),
        level: int = 0
        ) -> None:

        if slide_path is None:
            raise ValueError('Must provide filename')

        self.slide_path = slide_path
        self.annotations = annotations
        self.patch_size = patch_size
        self.level = level

        self.slide = openslide.open_slide(self.slide_path)

    @property
    def slide_size(self) -> Coords:
        return self.slide.level_dimensions[self.level]


    def __str__(self) -> str:
        return f"SlideObject for {self.slide_path}"
    

    # TODO: add downfactor to sample from different levels (not necessary for now)
    def load_image(self, coords: Coords) -> np.ndarray:
        """Returns a patch of the slide at the given coordinates."""
        patch = self.slide.read_region(coords, self.level, self.patch_size).convert('RGB')
        return np.array(patch)  


    def get_label(self, x: int, y: int):
        mfs = self.annotations.query('label == 1')
        idxs = (mfs.x > x) & (mfs.x < (x + self.patch_size[0])) \
            & (mfs.y > y) & (mfs.y < (y + self.patch_size[0]))
        if (np.count_nonzero(idxs) > 0):
            return 1
        else:
            return 0




class Mitosis_Classification_Dataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = None,
        image_dir: str = None,
        pseudo_epoch_length: int = 512,
        mit_prob: float = 0.5,
        arb_prob: float = 0.25,
        patch_size: Coords = (128,128), 
        level: int = 0,
        transforms: Union[List[Callable], Callable] = None) -> None:

        self.data = data
        self.image_dir = image_dir  
        self.pseudo_epoch_length = pseudo_epoch_length
        self.mit_prob = mit_prob
        self.arb_prob = arb_prob
        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms

        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()


    def load_database(self) -> pd.DataFrame:
        """Opens the sqlite database and converts it to pandas dataframe. 
        If indices are provided, data is splitted accordingly.

        Returns:
            pd.DataFrame: Dataframe [x, y, label, filename, slide].
        """
        # command to select all data from the table
        command = 'SELECT coordinateX as x, coordinateY as y, Annotations.agreedClass as label, Slides.filename, Annotations.slide\
                    FROM Annotations_coordinates \
                    INNER JOIN Annotations \
                    ON Annotations_coordinates.annoId = Annotations.uid \
                    INNER JOIN Slides \
                    ON Annotations.slide = Slides.uid \
                    WHERE Annotations.deleted=0'

        if self.sqlite_file:
            DB = sqlite3.connect(self.sqlite_file)

            # execute the command and create dataframe
            df = pd.read_sql_query(command, DB)

            # load data for current split
            if self.indices:
                df = df.query('slide in @indices')
        else:
            raise ValueError(
                'Cannot load database due to missing\
                 file for sqlite_file={}'.format(self.sqlite_file)
                 )

        return df 

    def load_slide_objects(self) -> Dict[str, SlideObject]:
        """Initializes slide objects from dataframe.

        Returns:
            Dict[str, SlideObject]: Dictionary with all slide objects. 
        """
        fns = self.data.filename.unique().tolist()
        slide_objects = {}
        for fn in tqdm(fns, desc='Initializing slide objects'):
            slide_path = os.path.join(self.image_dir, fn)
            annotations = self.data.query('filename == @fn')[['x', 'y', 'label']].reset_index()
            slide_objects[fn] = SlideObject(
                slide_path=slide_path,
                annotations=annotations,
                patch_size=self.patch_size,
                level=self.level
            )
        return slide_objects


    def sample_patches(
        self, 
        mit_prob: float=None, 
        arb_prob: float=None) -> Dict[str, Dict[str, Coords]]:
        """Samples patches from all slides with equal probability.
        Proportions of patches with mitotic figures or imposters can be adjusted with mit_prob or arb_prob.

        Args:
            mit_prob (float, optional): Percentage of patches with mitotic figures. Defaults to None.
            arb_prob (float, optional): Percentage of random patches. Defaults to None.

        Returns:
            Dict[str, Dict[str, Coords]]: Dictionary with {idx: coords, label}
        """
        # sample slides
        slides = choice(list(self.slide_objects.keys()), size=self.pseudo_epoch_length, replace=True)
        # sample patches
        patches = {}
        for idx, slide in enumerate(slides):
            patches[idx] = self.sample_func(slide, mit_prob, arb_prob)
        return patches



    def sample_func(
        self,
        fn: str, 
        mit_prob: float = None, 
        arb_prob: float = None
        ) -> Dict[str, Tuple[Coords, int]]:
        """Samples patches randomly from a slide.

        Labels are transformed into a binary problem [0=no mitosis, 1=mitosis].

        Args:
            fn (str): Filename (e.g. "042.tiff")
            mit_prob (float, optional): Proportion of patches with mitotic figure. Defaults to None.
            arb_prob (float, optional): Proportion of random patches. Defaults to None.

        Returns:
            Dict[str, Tuple[Coords, int]]: Dictionary with filenames, patch coordinates and the label.
        """

         # set sampling probabilities
        mit_prob = self.mit_prob if mit_prob is None else mit_prob
        arb_prob = self.arb_prob if arb_prob is None else arb_prob

        # get slide object
        sl = self.slide_objects[fn]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # create sampling probabilites
        sample_prob = np.array([self.arb_prob, self.mit_prob, 1-self.mit_prob-self.arb_prob])

        # sample case from probabilites (0 = random, 1 = mitosis, 2 = imposter)
        case = choice(3, p=sample_prob)

        if case == 0:
            # random patch 
            x = randint(patch_width / 2, slide_width-patch_width / 2)
            y = randint(patch_height / 2, slide_height-patch_height / 2)

        elif case == 1:     
            # filter mitosis cases
            mask = sl.annotations.label == 1

            if np.count_nonzero(mask) == 0:
                # no mitosis available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)
            else:       
                # get annotations
                MF = sl.annotations[['x', 'y']][mask]

                # sample mitosis
                idx = randint(MF.shape[0])
                x, y = MF.iloc[idx]

        elif case == 2:
            # sample imposter
            mask = sl.annotations.label == 2

            if np.count_nonzero(mask) == 0:
                # no imposter available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)

            else:
                # get annotations
                NMF = sl.annotations[['x', 'y']][mask]
                # sample imposter
                idx = randint(NMF.shape[0])
                x, y = NMF.iloc[idx]

        # set offsets
        offset_scale = 0.5
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        label = sl.get_label(x, y)

        return {'file': fn, 'coords': (x, y), 'label': label}



    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        # get sample
        sample = self.samples[idx]
        # extract sample info
        file, coords, label = sample['file'], sample['coords'], sample['label']
        # get slide object
        slide = self.slide_objects[file]
        # get img
        img = slide.load_image(coords)
        if self.transform is not None:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img / 255).permute(2, 0, 1).type(torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label


    def get_ids(self):
        return self.data['slide'].unique()


    def get_labels(self):
        return self.data['label'].unique()


    @staticmethod
    def collate_fn(batch):
        """Collate function for the data loader."""
        images = list()
        targets = list()
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        images = torch.stack(images, dim=0)
        return images, targets



class Mitosis_Validation_Dataset(Mitosis_Classification_Dataset):
    def __init__(
        self, 
        data: pd.DataFrame = None,
        image_dir: str = None,
        patch_size: Coords = (128,128), 
        level: int = 0,
        n_random_samples: int = 0,
        transforms: Union[List[Callable], Callable] = None) -> None:

        self.data = data
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms
        self.n_random_samples = n_random_samples

        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()


    def sample_patches(self) -> Dict[str, Dict[str, Coords]]:
        """Returns all annotations from the dataset.

        Optional: Adding n patches sampled randomly with n_random_samples. 

        Returns:
            Dict[str, Dict[str, Coords]]:  Dictionary with {idx: coords, label}.
        """
        size, _ = self.patch_size
        patches = {}
        for idx, row in self.data.iterrows():
            fn, x, y = row.filename, row.x - size / 2 , row.y - size / 2
            label = 1 if row.label == 1 else 0
            patches[idx] = {'file': fn, 'coords': (x, y), 'label': label}
            
        if self.n_random_samples > 0:
            slides = choice(list(self.slide_objects.keys()), size=self.pseudo_epoch_length, replace=True)
            for idx, slide in enumerate(slides):
                patches[idx] = self.sample_func(slide, mit_prob=0, arb_prob=1)

        return patches 


        













        




    









        

        
















    

    
        



