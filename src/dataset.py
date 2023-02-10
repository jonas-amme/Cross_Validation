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


class SlideObject:
    """Class to handle WSI objects.
    
    Args:
        slide_path (str, optional): Path to file. Defaults to None.
        annotations (pd.DataFrame, optional): Annotations [e.g. x, y, class]. Defaults to None.
        patch_size (int, optional): Patch size. Defaults to 128.
        level (int, optional): Level to sample patches. Defaults to 0.

    Raises:
            ValueError: If no filepath is provided.
    """
    def __init__(
        self,
        slide_path: str = None,
        annotations: pd.DataFrame = None,
        patch_size: int = 128,
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
    def slide_dims(self) -> Coords:
        return self.slide.level_dimensions[self.level]

    @property
    def patch_dims(self) -> Coords:
        return (self.patch_size, self.patch_size)


    def __str__(self) -> str:
        return f"SlideObject for {self.slide_path}"
    

    def load_image(self, coords: Coords) -> np.ndarray:
        """Returns a patch of the slide at the given coordinates."""
        size = (self.patch_size, self.patch_size)
        patch = self.slide.read_region(coords, self.level, size).convert('RGB')
        return patch


    def get_label(self, x: int, y: int) -> int:
        """Whether a patch contains a mitotic figure."""
        mfs = self.annotations.query('label == 1')
        idxs = (mfs.x > x) & (mfs.x < (x + self.patch_size)) \
            & (mfs.y > y) & (mfs.y < (y + self.patch_size))
        if (np.count_nonzero(idxs) > 0):
            return 1
        else:
            return 0


class Mitosis_Base_Dataset(Dataset):
    """Base dataset class for mitosis classificaiton.

    Contains functionality to load SlideObjects. Functionality for sampling 
    patches needs to be implemented with `sample_patches`, `sample_func`.

    Args:
        sqlite_file (str): Path to database. 
        image_dir (str): Directory with images. 
        indices (np.array, optinal): Indices to select slides from database. Defaults to None.
        sqlite_command (str, optional): Command to select data from sqlite database. Defaults to None.
    """
    def __init__(
        self,
        sqlite_file: str,
        image_dir: str,
        indices: np.array = None, 
        sqlite_command: str = None) -> None:

        self.sqlite_file = sqlite_file
        self.image_dir = image_dir  
        self.indices = indices
        self.sqlite_command = sqlite_command

        self.data = self.load_database()


    def load_database(self) -> pd.DataFrame:
        """Opens the sqlite database and converts it to pandas dataframe. 
        If indices are provided, data is splitted accordingly.

        Returns:
            pd.DataFrame: Dataframe [x, y, label, filename, slide].
        """
        # open database file 
        DB = sqlite3.connect(self.sqlite_file)
        
        # default command to select all data from the table
        command = 'SELECT coordinateX as x, coordinateY as y, Annotations.agreedClass as label, Slides.filename, Annotations.slide\
                    FROM Annotations_coordinates \
                    INNER JOIN Annotations \
                    ON Annotations_coordinates.annoId = Annotations.uid \
                    INNER JOIN Slides \
                    ON Annotations.slide = Slides.uid \
                    WHERE Annotations.deleted=0'

        # execute the command and create dataframe
        if self.sqlite_command is None:
            df = pd.read_sql_query(command, DB)
        else:
            df = pd.read_sql_query(self.sqlite_command, DB)

        # load data for current split
        if self.indices is not None:
            df = df.query('slide in @self.indices')

        return df.reset_index()


    def return_split(self, indices, training: bool = False, **kwargs):
        """Returns either training or validation set.

        Args:
            indices (np.array): Indices to select for split.
            training (bool, optional): Whether to use train or val dataset. Defaults to False.
        """
        if training:
            return Mitosis_Training_Dataset(indices=indices, **kwargs)
        else:
            return Mitosis_Validation_Dataset(indices=indices, **kwargs)
        

    def load_slide_objects(self):
        """Function to initialize slide objects from dataframe."""
        raise NotImplementedError


    def sample_patches(self):
        """Sample patches for each epoch."""
        raise NotImplementedError


    def sample_func(self):
        """Sample coordinates for a single patch."""
        raise NotImplementedError


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None


    def get_ids(self):
        return self.data['slide'].unique()


    def get_labels(self):
        return self.data['label'].unique()


    def summarize(self):
        print('\nNumber of slides: {}'.format(len(self.data['slide'].unique())))
        print('Number of mitosis: {}'.format(len(self.data.query('label == 1'))))
        print('Number of imposter: {}'.format(len(self.data.query('label == 2'))))


    @staticmethod
    def collate_fn(batch):
        """Collate function for the data loader."""
        images = list()
        targets = list()
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.int64)
        return images, targets



class Mitosis_Training_Dataset(Mitosis_Base_Dataset):
    """Datasat for mitosis classification.

    Randomly samples patches around mitotic figures, imposters or random locations.
    Patch coordinsates are slightly shifted after sampling to add more variability.

    This class is used for training. The validtion dataset should be constructed 
    from `Mitosis_Validation_Dataset`.

    Args:
        sqlite_file (str): Path to database. 
        image_dir (str): Directory with images. 
        indices (np.array, optinal): Indices to select slides from database. Defaults to None.
        pseudo_epoch_length (int, optional): Number of patches for each epoch. Defaults to 512.
                sqlite_command (str, optional): Command to select data from sqlite database. Defaults to None.
        mit_prob (float, optional): Percentage of patches with mitotic figures. Defaults to 0.5.
        arb_prob (float, optional): Percentage of random patches. Defaults to 0.25.
        patch_size (int, optional): Patch size. Defaults to 128.
        level (int, optional): Level to sample. Defaults to 0.
        transforms (Union[List[Callable], Callable], optional): Transformations. Defaults to None.
    """
    def __init__(
        self,
        sqlite_file: str,
        image_dir: str,
        indices: np.array = None,
        sqlite_command: str = None, 
        pseudo_epoch_length: int = 512,
        mit_prob: float = 0.5,
        arb_prob: float = 0.25,
        patch_size: int = 128, 
        level: int = 0,
        transforms: Union[List[Callable], Callable] = None) -> None:

        self.sqlite_file = sqlite_file
        self.image_dir = image_dir
        self.indices = indices  
        self.sqlite_command = sqlite_command
        self.pseudo_epoch_length = pseudo_epoch_length
        self.mit_prob = mit_prob
        self.arb_prob = arb_prob
        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms

        self.data = self.load_database()
        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()


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


    def sample_patches(self) -> Dict[str, Dict[str, Coords]]:
        """Samples patches from all slides with equal probability.
        Proportions of patches with mitotic figures or imposters can be adjusted with mit_prob or arb_prob.

        Returns:
            Dict[str, Dict[str, Coords]]: Dictionary with {idx: coords, label}
        """
        # sample slides
        slides = choice(list(self.slide_objects.keys()), size=self.pseudo_epoch_length, replace=True)
        # sample patches
        patches = {}
        for idx, slide in enumerate(slides):
            patches[idx] = self.sample_func(slide, self.mit_prob, self.arb_prob)
        return patches


    def resample_patches(self):
        """Loads a new set of patches."""
        self.samples = self.sample_patches()
        print('Sampled new patches!')


    def sample_func(
        self,
        fn: str, 
        mit_prob: float, 
        arb_prob: float,
        ) -> Dict[str, Tuple[Coords, int]]:
        """Samples patches randomly from a slide.

        Labels are transformed into a binary problem [0=no mitosis, 1=mitosis].

        Args:
            fn (str): Filename (e.g. "042.tiff")
            mit_prob (float): Proportion of patches with mitotic figure. Defaults to None.
            arb_prob (float): Proportion of random patches. Defaults to None.

        Returns:
            Dict[str, Tuple[Coords, int]]: Dictionary with filenames, patch coordinates and the label.
        """
        # get slide object
        sl = self.slide_objects[fn]

        # get dims
        slide_width, slide_height = sl.slide_dims
        patch_width, patch_height = sl.patch_dims

        # create sampling probabilites
        sample_prob = np.array([arb_prob, mit_prob, 1-mit_prob-arb_prob])

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
        offset_scale = 0.2
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
        sample = self.samples[idx]
        file, coords, label = sample['file'], sample['coords'], sample['label']
        slide = self.slide_objects[file]

        img = slide.load_image(coords)
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(np.array(img) / 255).permute(2, 0, 1).type(torch.float32)

        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label



class Mitosis_Validation_Dataset(Mitosis_Training_Dataset):
    """Dataset for mitosis classification. 

    Used for validation of a classifier. Instead of sampling patches randomly,
    all annotations from the dataset are loaded and centered around the annotation.

    Args:
        sqlite_file (str): Path to database. 
        image_dir (str): Directory with images. 
        indices (np.array, optinal): Indices to select slides from database. Defaults to None.
                sqlite_command (str, optional): Command to select data from sqlite database. Defaults to None.
        patch_size (int, optional): _description_. Defaults to 128.
        level (int, optional): _description_. Defaults to 0.
        n_random_samples (int, optional): _description_. Defaults to 0.
        transforms (Union[List[Callable], Callable], optional): _description_. Defaults to None.
    """
    def __init__(
        self, 
        sqlite_file: str,
        image_dir: str,
        indices: np.array = None,
        sqlite_command: str = None,
        patch_size: int = 128, 
        level: int = 0,
        n_random_samples: int = 0,
        transforms: Union[List[Callable], Callable] = None) -> None:

        self.sqlite_file = sqlite_file
        self.image_dir = image_dir
        self.indices = indices  
        self.sqlite_command = sqlite_command
        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms
        self.n_random_samples = n_random_samples

        self.data = self.load_database()
        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()


    def sample_patches(self) -> Dict[str, Dict[str, Coords]]:
        """Returns all annotations from the dataset.

        Optional: Adding n patches sampled randomly with n_random_samples. 

        Returns:
            Dict[str, Dict[str, Coords]]:  Dictionary with {idx: coords, label}.
        """
        patches = {}
        for idx, row in self.data.iterrows():
            fn, x, y = row.filename, int(row.x - self.patch_size / 2) , int(row.y - self.patch_size / 2)
            label = 1 if row.label == 1 else 0
            patches[idx] = {'file': fn, 'coords': (x, y), 'label': label}
            
        if self.n_random_samples > 0:
            slides = choice(list(self.slide_objects.keys()), size=self.n_random_samples, replace=True)
            for idx, slide in enumerate(slides):
                patches[(idx + len(self.data))] = self.sample_func(slide, mit_prob=0, arb_prob=1)

        return patches 


    def __len__(self):
        return len(self.samples)


        













        




    









        

        
















    

    
        



