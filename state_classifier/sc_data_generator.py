import collections
import os

import imageio
import numpy as np
from keras.utils import Sequence


class StateClassifierDataGenerator(Sequence):

    def __init__(self, task_name, batch_size=32, shuffle=True, is_validation=False):
        self.task_name = task_name
        self.trial_folder = os.path.join(f"data/labels/{task_name}")
        trial_names = sorted(os.listdir(self.trial_folder))
        if is_validation:
            trial_names = trial_names[int(0.9*len(trial_names)):]
        else:
            trial_names = trial_names[:int(0.9 * len(trial_names))]
        self.data_pairs_per_trial = collections.defaultdict(list)
        self.n_classes = None
        self.image_size = None
        self.num_samples = 0
        self.data_pairs = []
        for trial_name in trial_names:
            trial_path = os.path.join(self.trial_folder, trial_name)
            image_names = sorted([image_name for image_name in os.listdir(trial_path) if image_name.endswith(".png")])
            label_names = sorted([image_name for image_name in os.listdir(trial_path) if image_name.endswith(".npy")])
            for image_name, label_name in zip(image_names, label_names):
                self.data_pairs_per_trial[trial_name].append((image_name, label_name))
                self.data_pairs.append((trial_name, image_name, label_name))
                self.num_samples += 1
                if self.n_classes is None:
                    self.n_classes = np.load(os.path.join(trial_path, label_name)).shape[0]
                if self.image_size is None:
                    image = imageio.imread(os.path.join(trial_path, image_name))
                    self.image_size = image.shape

        self.indexes = np.arange(len(self.data_pairs))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return self.num_samples

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_pairs):
        x = np.zeros((self.batch_size, *self.image_size), dtype=float)
        y = np.zeros((self.batch_size, self.n_classes), dtype=float)

        # Generate data
        for index, (trial_name, image_name, label_name) in enumerate(data_pairs):
            x[index] = imageio.imread(os.path.join(self.trial_folder, trial_name, image_name))/255.0
            y[index] = np.load(os.path.join(self.trial_folder, trial_name, label_name))

        # print(y)
        # z = input()
        return x, y

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        data_pairs = [self.data_pairs[i] for i in indexes]
        x, y = self.__data_generation(data_pairs)

        return x, y
