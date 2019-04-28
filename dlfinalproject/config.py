import os
import os.path as osp

import numpy as np
import torch

basedir = os.path.abspath(os.path.dirname(__file__))
CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class Config:
    def __init__(self):
        self.data_dir = osp.realpath(osp.join(CURRENT_PATH, '..', 'data'))
        self.model_dir = osp.realpath(osp.join(CURRENT_PATH, '..', 'models'))
        self.img_means = np.array([127.91911921, 120.47569851, 107.77482796])
        self.img_stds = np.array([40.22395673, 39.02798012, 45.09549037])
        self.img_size = (96, 96)

        if os.getenv('FORCE_CPU') == '1':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")


config = Config()
