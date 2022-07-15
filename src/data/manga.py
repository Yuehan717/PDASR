import os
import glob
from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Manga(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Manga, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data,"Manga109")
        # self.apath = os.path.join(self.apath, 'x'+str(self.scale[0]))
        self.dir_hr = os.path.join(self.apath,'HR/Manga109','x'+str(self.scale[0]))
        self.dir_lr = os.path.join(self.apath, 'LR/LRBI/Manga109','x'+str(self.scale[0]))
        self.ext = ('', '.png')

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            filename = filename.replace('HR','LRBI')
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                       filename, self.ext[1]
                    )
                ))

        return names_hr, names_lr