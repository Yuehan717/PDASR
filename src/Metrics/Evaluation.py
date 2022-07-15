import Metrics.LPIPS as models
import os
import numpy as np
from PIL import Image


def calc_LPIPS(sr, hr, scale):
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
    sr = sr / 127.5 - 1
    hr = hr / 127.5 - 1
    dist = model.forward(sr, hr).detach().squeeze().cpu().numpy()
    return dist