import numpy as np
import matplotlib.pyplot as plt
from utils.XYZ_to_SRGB import XYZ_TO_SRGB
from debayering import menon
import bm3d
from sklearn.linear_model import LinearRegression 
import os
import sys


files = os.listdir(sys.argv[1])
for i in range(0, len(files) - 1, 2):
    gt = np.load('output_generated_data/' + files[i], allow_pickle=True)
    sample = np.load('output_generated_data/' + files[i + 1], allow_pickle=True)
    gt_xyz  = gt.item().get('xyz')
    SRGB = XYZ_TO_SRGB()
    sRGB_img = SRGB.XYZ_to_sRGB(gt_xyz)
    sample_img   = sample.item().get('image')
    sample_bayer = sample.item().get('bayer')
    sample_img /= np.max(sample_img)
    # debayering
    rgb = menon.bayer2rgb(sample_img, pattern=sample_bayer)
    trgb = rgb + abs(np.min(rgb))
    trgb /= np.max(rgb)
    # denoising
    bm3d_img = bm3d.bm3d(trgb, sigma_psd=[0.1], stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    bm3d_img /= np.max(bm3d_img)
    # color space transform
    linear_regres = LinearRegression(fit_intercept=False)
    linear_regres = linear_regres.fit(bm3d_img.reshape(-1, 3), gt_xyz.reshape(-1, 3))
    test = linear_regres.predict(bm3d_img.reshape(-1, 3)).reshape(512, 512, 3)
    # save pictures
    plt.imsave(sys.argv[2] + 'predicts/' + str(i) + '.png', SRGB.XYZ_to_sRGB(test))
    plt.imsave(sys.argv[2] + 'targets/' + str(i) + '.png', sRGB_img)
