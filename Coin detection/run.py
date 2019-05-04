import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from generate_patches import GetCircles, NMS, GetCrops
from extract_features import GetHog
from scipy.ndimage import rotate


# './Coins/C[0].jpg'
def RunMdls(features, mdl):
    mu, sig, clf = mdl['mu'], mdl['sig'], mdl['clf']
    F = (features - mu[None, :]) / (sig[None, :])
    S = clf.predict(F)
    return S


# with open('./Models/coin.mdl', 'rb') as f:
with open('./Models1/coin.mdl', 'rb') as f:
    mdl = pickle.load(f)

in_file = 'two.jpg'

I = cv2.imread(in_file)
# I = cv2.resize(I, (0,0), fx=0.25, fy=0.25)

circle_list = GetCircles(I)
circle_list = NMS(circle_list)
max_circles = 1
circle_list = circle_list[:min(max_circles, circle_list.shape[0]), :]

patches = GetCrops(I, circle_list)

features = []
for roll in range(0, 360, 60):
    Ir = rotate(I, roll, reshape=False)
    F = GetHog(I)
    features += [F]
features = np.array(features)
scores = RunMdls(features, mdl)

lbls = ['NoCoin', '1 Rupees', '2 Rupee', '5 Rupees', '10 Rupees', ]
plt.imshow(I)
for i in range(circle_list.shape[0]):
    if scores[i]:
        x, y = circle_list[i, :2]
        circle = plt.Circle((x, y), circle_list[i, 2], color='r', fill=False)
        plt.gca().add_artist(circle)
        plt.text(x, y, '%s' % lbls[scores[i]])
plt.waitforbuttonpress()

dummy = 0