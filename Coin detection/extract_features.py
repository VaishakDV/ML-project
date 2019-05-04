import numpy as np
from skimage.feature import hog
from scipy.ndimage import rotate
import cv2


def GetHog(patch):
    patch = cv2.resize(patch,(64,64))

    fd, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd


if __name__ == '__main__':
    import os


    for t in ['A', 'B', 'C', 'D']:
        skip = 30
        root = './TrainingSet/%s/'%t
        features = []
        for f in os.listdir(root):
            #if f.endswith(".png"):
            if f.endswith(".jpg"):
                in_file = os.path.join(root, f)
                print(in_file)

                I = cv2.imread(in_file)
                for roll in range(0,360,skip):
                    Ir = rotate(I, roll, reshape=False)
                    F = GetHog(I)
                    features += [F]

        features = np.array(features, dtype=np.float)
        with open('./Features1/%s.npy'%t, 'wb') as f:
            np.save(f, features)
