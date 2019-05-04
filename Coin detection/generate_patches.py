import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import hough_circle
from skimage.feature import canny

#Construct Image pyramid and extract hough circles for radius 50 to 100 in each scale
def GetCircles(I):
    Y = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    r_min = 50
    radius_list = np.arange(r_min,int(round(2.1*r_min)))
    num_circles_per_scale = 50
    scale = 1.0
    pyramid = []
    circle_list = []
    while min(Y.shape[:2]) > r_min:
        M = canny(Y, sigma=2, low_threshold=10, high_threshold=20)
        # plt.imshow(M, cmap='gray')
        # plt.waitforbuttonpress()
        H = hough_circle(M, radius_list[radius_list < min(Y.shape[:2]) / 2])

        r_idx, y, x = np.unravel_index(H.reshape(-1).argsort()[::-1][:min(num_circles_per_scale, H.size)], H.shape)
        r = radius_list[r_idx]
        pyramid += [Y]
        circle_list += [np.array([scale * x, scale * y, scale * r, H[r_idx, y, x] / r]).T ]

        Y = cv2.resize(Y, (0,0),fx=0.5,fy=0.5)
        scale *= 2.0
    circle_list = np.concatenate(circle_list)

    return circle_list


# Non-maximal supression. Retain non-overlapping circles
# Uses same algorithm as object detection
# overlap = intersection / union
def NMS(circles, ov_thr = 0.5):
    bbs = np.array([circles[:, 0] - circles[:, 2],
                    circles[:, 1] - circles[:, 2],
                    circles[:, 0] + circles[:, 2],
                    circles[:, 1] + circles[:, 2],
                    circles[:, 3],
                    4*(circles[:, 2]**2)]).T
    i, n = 0, bbs.shape[0]
    while i < n:
        idx = i + bbs[i:n,4].argmax()
        bbs[i,:], bbs[idx,:] = bbs[idx,:], bbs[i,:]
        j = i + 1
        while j < n:
            xx1 = max(bbs[i,0], bbs[j,0])
            yy1 = max(bbs[i,1], bbs[j,1])
            xx2 = min(bbs[i,2], bbs[j,2])
            yy2 = min(bbs[i,3], bbs[j,3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            intersect = float(w * h)
            xx1 = min(bbs[i, 0], bbs[j, 0])
            yy1 = min(bbs[i, 1], bbs[j, 1])
            xx2 = max(bbs[i, 2], bbs[j, 2])
            yy2 = max(bbs[i, 3], bbs[j, 3])
            union = (xx2 - xx1)*(yy2 - yy1)
            overlap = intersect / union
            if overlap > ov_thr:
                n -= 1
                bbs[j, :], bbs[n, :] = bbs[n, :], bbs[j, :]
            else:
                j += 1
        i += 1
    bbs = bbs[:n, :]
    r = (bbs[:,2] - bbs[:,0])/2
    circles = np.array([bbs[:,0]+r, bbs[:,1]+r, r, bbs[:,4]]).T
    return circles


def GetCrops(I, circle_list):
    x, y, r = circle_list[:, 0], circle_list[:, 1], circle_list[:, 2]
    scale = 2.0
    xs, xe, ys, ye = x - scale*r, x + scale*r, y - scale*r, y + scale*r
    f = lambda v: np.round(v).astype(np.int)
    xs, xe, ys, ye = f(xs), f(xe), f(ys), f(ye)
    r, c = I.shape[:2]
    f = lambda v: np.maximum(-v, 0)
    pl, pr, pt, pb = f(xs), f(c - 1 - xe), f(ys), f(r - 1 - ye)
    f = lambda v,k: np.minimum(np.maximum(v, 0), k-1)
    xs, xe, ys, ye = f(xs,c), f(xe,c), f(ys,r), f(ye,r)

    J = []
    for i in range(circle_list.shape[0]):
        P = I[ys[i]:ye[i], xs[i]:xe[i], :]
        P = np.pad(P, ((pt[i],pb[i]),(pl[i],pr[i]),(0,0)),mode='constant')
        P = cv2.resize(P, (256,256),interpolation=cv2.INTER_CUBIC)
        J += [P]
    return J


if __name__ == '__main__':
    import os
    try:
        os.makedirs('./Coin_Patches/')
    except:
        pass
    # for t in ['A', 'B', 'C', 'D', 'E', 'F']:
    #     for i in range(10):
    #         in_file = './Coins/%s[%d].jpg'%(t,i)
    #         print(in_file)
    #         I = cv2.imread(in_file)
    #         I = cv2.resize(I, (0,0), fx=0.25, fy=0.25)
    #
    #         circle_list = GetCircles(I)
    #         circle_list = NMS(circle_list)
    #         max_circles = 16
    #         circle_list = circle_list[:min(max_circles, circle_list.shape[0]),:]
    #         # plt.imshow(I[:,:,::-1])
    #         # for i in range(circle_list.shape[0]):
    #         #     circle = plt.Circle(circle_list[i,:2], circle_list[i,2], color='r', fill=False)
    #         #     plt.gca().add_artist(circle)
    #         # plt.waitforbuttonpress()
    #
    #         patches = GetCrops(I, circle_list)
    #         for j, J in enumerate(patches):
    #             out_path = './Coin_Patches/%s[%d]_%d.png'%(t,i,j)
    #             cv2.imwrite(out_path, J)

    for i in range(1,19):
        in_file = './FakeCoins/Fake (%d).jpg'%(i)
        print(in_file)
        I = cv2.imread(in_file)
        I = cv2.resize(I, (0,0), fx=0.25, fy=0.25)

        circle_list = GetCircles(I)
        circle_list = NMS(circle_list)
        max_circles = 16
        circle_list = circle_list[:min(max_circles, circle_list.shape[0]),:]
        # plt.imshow(I[:,:,::-1])
        # for i in range(circle_list.shape[0]):
        #     circle = plt.Circle(circle_list[i,:2], circle_list[i,2], color='r', fill=False)
        #     plt.gca().add_artist(circle)
        # plt.waitforbuttonpress()

        patches = GetCrops(I, circle_list)
        for j, J in enumerate(patches):
            out_path = './Coin_Patches/Fake_(%d)_%d.png'%(i,j)
            cv2.imwrite(out_path, J)