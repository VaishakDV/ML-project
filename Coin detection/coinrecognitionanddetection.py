# import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# capturing coin image using webcam
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "c{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


# Load picture
im_path = "c0.png"
img = cv2.imread(im_path)
cv2.imwrite('aaa_small.png', img)
# exit(0)
# plt.imshow(img[:,:,::-1])
# plt.waitforbuttonpress()


# Resize image to managable resolution
img=cv2.resize(img,(0,0),fx=1/5,fy=1/5,interpolation=cv2.INTER_CUBIC)
# plt.imshow(img[:,:,::-1])
# plt.waitforbuttonpress()

#Convert image to greyscale and compute edges
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(grey, cmap='gray')
# plt.waitforbuttonpress()

dx, dy = ndimage.sobel(grey/256.0,0), ndimage.sobel(grey/256.0,1)
M = (dx**2 + dy**2)**0.5 # Gradient magnitude (Jacobian)
M[M<0.4] = 0

# plt.imshow(dx, cmap='gray')
# plt.waitforbuttonpress()
# plt.imshow(dy, cmap='gray')
# plt.waitforbuttonpress()
#plt.imshow(M, cmap='gray')
#plt.waitforbuttonpress()

#M = canny(grey, sigma=3, low_threshold=10, high_threshold=50)
#plt.imshow(edges, cmap='gray')
#plt.waitforbuttonpress()

# Detect two radii
hough_radii = np.arange(20,50, 1)
hough_res = hough_circle(M, hough_radii)

# Select the most prominent 5 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
print(cx,cy,radii)

plt.imshow(img)
circle_hdl = plt.Circle((cx[0], cy[0]), radii[0], color='b', fill=False, linewidth=2.0)
plt.gca().add_artist(circle_hdl)
plt.waitforbuttonpress()