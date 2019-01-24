import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data,io
from skimage.filters import gaussian
from acm import active_contour
import cv2
from scipy.interpolate import interp1d

import argparse

# parser = argparse.ArgumentParser()
#
# parser.add_argument("--image", help="name of image", type=str)
#
# args = parser.parse_args()

#
# imageName = args.image



imageName = "cat2.jpg"
drawing = False # true if mouse is pressed

ix,iy = -1,-1
x_list = []
y_list = []

def draw_contour(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        x_list.append(ix)
        y_list.append(iy)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
            x_list.append(x)
            y_list.append(y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def acm(imageName,x_list,y_list):
    img = io.imread(imageName,0)
    color_img = img
    gray_img = rgb2gray(img)
    init = np.array([x_list, y_list]).T
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '-r', lw=3)
    gaussian_img = gaussian(gray_img, 3)
    snake = active_contour(gaussian_img,color_img,
                           init, alpha=0.015, beta=10, gamma=0.001)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

def intp(x_list,y_list):
    new_x = []
    new_y = []
    lenth = len(x_list)
    print('contour lenghth : ',lenth)
    if lenth < 400:
        for i in range(1,lenth-1):
            new_x.append(x_list[i - 1])
            new_y.append(y_list[i - 1])
            inp_x = (x_list[i-1] + x_list[i]) //2
            inp_y = (y_list[i-1] + y_list[i]) //2

            new_x.append(inp_x)
            new_y.append(inp_y)
    else:
        return x_list,y_list
    return new_x,new_y

img = io.imread(imageName,0)
img[:,:,:] = img[:,:,::-1]
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_contour)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
x,y = intp(x_list,y_list)
acm(imageName,x,y)


