import numpy as np
import scipy.linalg
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage import io
import matplotlib.pyplot as plt

def active_contour(image,color, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc='periodic', max_px_move=1.0,
                   max_iterations=1000, convergence=0.25):

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations should be >0.")
    convergence_order = 10

    img = img_as_float(image)

    RGB = img.ndim == 3

    if w_edge != 0:
        if RGB:
            edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                    sobel(img[:, :, 2])]
        else:
            edge = [sobel(img)]

        for i in range(3 if RGB else 1):
            edge[i][0, :] = edge[i][1, :]
            edge[i][-1, :] = edge[i][-2, :]
            edge[i][:, 0] = edge[i][:, 1]
            edge[i][:, -1] = edge[i][:, -2]
    else:
        edge = [0]

    if RGB:
        img = w_line*np.sum(img, axis=2) \
            + w_edge*sum(edge)
    else:
        img = w_line*img + w_edge*edge[0]


    intp = RectBivariateSpline(np.arange(image.shape[1]), #img
                               np.arange(image.shape[0]),
                               img.T, kx=2, ky=2, s=0)


    x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
    xsave = np.empty((convergence_order, len(x)))
    ysave = np.empty((convergence_order, len(x)))


    n = len(x)

    a = np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1) - 2*np.eye(n)
    b = np.roll(np.eye(n), -2, axis=0) + \
        np.roll(np.eye(n), -2, axis=1) - \
        4*np.roll(np.eye(n), -1, axis=0) - \
        4*np.roll(np.eye(n), -1, axis=1) + \
        6*np.eye(n)
    A = -alpha*a #+ beta*b
    print(A.shape)
    print((gamma*np.eye(n)).shape)

    inv = np.linalg.inv(A + gamma*np.eye(n))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(color)


    for i in range(max_iterations):

        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)

        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)


        dx = max_px_move*np.tanh(xn-x)
        dy = max_px_move*np.tanh(yn-y)

        x += dx
        y += dy

        j = i % (convergence_order+1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave-x[None, :]) +  np.abs(ysave-y[None, :]), 1))
            if dist < convergence:
                break


        snake = np.array([x, y]).T
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_title('iteration : %s'%(i))
        ax.set_xticks([]), ax.set_yticks([])
        plt.show(block=False), plt.pause(0.001)
        del ax.lines[0]
    plt.close()
    return np.array([x, y]).T































