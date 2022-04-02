import numpy as np
import cv2
from Canny import *
import matplotlib.pyplot as plt


def detectCircles(img, threshold, region, radius=None):
    """

    :param img:
    :param threshold:
    :param region:
    :param radius:
    :return:
    # """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img)
    # plt.show()
    # img = cv2.GaussianBlur(img, (5, 5), 1.5)
    # plt.imshow(img)
    # plt.show()
    # img=cv2.Canny(img,20,100)
    img = Canny(img)
    plt.imshow(img)
    plt.show()
    (M, N) = img.shape
    if radius == None:
        R_max = np.max((M, N))
        R_min = 3
    else:
        [R_max, R_min] = radius

    R = R_max - R_min
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X coordinate and Y coordinate resectively.
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
    B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(img[:, :])  # Extracting all edge coordinates
    for val in range(R):
        r = R_min + val
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  # Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]


def displayCircles(A, img):
    """

    :param A:
    :param img:
    :return:
    """
    circleCoordinates = np.argwhere(A)  # Extracting the circle information
    for r, x, y in circleCoordinates:
        cv2.circle(img, (y, x), r, color=(0, 255, 0), thickness=2)

    plt.imshow(img)
    plt.show()
    return img


def hough_circles(source: np.ndarray, min_radius: int = 10, max_radius: int = 50) -> np.ndarray:
    """

    :param source:
    :param min_radius:
    :param max_radius:
    :return:
    """

    src = np.copy(source)
    circles = detectCircles(src, threshold=10, region=15, radius=[max_radius, min_radius])
 
    return displayCircles(circles, src)

# image=cv2.imread('circles_v2.png')
image=cv2.imread('coins.jpg')
image=hough_circles(image,20,50)
plt.imshow(image)
plt.show()