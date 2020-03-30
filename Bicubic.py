import cv2
import numpy as np
import math

def getBicPixelChannel(img, x, y):
    if (x < img.shape[1]) and (y < img.shape[0]):
        return img[y, x] & 0xFF

    return 0

def bicubic(image,scale_factor):

    rows,cols = image.shape
    scaled_height = int(math.ceil(float(rows * scale_factor[0])))
    scaled_weight = int(math.ceil(float(cols * scale_factor[1])))

    scaled_image = np.zeros((scaled_weight,scaled_height),np.uint8)

    x_ratio = float(cols / scaled_weight)
    y_ratio = float(rows / scaled_height)

    C=np.zeros(5)

    for i in range(scaled_height):
        for j in range(scaled_weight):

            x_int = int(j * x_ratio)
            y_int = int(i * y_ratio)

            dx = x_ratio * j - x_int
            dy = y_ratio * i - y_int


            for jj in range(0,4):
                o_y = y_int - 1 + jj
                a0 = getBicPixelChannel(image, x_int, o_y)
                d0 = getBicPixelChannel(image, x_int - 1, o_y) - a0
                d2 = getBicPixelChannel(image, x_int + 1, o_y) - a0
                d3 = getBicPixelChannel(image, x_int + 2, o_y) - a0

                a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
                a2 = 1. / 2 * d0 + 1. / 2 * d2
                a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
                C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx

            d0 = C[0] - C[1]
            d2 = C[2] - C[1]
            d3 = C[3] - C[1]
            a0 = C[1]
            a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
            a2 = 1. / 2 * d0 + 1. / 2 * d2
            a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
            scaled_image[j, i] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy

    return cv2.medianBlur(scaled_image,3)


def show_image(img, img_bicubic):
    # cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original', img)
    # cv2.namedWindow('nearest_function', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('bicubic_function', img_bicubic)

    cv2.waitKey(0)

if __name__ == '__main__':

    img = cv2.imread("cameraman.tif", 0)  # read image as a greyscale

    x = float(input("enter scaling factor for x: "))
    y = float(input("enter scaling factor for y: "))

    rate = (x, y)

    print('original =', img.shape)

    a = bicubic(img, rate)
    img_bicubic_self = a.astype('uint8')
    print("bicubic =", img_bicubic_self.shape)
    img_bicubic_self = cv2.rotate(img_bicubic_self, cv2.ROTATE_90_CLOCKWISE)
    img_bicubic_self = cv2.flip(img_bicubic_self, 1)
    show_image(img, img_bicubic_self)