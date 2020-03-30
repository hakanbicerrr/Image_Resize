import cv2
import numpy as np
import math

def getBicPixel(img, x, y):
    if (x < img.shape[1]) and (y < img.shape[0]):
        return img[y, x] & 0xFF

    return 0

def my_resize(image,scale_factor,method):
    if method == "nearest":
        # Extract size
        (rows, cols) = image.shape
        # Compute new size
        scaled_height = math.ceil(rows * scale_factor[0])
        scaled_weight = math.ceil(cols * scale_factor[1])
        # Compute ratio
        row_ratio = rows / scaled_height
        col_ratio = cols / scaled_weight
        row_position = np.floor(np.arange(scaled_height) * row_ratio).astype(int)
        column_position = np.floor(np.arange(scaled_weight) * col_ratio).astype(int)
        # Initialize scaled image
        scaled_image = np.zeros((scaled_height, scaled_weight), np.uint8)

        for i in range(scaled_height):
            for j in range(scaled_weight):
                scaled_image[i, j] = image[row_position[i], column_position[j]]
        return scaled_image

    elif method == "bilinear":
        # Extract size
        rows, cols = image.shape
        # Compute new size
        scaled_height = math.ceil(rows * scale_factor[1])
        scaled_weight = math.ceil(cols * scale_factor[0])
        # Initialize scaled image
        scaled_image = np.zeros((scaled_height, scaled_weight), np.uint8)
        # Compute ratio
        x_ratio = float((cols - 1)) / scaled_weight;
        y_ratio = float((rows - 1)) / scaled_height;

        for i in range(0, scaled_height):
            for j in range(0, scaled_weight):
                x = int(x_ratio * j)
                y = int(y_ratio * i)
                x_diff = (x_ratio * j) - x
                y_diff = (y_ratio * i) - y

                c1 = image[x][y] * (1 - y_diff) + image[x][y + 1] * y_diff
                c2 = image[x + 1][y] * (1 - y_diff) + image[x + 1][y + 1] * y_diff
                scaled_image[i][j] = int(c1 * (1 - x_diff) + c2 * (x_diff))
        return scaled_image

    elif method == "bicubic":

        rows, cols = image.shape
        scaled_height = int(math.ceil(float(rows * scale_factor[0])))
        scaled_weight = int(math.ceil(float(cols * scale_factor[1])))

        scaled_image = np.zeros((scaled_weight, scaled_height), np.uint8)

        x_ratio = float(cols / scaled_weight)
        y_ratio = float(rows / scaled_height)

        C = np.zeros(5)

        for i in range(0,scaled_height):
            for j in range(0,scaled_weight):

                x_int = int(j * x_ratio)
                y_int = int(i * y_ratio)

                dx = x_ratio * j - x_int
                dy = y_ratio * i - y_int

                for jj in range(0, 4):
                    o_y = y_int - 1 + jj
                    a0 = getBicPixel(image, x_int, o_y)
                    d0 = getBicPixel(image, x_int - 1, o_y) - a0
                    d2 = getBicPixel(image, x_int + 1, o_y) - a0
                    d3 = getBicPixel(image, x_int + 2, o_y) - a0

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

        return cv2.medianBlur(scaled_image, 3)

if __name__ == '__main__':
    # Read image as a greyscale
    img = cv2.imread("cameraman.tif", 0)

    x = float(input("enter scaling factor for x: "))
    y = float(input("enter scaling factor for y: "))
    method = input("enter interpolation method: ")
    rate = (x, y)

    print('original =',img.shape)

    a = my_resize(img,rate,method)
    if method == "nearest":

        img_nearest_self = a.astype('uint8')
        print("nearest =", img_nearest_self.shape)
        cv2.imshow("original",img)
        cv2.imshow("nearest",img_nearest_self)
        cv2.waitKey(0)

    elif method == "bilinear":

        img_bilinear_self = a.astype("uint8")
        print("bilinear =", img_bilinear_self.shape)
        img_bilinear_self = cv2.rotate(img_bilinear_self, cv2.ROTATE_90_CLOCKWISE)
        img_bilinear_self = cv2.flip(img_bilinear_self, 1)
        cv2.imshow("original",img)
        cv2.imshow("bilinear",img_bilinear_self)
        cv2.waitKey(0)

    elif method == "bicubic":

        img_bicubic_self = a.astype("uint8")
        print("bilinear =", img_bicubic_self.shape)
        img_bicubic_self = cv2.rotate(img_bicubic_self, cv2.ROTATE_90_CLOCKWISE)
        img_bicubic_self = cv2.flip(img_bicubic_self, 1)
        cv2.imshow("original", img)
        cv2.imshow("bicubic", img_bicubic_self)
        cv2.waitKey(0)
