import cv2
import numpy as np
import math

def nn_interpolate(image, scale_factor):
    # Extract size
    (rows, cols) = image.shape
    scaled_height = math.ceil(rows * scale_factor[0])
    scaled_weight = math.ceil(cols * scale_factor[1])

    # Compute ratio
    row_ratio = rows / scaled_height
    col_ratio = cols / scaled_weight

    row_position = np.floor(np.arange(scaled_height) * row_ratio).astype(int)
    column_position = np.floor(np.arange(scaled_weight) * col_ratio).astype(int)
    #print(row_position)
    #print(len(row_position))

    # Initialize scaled image
    scaled_image = np.zeros((scaled_height, scaled_weight), np.uint8)

    for i in range(scaled_height):
        for j in range(scaled_weight):
            scaled_image[i, j] = image[row_position[i], column_position[j]]
    return scaled_image


def show_image(img,img_nearest):
    #cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original', img)
    #cv2.namedWindow('nearest_function', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('nearest_function', img_nearest)

    cv2.waitKey(0)

if __name__ == '__main__':

    img = cv2.imread("cameraman.tif", 0) #read image as a greyscale

    x = float(input("enter scaling factor for x: "))
    y = float(input("enter scaling factor for y: "))

    rate = (x, y)

    print('original =',img.shape)

    a = nn_interpolate(img,rate)
    img_nearest_self = a.astype('uint8')
    print("nearest =",img_nearest_self.shape)

    show_image(img,img_nearest_self)
