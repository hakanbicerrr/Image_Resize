import cv2
import numpy as np
import math


def bilinear(image,scale_factor):

    rows,cols = image.shape
    scaled_height = math.ceil(rows * scale_factor[0])
    scaled_weight = math.ceil(cols * scale_factor[1])

    scaled_image = np.zeros((scaled_height, scaled_weight), np.uint8)

    #x_ratio = float((cols)) / scaled_weight;
    #y_ratio = float((rows)) / scaled_height;
    x_ratio = float((cols - 1)) / scaled_weight;
    y_ratio = float((rows - 1)) / scaled_height;
    #print(x_ratio,y_ratio)
    for i in range(0,scaled_height):
        for j in range(0,scaled_weight):

            #----------------------------------------------
            #x_ratio = i/scale_factor[0]
            #y_ratio = j/scale_factor[1]

            #x1 = math.floor(x_ratio)
            #x2 = math.ceil(x_ratio)
            #y1 = math.floor(y_ratio)
            #y2 = math.ceil(y_ratio)
            #print(i,j,x1,x2,y1,y2)
            #----------------------------------------------

            x = int(x_ratio * j)
            y = int(y_ratio * i)
            x_diff = (x_ratio * j) - x
            y_diff = (y_ratio * i) - y

            #x_diff = (i % scale_factor[0])/scale_factor[0]
            #y_diff = (j % scale_factor[1])/scale_factor[1]

            c1 = image[x][y] * (1-y_diff) + image[x][y+1] * y_diff
            c2 = image[x+1][y] * (1-y_diff) + image[x+1][y+1] * y_diff

            scaled_image[i][j] = int(c1 * (1-x_diff) + c2 * (x_diff))

            #c2 = (((scale_factor[1]-j)/scale_factor[1])*image[i+1][math.floor(j/scale_factor[1])]) + ((j/scale_factor[1])*image[i+1][math.floor(j/scale_factor[1]+1)])
    return scaled_image


def show_image(img, img_bilinear):
    # cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original', img)
    # cv2.namedWindow('nearest_function', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('bilinear_function', img_bilinear)

    cv2.waitKey(0)

if __name__ == '__main__':

    img = cv2.imread("cameraman.tif", 0)  # read image as a greyscale

    x = float(input("enter scaling factor for x: "))
    y = float(input("enter scaling factor for y: "))

    rate = (x, y)

    print('original =', img.shape)

    a = bilinear(img, rate)
    img_bilinear_self = a.astype('uint8')
    print("bilinear =", img_bilinear_self.shape)
    img_bilinear_self = cv2.rotate(img_bilinear_self,cv2.ROTATE_90_CLOCKWISE)
    img_bilinear_self = cv2.flip(img_bilinear_self,1)
    show_image(img, img_bilinear_self)

    #resized = cv2.resize(img, (768,768), interpolation=cv2.INTER_LINEAR)
    #print('Resized Dimensions : ', resized.shape)
    #cv2.imshow("Resized image", resized)
    #cv2.waitKey()