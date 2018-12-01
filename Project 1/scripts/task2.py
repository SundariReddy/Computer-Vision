import cv2
import math
import numpy as np


def flip(img):
    image = np.ones([img.shape[0], img.shape[1]])
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            image[x][y] = img[6-x][6-y]
    return image


def resize_colored_octave(octave):
    resized_octave = np.zeros((math.ceil(octave.shape[0]/2) , math.ceil(octave.shape[1]/2) , math.ceil(octave.shape[2])))
    for c in range (3):
        for x in range (0, octave.shape[0], 2):
            for y in range (0, octave.shape[1], 2):
                resized_octave[math.ceil(x/2)][math.ceil(y/2)][c] = octave[x][y][c]
    return resized_octave


def resize_grayscale_octave(octave):
    resized_octave = np.zeros((math.ceil(octave.shape[0]/2) , math.ceil(octave.shape[1]/2)))
    for x in range (0, octave.shape[0], 2):
        for y in range (0, octave.shape[1], 2):
            resized_octave[math.ceil(x/2)][math.ceil(y/2)] = octave[x][y]
    return resized_octave


def gaussian_generator(sigma):
    g = np.zeros((7,7))
    for x in range (-3, 4):
        for y in range (3, -4, -1):
            temp = -((x ** 2) + (y ** 2))
            temp2 = temp / (2 * (sigma ** 2))
            exp = math.exp(temp2)
            fin = exp / (2 * math.pi * (sigma ** 2))
            g[x][y] = fin
    return g


def convolve(f, g):
    conv_img = np.zeros((f.shape[0], f.shape[1]))
    for x in range (3, f.shape[0] - 3):
        for y in range (3, f.shape[1] - 3):
            conv_img[x][y] = mat_mul(f[x-3:x+4,y-3:y+4], g)
    return conv_img


def mat_mul(image1, image2):
    sum = 0
    for x in range (image2.shape[0]):
        for y in range (image2.shape[1]):
            sum = sum + (image1[x][y] * image2[x][y])
    return sum


def is_distinguishable(mat1, mat2, mat3):
    lst = []
    for x in range (mat2.shape[0]):
        for y in range (mat2.shape[1]):
            lst.append(mat1[x][y])
            lst.append(mat2[x][y])
            lst.append(mat3[x][y])

    if (min(lst) == mat2[1][1] or max(lst) == mat2[1][1]):
        return True


def key_point_detection(octave, lst):
    octcopy1 = octave
    octcopy2 = octave

    set1 = [lst[0], lst[1], lst[2]]
    set2 = [lst[1], lst[2], lst[3]]

    for x in range (1, lst[1].shape[0]-1):
        for y in range (1, lst[1].shape[1]-1):
            if (is_distinguishable(lst[0][x-1:x+2,y-1:y+2], lst[1][x-1:x+2,y-1:y+2], lst[2][x-1:x+2,y-1:y+2])):
                octcopy1[x][y] = 255

    for x in range (1, lst[2].shape[0]-1):
        for y in range (1, lst[2].shape[1]-1):
            if (is_distinguishable(lst[1][x-1:x+2,y-1:y+2], lst[2][x-1:x+2,y-1:y+2], lst[3][x-1:x+2,y-1:y+2])):
                octcopy2[x][y] = 255

    return octcopy1, octcopy2


def diff_of_gauss(lst):
    dog = []
    for i in range (len(lst)-1):
        dog.append(lst[i+1] - lst[i])

    return dog


def task2():

    oct1c = cv2.imread('../resources/task2.jpg')
    oct1g = cv2.imread('../resources/task2.jpg', 0)

    oct2g = resize_grayscale_octave(oct1g)
    oct3g = resize_grayscale_octave(oct2g)
    oct4g = resize_grayscale_octave(oct3g)

    octave_grayscale = []
    octave_grayscale.append(oct1g)
    octave_grayscale.append(oct2g)
    octave_grayscale.append(oct3g)
    octave_grayscale.append(oct4g)

    oct2c = resize_colored_octave(oct1c)
    oct3c = resize_colored_octave(oct2c)
    oct4c = resize_colored_octave(oct3c)

    octave_colored = []
    octave_colored.append(oct1c)
    octave_colored.append(oct2c)
    octave_colored.append(oct3c)
    octave_colored.append(oct4c)


    sigma = [[1 / math.sqrt(2)], [1], [math.sqrt(2)], [2], [2 * math.sqrt(2)],\
            [math.sqrt(2)], [2], [2 * math.sqrt(2)], [4], [4 * math.sqrt(2)],\
            [2 * math.sqrt(2)], [4], [4 * math.sqrt(2)], [8], [8 * math.sqrt(2)],\
            [4 * math.sqrt(2)], [8], [8 * math.sqrt(2)], [16], [16 * math.sqrt(2)]]

    sigma = np.array(sigma).reshape(4,5)

    for i in range (len(octave_grayscale)):
        lst = []

        for j in range (5):
            g = np.zeros((7,7))
            conv_img = np.zeros((octave_grayscale[i].shape[0], octave_grayscale[i].shape[1]))

            g = gaussian_generator(sigma[i][j])

            gflip = flip(g)

            conv_img = convolve(octave_grayscale[i], gflip)
            lst.append(conv_img)
            cv2.imwrite('../output_images/octave'+str(i+1)+'sigma'+str(j+1)+'.png', conv_img)

        dog = diff_of_gauss(lst)

        for k in range (len(dog)):
            cv2.imwrite('../output_images/octave'+str(i+1)+'DoG'+str(k+1)+str(k+2)+'.png', dog[k])

        kpd1, kpd2 = key_point_detection(octave_colored[i], dog)

        cv2.imwrite('../output_images/octave'+str(i+1)+'KeyPointDetection1.png', kpd1)
        cv2.imwrite('../output_images/octave'+str(i+1)+'KeyPointDetection2.png', kpd2)


if __name__ == '__main__':
    task2()
