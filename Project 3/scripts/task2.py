#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import task1


def convolve(f, g):
    conv_img = np.zeros((f.shape[0], f.shape[1]))
    xoff = math.ceil(kernel.shape[0]/2)
    yoff = math.ceil(kernel.shape[1]/2)

    for x in range(xoff, f.shape[0] - xoff):
        for y in range(yoff, f.shape[1] - yoff):
            conv_img[x][y] = mat_mul(f[x-xoff:x+xoff+1, y-yoff:y+yoff+1], g)
    return conv_img


def mat_mul(image1, image2):
    sum_ = 0
    for x in range(image2.shape[0]):
        for y in range(image2.shape[1]):
            sum_ = sum_ + (image1[x][y] * image2[x][y])

    return sum_


def add_cord(cord, img2):
    flag = True
    bias = 10
    for i in cord:
        if i[0][0] - bias <= img2[0][0] <= i[1][0] + bias and i[0][0] - bias <= img2[1][0] <= i[1][0] + bias \
                and i[0][1] - bias <= img2[0][1] <= i[1][1] + bias:
            flag = False
            break
    if flag:
        cord.append(img2)

    return cord


def draw(img1, img2):
    cord = []
    y_max = img1.shape[1]

    for x in range(img1.shape[0]):
        y = 0
        while y < y_max:
            if img1[x][y] != 0:
                y += 1

            else:
                ctr = 0
                ctr_ = 0

                if img1[x + 10][y + ctr] == 0:
                    while img1[x + 10][y + ctr] == 0:
                        ctr += 1

                else:
                    while img1[x][y + ctr] == 0:
                        ctr += 1

                while img1[x + ctr_][y] == 0:
                    ctr_ += 1

                if ctr > 20 and ctr_ > 20:
                    cord = add_cord(cord, [(x, y), (x + ctr_, y + ctr)])

                y += ctr
    # print(len(cord), cord)

    for i in cord:
        cv2.rectangle(img2, i[0][::-1], i[1][::-1], (255, 0, 0), 1)
    # cv2.imwrite('../output_images/task2.8_rectangle.jpg', img2)


if __name__ == '__main__':
    kernel_size = (3, 3)
    threshold = 275
    img1 = cv2.imread(r'../resources/point2.jpg', 0)
    img2 = cv2.imread(r'../resources/segment.jpg', 0)
    kernel = np.ones(kernel_size) * -1
    kernel[0][0] = 0
    kernel[0][2] = 0
    kernel[2][0] = 0
    kernel[2][2] = 0
    kernel[1][1] = 4

    kernel2 = np.ones(kernel_size) * 255

    new_img = convolve(img1, kernel)
    cv2.imwrite('../output_images/task2.1_point_detection.jpg', new_img)

    for x in range(new_img.shape[0]):
        for y in range(new_img.shape[1]):
            if new_img[x][y] > threshold:
                new_img[x][y] = 255
            else:
                new_img[x][y] = 0

    cv2.imwrite('../output_images/task2.2_point_detection_thresholded.jpg', new_img)

    # find the coordinates of the point
    cord = (0, 0)
    for x in range(new_img.shape[0]):
        for y in range(new_img.shape[1] - 5):
            if new_img[x][y] > 0:
                cord = (x , y)
                print('Coordinates of porosity : ', cord)

    # Plot to find optimal threshold
    plot_ = np.zeros(255)
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if img2[x][y] > 0 and img2[x][y] < 256:
                intensity = img2[x][y]
                plot_[intensity] += 1

    x, y = list(range(1, 256)), plot_
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels')
    plt.title('Thresholding')
    plt.plot(x, y)
    plt.savefig(r'../output_images/task2.3_optimal_threshold.jpg')

    threshold = 204
    new_img = np.zeros(img2.shape)

    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if img2[x][y] > threshold:
                new_img[x][y] = 255

    cv2.imwrite('../output_images/task2.4_segment.jpg', new_img)
    new_img = task1.morphological_operation(new_img, kernel2, 'dilation')
    cv2.imwrite('../output_images/task2.5_segment_morphed.jpg', new_img)

    new_img2 = np.zeros(new_img.shape)
    new_img3 = np.copy(new_img)

    for x in range(new_img.shape[0]):
        ctr = 0
        for y in range(new_img.shape[1]):
            if new_img[x][y] == 0:
                ctr += 1

        if ctr == new_img.shape[1]:
            new_img2[x, :] = 75

    for y in range(new_img.shape[1]):
        ctr = 0
        for x in range(new_img.shape[0]):
            if new_img[x][y] == 0:
                ctr += 1

        if ctr == new_img.shape[0]:
            new_img2[:, y] = 75

    cv2.imwrite('../output_images/task2.6_box.jpg', new_img2)

    for x in range(new_img.shape[0]):
        ctr = 0
        while ctr < new_img.shape[1]:
            if new_img2[x][ctr] != 75:
                ctr_ = 0

                while new_img2[x][ctr + ctr_] == 0:
                    ctr_ += 1

                if sum(new_img[x, ctr:ctr+ctr_]) == 0:
                    new_img3[x, ctr:ctr+ctr_] = 75

                ctr += ctr_

            else:
                ctr += 1

    for x in range(new_img.shape[0]):
        for y in range(new_img.shape[1]):
            new_img[x][y] = max(new_img2[x][y], new_img3[x][y])

    cv2.imwrite('../output_images/task2.7_segment_box.jpg', new_img)

    draw(new_img, img2)
