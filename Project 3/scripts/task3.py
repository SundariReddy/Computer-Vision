#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np
import math
import task1


def check_morphology(img, kernel, operation, orientation):
    ctr = 0
    flag = False

    if orientation == 'diagonal':
        for i in range(kernel.shape[0]):
            for _ in range(kernel.shape[1]):
                if img[i][i] > 0:
                    ctr += 1

    elif orientation == 'vertical':
        for i in range(kernel.shape[0]):
            for _ in range(kernel.shape[1]):
                if img[i][math.floor(kernel.shape[1]/2)] > 0:
                    ctr += 1

    else:
        print('Error: incorrect orientation')
        exit()

    if operation == 'erosion':
        if ctr == kernel.shape[0]*kernel.shape[1]:
            flag = True

    elif operation == 'dilation':
        if ctr > 0:
            flag = True

    else:
        print('Error: incorrect Operation')
        exit()

    return flag


def morphological_operation(img, kernel, operation, orientation):
    new_img = np.zeros(img.shape)
    xoff = math.ceil(kernel.shape[0]/2)
    yoff = math.ceil(kernel.shape[1]/2)

    for x in range(xoff, img.shape[0]-xoff):
        for y in range(yoff, img.shape[1]-yoff):
            tmp = img[x-xoff:x+xoff+1, y-yoff:y+yoff+1]

            if check_morphology(tmp, kernel, operation, orientation):
                new_img[x][y] = 255

    return new_img


def perform_canny_edge_detection(img):
    return cv2.Canny(img, 50, 150, 3)


def find_rho_theta(img, list_, x, y):
    dict = {}
    for rho in range(x):
        for theta in range(y):
            if img[rho][theta] in list_:
                dict[rho] = theta
    return dict


def calculate_hough_space(img, rho, theta):
    new_img = np.zeros((rho, theta))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] == 255:
                for theta_ in range(theta):
                    sin_ = math.sin(math.radians(theta_))
                    cos_ = math.cos(math.radians(theta_))
                    rho_ = int((x * cos_) + (y * sin_))

                    if rho_ < 0:
                        rho_ *= -1
                    new_img[rho_][theta_] += 1

    return new_img


def hough_transform(img, threshold):
    rho = int((2 * (((img.shape[0] ** 2) + (img.shape[1] ** 2)) ** 0.5)) + 1)
    theta = 180

    hough_space = calculate_hough_space(img, rho, theta)
    list_ = hough_space.ravel().flatten()
    list_.sort()
    max_ = list_[-threshold:]

    rho_theta = find_rho_theta(hough_space, max_, rho, theta)
    hough_img = np.zeros(img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for rho_, theta_ in rho_theta.items():
                value = int(((-1)/math.tan(math.radians(theta_)))*x + rho_/math.sin(math.radians(theta_)))
                if y == value:
                    hough_img[x][y] = 255
                    break

    return hough_space, hough_img


if __name__ == '__main__':
    imgc = cv2.imread('../resources/hough.jpg')
    img = cv2.imread('../resources/hough.jpg', 0)
    kernel_size = (3, 3)
    kernel = np.ones(kernel_size) * 255
    vertical_kernel = np.ones(kernel_size) * -1
    diagonal_kernel = np.ones(kernel_size) * -1

    for i in range(vertical_kernel.shape[0]):
        vertical_kernel[i][math.floor(vertical_kernel.shape[1]/2)] = 2

    for i in range(diagonal_kernel.shape[0]):
        diagonal_kernel[i][i] = 2

    edge_img = perform_canny_edge_detection(img)
    cv2.imwrite('../output_images/task3.1_edge.jpg', edge_img)

    vertical_erosion = task1.morphological_operation(
        task1.morphological_operation(
            task1.morphological_operation(
                task1.morphological_operation(
                    morphological_operation(edge_img, vertical_kernel, 'erosion', 'vertical'),
                    kernel, 'dilation'),
                kernel, 'erosion'),
            kernel, 'erosion'),
        kernel, 'dilation')

    cv2.imwrite('../output_images/task3.2_vertical_erosion.jpg', vertical_erosion)

    diagonal_erosion = task1.morphological_operation(
        task1.morphological_operation(
            task1.morphological_operation(
                task1.morphological_operation(
                    task1.morphological_operation(
                        task1.morphological_operation(
                            morphological_operation(edge_img, diagonal_kernel, 'erosion', 'diagonal'),
                            kernel, 'dilation'),
                        kernel, 'erosion'),
                    kernel, 'erosion'),
                kernel, 'dilation'),
            kernel, 'erosion'),
        kernel, 'dilation')

    cv2.imwrite('../output_images/task3.3_diagonal_erosion.jpg', diagonal_erosion)

    hough_space, hough_vertical = hough_transform(vertical_erosion, 12)
    cv2.imwrite(r'../output_images/task3.4_vertical_hough_space.jpg', hough_space)
    cv2.imwrite(r'../output_images/task3.6_vertical_hough_image.jpg', hough_vertical)

    hough_space, hough_diagonal = hough_transform(diagonal_erosion, 25)
    cv2.imwrite(r'../output_images/task3.5_diagonal_hough_space.jpg', hough_space)
    cv2.imwrite(r'../output_images/task3.7_diagonal_hough_image.jpg', hough_diagonal)

    hough_img_vertical = np.copy(imgc)
    hough_img_diagonal = np.copy(imgc)

    for x in range(imgc.shape[0]):
        for y in range(imgc.shape[1]):
            if hough_vertical[x][y] > 0:
                hough_img_vertical[x][y][0] = 0
                hough_img_vertical[x][y][1] = 255
                hough_img_vertical[x][y][2] = 0

            if hough_diagonal[x][y] > 0:
                hough_img_diagonal[x][y][0] = 0
                hough_img_diagonal[x][y][1] = 255
                hough_img_diagonal[x][y][2] = 0

    cv2.imwrite(r'../output_images/task3.8_hough_vertical.jpg', hough_img_vertical)
    cv2.imwrite(r'../output_images/task3.9_hough_diagonal.jpg', hough_img_diagonal)
