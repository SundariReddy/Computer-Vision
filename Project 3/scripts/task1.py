#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np
import math


def check_morphology(img, kernel, operation):
    ctr = 0
    flag = False
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if img[i][j] == kernel[i][j]:
                ctr += 1

    if operation == 'erosion':
        if ctr == kernel.shape[0]*kernel.shape[1]:
            flag = True

    elif operation == 'dilation':
        if ctr > 0:
            flag = True

    else:
        print('Error: incorrect operation')
        exit()

    return flag


def morphological_operation(img, kernel, operation):
    new_img = np.zeros(img.shape)
    xoff = math.ceil(kernel.shape[0]/2)
    yoff = math.ceil(kernel.shape[1]/2)

    for x in range(xoff, img.shape[0]-xoff):
        for y in range(yoff, img.shape[1]-yoff):
            tmp = img[x-xoff:x+xoff+1, y-yoff:y+yoff+1]

            if check_morphology(tmp, kernel, operation):
                new_img[x][y] = 255

    return new_img


def morphology(img, kernel, operation):
    if operation == 'opening':
        morphology_img = morphological_operation(morphological_operation(img, kernel, 'erosion'), kernel, 'dilation')
        return morphology_img

    elif operation == 'closing':
        morphology_img = morphological_operation(morphological_operation(img, kernel, 'dilation'), kernel, 'erosion')
        return morphology_img

    else:
        print('Error: incorrect operation')


if __name__ == '__main__':
    img = cv2.imread(r'../resources/noise.jpg', 0)
    kernel_size = (4, 4)

    kernel = np.ones(kernel_size) * 255

    res_noise1 = morphology(morphology(img, kernel, 'opening'), kernel, 'closing')
    res_noise2 = morphology(morphology(img, kernel, 'closing'), kernel, 'opening')

    res_bound1 = np.subtract(res_noise1, morphological_operation(res_noise1, kernel, 'erosion'))
    res_bound2 = np.subtract(res_noise2, morphological_operation(res_noise2, kernel, 'erosion'))

    cv2.imwrite(r'../output_images/res_noise1.jpg', res_noise1)
    cv2.imwrite(r'../output_images/res_noise2.jpg', res_noise2)
    cv2.imwrite(r'../output_images/res_bound1.jpg', res_bound1)
    cv2.imwrite(r'../output_images/res_bound2.jpg', res_bound2)
