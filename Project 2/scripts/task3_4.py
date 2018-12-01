#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np
import random
import math


def gen_mu(k):
    mu = []
    for i in range (k):
        mu.append([np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)])
    return mu


def update_mu(mu, clusters, point_list):
    newmu = []

    for i, val in enumerate (mu):
        r = 0
        g = 0
        b = 0
        ctr = 0

        for j,k in zip (clusters, point_list):
            if (i == j):
                r += k[0]
                g += k[1]
                b += k[2]
                ctr += 1

        if (ctr == 0):
            newmu.append(k)

        else:
            r = int(r/ctr)
            g = int(g/ctr)
            b = int(b/ctr)

        newmu.append([r,g,b])

    return newmu


def clustering(img, mu):
    clusters = []
    point_list = []
    for i in img:
        for j in i:
            r,g,b = j[0], j[1], j[2]
            point_list.append([r,g,b])

    for i in range(len(point_list)):
        min = None
        minmu = 30
        for j in range(len(mu)):
            euclidean = ((mu[j][0] - point_list[i][0]) ** 2) + ((mu[j][1] - point_list[i][1]) ** 2) + ((mu[j][2] - point_list[i][2]) ** 2)
            euclidean = math.sqrt(euclidean)
            if (min == None or euclidean < min):
                min = euclidean
                minmu = j

        clusters.append(minmu)

    return point_list, clusters

def update_img(mu, clusters, point_list, img):
    newimg = np.zeros([512,512,3])
    ctr = 0
    for j,k in zip (clusters, point_list):

            newimg[int(ctr/512)][int(ctr%512)] = mu[j]
            ctr += 1

    return newimg


def task3_4():
    img = cv2.imread(r'../resources/baboon.jpg')
    iterations = 10
    for k in [3, 5, 10, 20]:
        mu = gen_mu(k)

        for i in range (iterations):
            point_list, clusters = clustering(img, mu)
            mu = update_mu(mu, clusters, point_list)

        newimg = update_img(mu, clusters, point_list, img)

        cv2.imwrite('../output_images/task3_baboon_' + str(k) + '.jpg', newimg)


if __name__ == '__main__':
    task3_4()
