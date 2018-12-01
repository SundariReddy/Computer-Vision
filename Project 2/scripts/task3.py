#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import math
import task3_4


def task3():
    X = [ (5.9, 3.2), (4.6, 2.9), (6.2, 2.8), (4.7, 3.2), (5.5, 4.2), (5.0, 3.0), (4.9, 3.1), (6.7, 3.1), (5.1, 3.8), (6.0, 3.0) ]
    mu = [ (6.2, 3.2), (6.5, 3.0), (6.6, 3.7) ]

    # 1st iteration
    cluster0 = []
    cluster1 = []
    cluster2 = []
    lst = []
    for i in range (len(X)):
        x1,y1 = X[i]
        min = (x1 + y1) ** 2
        minmu = 10
        for j in range (len(mu)):
            x2,y2 = mu[j]
            euclidean = ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
            euclidean = math.sqrt(euclidean)
            if (euclidean < min):
                min = euclidean
                minmu = j

        lst.append(minmu + 1)

        if (minmu == 0):
            cluster0.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "red", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 1):
            cluster1.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "blue", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 2):
            cluster2.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "green", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

    print ('Classification Vector for iteration 1 ' + str(lst))

    plt.scatter(mu[0][0], mu[0][1], edgecolor = "red", facecolor = "red", marker = "o")
    plt.text(mu[0][0], mu[0][1] + 0.05, '%s, %s' % (str(mu[0][0])[:3], str(mu[0][1])[:3]), fontsize=7)
    plt.scatter(mu[1][0], mu[1][1], edgecolor = "blue", facecolor = "blue", marker = "o")
    plt.text(mu[1][0], mu[1][1] + 0.05, '%s, %s' % (str(mu[1][0])[:3], str(mu[1][1])[:3]), fontsize=7)
    plt.scatter(mu[2][0], mu[2][1], edgecolor = "green", facecolor = "green", marker = "o")
    plt.text(mu[2][0], mu[2][1] + 0.05, '%s, %s' % (str(mu[2][0])[:3], str(mu[2][1])[:3]), fontsize=7)
    plt.savefig('../output_images/task3_iter1_a.jpg')
    plt.clf()


    for i in range (len(X)):
        x1,y1 = X[i]
        min = (x1 + y1) ** 2
        minmu = 10
        for j in range (len(mu)):
            x2,y2 = mu[j]
            euclidean = ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
            euclidean = math.sqrt(euclidean)
            if (euclidean < min):
                min = euclidean
                minmu = j

        if (minmu == 0):
            cluster0.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "red", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 1):
            cluster1.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "blue", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 2):
            cluster2.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "green", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)


    # Update MU
    avgx = 0
    avgy = 0

    for x,y in cluster0:
        avgx += x
        avgy += y

    mu[0] = (avgx / len(cluster0), avgy / len(cluster0))

    avgx = 0
    avgy = 0

    for x,y in cluster1:
        avgx += x
        avgy += y

    mu[1] = (avgx / len(cluster1), avgy / len(cluster1))

    avgx = 0
    avgy = 0

    for x,y in cluster2:
        avgx += x
        avgy += y

    mu[2] = (avgx / len(cluster2), avgy / len(cluster2))

    plt.scatter(mu[0][0], mu[0][1], edgecolor = "red", facecolor = "red", marker = "o")
    plt.text(mu[0][0], mu[0][1] + 0.05, '%s, %s' % (str(mu[0][0])[:3], str(mu[0][1])[:3]), fontsize=7)
    plt.scatter(mu[1][0], mu[1][1], edgecolor = "blue", facecolor = "blue", marker = "o")
    plt.text(mu[1][0], mu[1][1] + 0.05, '%s, %s' % (str(mu[1][0])[:3], str(mu[1][1])[:3]), fontsize=7)
    plt.scatter(mu[2][0], mu[2][1], edgecolor = "green", facecolor = "green", marker = "o")
    plt.text(mu[2][0], mu[2][1] + 0.05, '%s, %s' % (str(mu[2][0])[:3], str(mu[2][1])[:3]), fontsize=7)
    plt.savefig('../output_images/task3_iter1_b.jpg')
    plt.clf()


    # 2nd iteration

    cluster0 = []
    cluster1 = []
    cluster2 = []
    lst = []

    for i in range (len(X)):
        x1,y1 = X[i]
        min = (x1 + y1) ** 2
        minmu = 10
        for j in range (len(mu)):
            x2,y2 = mu[j]
            euclidean = ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
            euclidean = math.sqrt(euclidean)
            if (euclidean < min):
                min = euclidean
                minmu = j

        lst.append(minmu + 1)

        if (minmu == 0):
            cluster0.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "red", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 1):
            cluster1.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "blue", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 2):
            cluster2.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "green", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

    print ('Classification Vector for iteration 2 ' + str(lst))

    plt.scatter(mu[0][0], mu[0][1], edgecolor = "red", facecolor = "red", marker = "o")
    plt.text(mu[0][0], mu[0][1] + 0.05, '%s, %s' % (str(mu[0][0])[:3], str(mu[0][1])[:3]), fontsize=7)
    plt.scatter(mu[1][0], mu[1][1], edgecolor = "blue", facecolor = "blue", marker = "o")
    plt.text(mu[1][0], mu[1][1] + 0.05, '%s, %s' % (str(mu[1][0])[:3], str(mu[1][1])[:3]), fontsize=7)
    plt.scatter(mu[2][0], mu[2][1], edgecolor = "green", facecolor = "green", marker = "o")
    plt.text(mu[2][0], mu[2][1] + 0.05, '%s, %s' % (str(mu[2][0])[:3], str(mu[2][1])[:3]), fontsize=7)
    plt.savefig('../output_images/task3_iter2_a.jpg')
    plt.clf()


    for i in range (len(X)):
        x1,y1 = X[i]
        min = (x1 + y1) ** 2
        minmu = 10
        for j in range (len(mu)):
            x2,y2 = mu[j]
            euclidean = ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
            euclidean = math.sqrt(euclidean)
            if (euclidean < min):
                min = euclidean
                minmu = j

        if (minmu == 0):
            cluster0.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "red", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 1):
            cluster1.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "blue", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

        elif (minmu == 2):
            cluster2.append(X[i])
            plt.scatter(x1 , y1, edgecolor = "green", facecolor = "white", marker = "^")
            plt.text(x1, y1 + 0.05, '%s, %s' % (str(x1), str(y1)), fontsize=7)

    # Update MU
    avgx = 0
    avgy = 0

    for x,y in cluster0:
        avgx += x
        avgy += y

    mu[0] = (avgx / len(cluster0), avgy / len(cluster0))

    avgx = 0
    avgy = 0

    for x,y in cluster1:
        avgx += x
        avgy += y

    mu[1] = (avgx / len(cluster1), avgy / len(cluster1))

    avgx = 0
    avgy = 0

    for x,y in cluster2:
        avgx += x
        avgy += y

    mu[2] = (avgx / len(cluster2), avgy / len(cluster2))

    plt.scatter(mu[0][0], mu[0][1], edgecolor = "red", facecolor = "red", marker = "o")
    plt.text(mu[0][0], mu[0][1] + 0.05, '%s, %s' % (str(mu[0][0])[:3], str(mu[0][1])[:3]), fontsize=7)
    plt.scatter(mu[1][0], mu[1][1], edgecolor = "blue", facecolor = "blue", marker = "o")
    plt.text(mu[1][0], mu[1][1] + 0.05, '%s, %s' % (str(mu[1][0])[:3], str(mu[1][1])[:3]), fontsize=7)
    plt.scatter(mu[2][0], mu[2][1], edgecolor = "green", facecolor = "green", marker = "o")
    plt.text(mu[2][0], mu[2][1] + 0.05, '%s, %s' % (str(mu[2][0])[:3], str(mu[2][1])[:3]), fontsize=7)
    plt.savefig('../output_images/task3_iter2_b.jpg')
    plt.clf()


if __name__ == '__main__':
    task3()
    task3_4.task3_4()
