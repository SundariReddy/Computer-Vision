#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np


def get_epilines(img, lines, pts1, pts2):
   row, col = img.shape
   img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
   R = 200
   G = 255
   B = 17
   for r, pt1, pt2 in zip(lines, pts1, pts2):
       color = tuple([R, G, B])
       xInit, yInit = map(int, [0, -r[2] / r[1]])
       xNew, yNew = map(int, [col, -(r[2] + r[0] * col) / r[1]])
       img = cv2.line(img, (xInit, yInit), (xNew, yNew), color, 1)
       img = cv2.circle(img, tuple(pt1), 5, color, -1)
       R += 30
       G -= 20
       B += 5
   return img


def task2():
    # reading images
    imgL_color = cv2.imread(r'../resources/tsucuba_left.png')
    imgR_color = cv2.imread(r'../resources/tsucuba_right.png')

    imgL_gray = cv2.imread(r'../resources/tsucuba_left.png', 0)
    imgR_gray = cv2.imread(r'../resources/tsucuba_right.png', 0)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(imgL_color, None)
    kp2, des2 = sift.detectAndCompute(imgR_color, None)

    # draw key points on the images
    key_points_imgL = cv2.drawKeypoints(imgL_color, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    key_points_imgR = cv2.drawKeypoints(imgR_color, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('../output_images/task2_sift1.jpg', key_points_imgL)
    cv2.imwrite('../output_images/task2_sift2.jpg', key_points_imgR)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, k = 2)

    good_matches = []
    ptsL = []
    ptsR = []
    for m, n in matches:
       if m.distance < 0.75*n.distance:
           good_matches.append([m])
           ptsL.append(kp1[m.queryIdx].pt)
           ptsR.append(kp2[m.trainIdx].pt)

    matches_knn = cv2.drawMatchesKnn(imgL_color, kp1, imgR_color, kp2, good_matches, outImg=None, matchColor = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('../output_images/task2_matches_knn.jpg', matches_knn)

    ptsL = np.int32(ptsL)
    ptsR = np.int32(ptsR)

    # Calculate fundamental matrix
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.RANSAC)
    print ('Fundamental Matrix')
    print(F)

    # Select inlier points
    ptsL = ptsL[mask.ravel() == 1]
    ptsR = ptsR[mask.ravel() == 1]

    inliersL = []
    inliersR = []

    # selecting 10 inlier match pairs
    for i in [np.random.randint(0, len(ptsL) - 1) for x in range (10)]:
        inliersL.append(ptsL[i])
        inliersR.append(ptsR[i])

    inliersL = np.int32(inliersL)
    inliersR = np.int32(inliersR)

    linesR = (cv2.computeCorrespondEpilines(inliersL.reshape(-1, 1, 2), 1, F)).reshape(-1, 3)
    img1 = get_epilines(imgR_gray, linesR, inliersR, inliersL)

    linesL = (cv2.computeCorrespondEpilines(inliersR.reshape(-1, 1, 2), 2, F)).reshape(-1, 3)
    img2 = get_epilines(imgL_gray, linesL, inliersL, inliersR)

    cv2.imwrite('../output_images/task2_epi_right.jpg', img1)
    cv2.imwrite('../output_images/task2_epi_left.jpg', img2)

    stereo = cv2.StereoBM_create(numDisparities = 64, blockSize = 27)
    imageDisparity = stereo.compute(imgL_gray, imgR_gray)
    cv2.imwrite('../output_images/task2_disparity.jpg', imageDisparity)


if __name__ == '__main__':
    task2()
