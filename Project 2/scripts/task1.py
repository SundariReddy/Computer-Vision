#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import cv2
import numpy as np
import random


def task1():
    img1_color = cv2.imread(r'../resources/mountain1.jpg')
    img2_color = cv2.imread(r'../resources/mountain2.jpg')

    img1_gray = cv2.imread(r'../resources/mountain1.jpg', 0)
    img2_gray = cv2.imread(r'../resources/mountain2.jpg', 0)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    key_points_img1 = cv2.drawKeypoints(img1_color, kp1, outImage = np.array([]), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    key_points_img2 = cv2.drawKeypoints(img2_color, kp2, outImage = np.array([]), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('../output_images/task1_sift1.jpg', key_points_img1)
    cv2.imwrite('../output_images/task1_sift2.jpg', key_points_img2)


    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches_knn = matcher.knnMatch(des1, des2, 2)
    ratio_thresh = 0.75
    good_matches = []
    pts1 = []
    pts2 = []
    for m,n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)


    task1_matches_knn = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None, matchColor = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('../output_images/task1_matches_knn.jpg', task1_matches_knn)

    H, mask =  cv2.findHomography(np.asarray(pts1), np.asarray(pts2), cv2.RANSAC)
    print ('Homography Matrix')
    print (H)
    iH = np.linalg.inv(H)

    matchesMask = mask.ravel().tolist()

    task1_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, np.random.choice(good_matches,10), None, matchColor = (0, 255, 0), matchesMask = random.sample(matchesMask, 10), flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('../output_images/task1_matches.jpg', task1_matches)

    rows1, cols1 = img1_gray.shape
    rows2, cols2 = img2_gray.shape

    lp1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    lp2 = cv2.perspectiveTransform(temp, H)
    lp = np.concatenate((lp1, lp2), axis = 0)

    [x_min, y_min] = np.int32(lp.min(axis = 0).ravel() - 0.5)
    [x_max, y_max] = np.int32(lp.max(axis = 0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img1_color, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    result[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img2_color

    cv2.imwrite('../output_images/task1_pano.jpg', result)


if __name__ == '__main__':
    task1()
