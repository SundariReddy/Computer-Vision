import cv2
import numpy as np


def task3():

    images = ["../resources/task3/pos_2.jpg",
                "../resources/task3/pos_7.jpg",
                "../resources/task3/pos_14.jpg"]

    template = cv2.imread('../resources/task3/template.png', 0)

    for i in range (len(images)):
        image = cv2.imread(images[i], 0)

        image1 = cv2.GaussianBlur(image,( 3, 3 ), 2)
        image1 = cv2.Laplacian(image1,cv2.CV_8U)
        ret, image1 = cv2.threshold(image1, 24, 255.0, cv2.THRESH_BINARY)
        image1 = cv2.GaussianBlur(image1,(3,3),1)

        temp = cv2.GaussianBlur(template,(3,3),2)
        temp = cv2.Laplacian(temp, cv2.CV_8U)
        ret, image1 = cv2.threshold(image1, 30, 255.0, cv2.THRESH_BINARY)
        temp = cv2.GaussianBlur(temp,(3,3),1)
        temp = cv2.resize(temp, (0, 0), fx=0.6, fy=0.6)

        result = cv2.matchTemplate(image1,temp, cv2.TM_CCORR_NORMED)

        width = temp.shape[0]
        height = temp.shape[1]

        threshold = 0.75

        loc = np.where(result>= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (255,255,255),2)

        cv2.imwrite('../output_images/task3test'+str(i+1)+'.png', image)


if __name__ == '__main__':
    task3()
