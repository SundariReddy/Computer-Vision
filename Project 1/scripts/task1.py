import cv2


def task1():
    img = cv2.imread('resources/task1.png', 0)

    smooth_img = cv2.imread('resources/task1.png', 0)
    gx = cv2.imread('resources/task1.png', 0)
    gy = cv2.imread('resources/task1.png', 0)

    # SMOOTHING
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            smooth_img[x][y] = (((img[x-1][y-1] * 1) / 9) + ((img[x][y-1] * 1) / 9) + ((img[x+1][y-1] * 1) / 9) +
                        ((img[x-1][y] * 1) / 9) + ((img[x][y] * 1) / 9) + ((img[x+1][y] * 1) / 9) +
                        ((img[x-1][y+1] * 1) / 9) + ((img[x][y+1] * 1) / 9) + ((img[x+1][y+1] * 1) / 9))


    # Filtering with Sobel along the x direction
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            gx[x][y] = ((smooth_img[x-1][y-1] * -1) + (smooth_img[x][y-1] * -2) + (smooth_img[x+1][y-1] * -1) +
                        (smooth_img[x-1][y] * 0) + (smooth_img[x][y] * 0) + (smooth_img[x+1][y] * 0) +
                        (smooth_img[x-1][y+1] * 1) + (smooth_img[x][y+1] * 2) + (smooth_img[x+1][y+1] * 1)) / 8

    #Filtering with Sobel along the y direction
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            gy[x][y] = ((smooth_img[x-1][y-1] * -1) + (smooth_img[x][y-1] * 0) + (smooth_img[x+1][y-1] * 1) +
                        (smooth_img[x-1][y] * -2) + (smooth_img[x][y] * 0) + (smooth_img[x+1][y] * 2) +
                        (smooth_img[x-1][y+1] * -1) + (smooth_img[x][y+1] * 0) + (smooth_img[x+1][y+1] * 1)) / 8


    cv2.imwrite('Gx.png', gx)
    cv2.imwrite('Gy.png', gy)

if __name__ == '__main__':
    task1()
