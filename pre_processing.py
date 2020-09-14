import cv2 as cv
import sys
import numpy as np
import tifffile as ti
import argparse
import itertools
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3


def CannyThreshold(val):
    low_threshold = val
    #img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(src_gray, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)


# Sort grey image colors by frequency of appearance
def freq_sort(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)

    frequencies = {}
    for item in flat_list:
        if item in frequencies:
            frequencies[item] += 1
        else:
            frequencies[item] = 1
    return sorted(frequencies.items(), key=lambda x: x[1], reverse=True)


# Remove colors of selection ranked by frequency
def gray_filter(img, p_map, start, end):
    # Slice the color range
    p_map = p_map[start:end]

    # Break down the dic
    selected_colors = []
    for p in p_map:
        selected_colors.append(p[0])

    # Replace out-off-range colors with black
    r_len = len(img)
    c_len = len(img[0])
    for i in range(r_len):
        for j in range(c_len):
            if img[i][j] not in selected_colors:
                img[i][j] = 0
    return img


# Remove disconnected noises
def de_noise(img, kernel_size=1, criteria=4, iterations=4, remove_all=False):
    cur = 0
    r_len = len(img)
    c_len = len(img[0])
    while cur < iterations:
        cur += 1
        for i in range(r_len):
            for j in range(c_len):
                # If the iterated pixel is already black
                if img[i][j] == 0:
                    continue
                try:
                    # X, Y = np.mgrid[j:j+kernel_size, i:i+kernel_size]
                    # print(np.vstack((X.ravel(), Y.ravel())))
                    # exit(1)
                    # Put adjacent pixels with given kernel size into the list
                    p_list = []
                    indices = [p for p in itertools.product(range(kernel_size, -kernel_size-1, -1), repeat=2) if p != (0,0)]
                    for idx in indices:
                        p_list.append(img[i+idx[0]][j+idx[1]])

                    # Remove the pixel if number of adjacent black pixels are greater than the preset value
                    if p_list.count(0) > criteria:
                        img[i][j] = 0
                        if remove_all:
                            for idx in indices:
                                img[i+idx[0]][j+idx[1]] = 0
                except IndexError:
                    pass
    return img


if __name__ == '__main__':
    src = cv.imread(cv.samples.findFile("input.tif"))
    img = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow('original', img_gray)

    freq_dic = freq_sort(img_gray)
    filtered_img = gray_filter(img_gray, freq_dic, 10, -80)
    cv.imshow('filtered', filtered_img)
    ti.imwrite("filtered.tif", np.array([[filtered_img] * 90], np.uint8))

    # de_noise_img = de_noise(filtered_img, 1, 4, 4)
    # de_noise_img = de_noise(de_noise_img, 2, 18, 1)

    de_noise_img = de_noise(filtered_img, 1, 5, 4)
    ti.imwrite("de_noise_img.tif", np.array([[de_noise_img] * 90], np.uint8))

    eroded = cv.dilate(de_noise_img, np.ones((2, 2), np.uint8), iterations=1)
    dilated = cv.dilate(eroded, np.ones((2, 2), np.uint8), iterations=1)

    med_blur = cv.medianBlur(de_noise_img, 3)
    cv.imshow('dilated', dilated)
    cv.imshow('de-noised-more-aggressive', de_noise_img)
    cv.imshow('med_blur', med_blur)

    cv.waitKey()

    # img_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # print(img_gray)
    # if img is None:
    #     sys.exit("Could not read the image.")
    #
    #
    # rows, cols, channels = img.shape
    # dst = img.copy()
    # a = 2.5
    # b = 380
    # for i in range(rows):
    #     for j in range(cols):
    #         for c in range(3):
    #             color = img[i, j][c]*a+b
    #             if color > 255:           # 防止像素值越界（0~255）
    #                 dst[i, j][c] = 255
    #             elif color < 0:           # 防止像素值越界（0~255）
    #                 dst[i, j][c] = 0
    #
    # blur_img = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
    # gaussian_gray = cv.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1, sigmaY=1)
    # ti.imwrite("Gaussian_blur.tif", np.array([[gaussian_gray]*90], np.uint8))
    #
    # med_blur_img = cv.medianBlur(img_gray, 3)
    # ti.imwrite("med_blur.tif", np.array([[med_blur_img]*90], np.uint8))
    #
    # ret, threshold = cv.threshold(blur_img, 85, 255, cv.THRESH_TOZERO_INV)
    # ret_gray, threshold_gray = cv.threshold(gaussian_gray, 85, 255, cv.THRESH_TOZERO_INV)
    #
    # kernel = np.ones((2, 2), np.uint8)
    # erosion = cv.erode(threshold, kernel, iterations=2)
    # erosion_gray = cv.erode(threshold_gray, kernel, iterations=2)
    # ti.imwrite("erosion.tif", np.array([[erosion_gray]*90], np.uint8))
    #
    # dilation = cv.dilate(erosion, kernel, iterations=2)
    # dilation_gray = cv.dilate(threshold_gray, kernel, iterations=2)
    # ti.imwrite("dilation.tif", np.array([[dilation_gray]*90], np.uint8))
    #
    # lower_grey = np.array([0, 0, 11])
    # upper_grey = np.array([0, 0, 60])
    # mask = cv.inRange(erosion, lower_grey, upper_grey)
    # mask = cv.fastNlMeansDenoising(mask, None, 5)
    # res = cv.bitwise_and(erosion, erosion, mask=mask)
    # res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    # ti.imwrite("filtered.tif", np.array([[res_gray]*90], np.uint8))
    #
    # # gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    # # grad_x = cv.Sobel(gray, -1, 1, 0, ksize=5)
    # # grad_y = cv.Sobel(gray, -1, 0, 1, ksize=5)
    # # grad = cv.addWeighted(grad_x, 1, grad_y, 1, 0)
    #
    # # src = cv.GaussianBlur(src, (3, 3), 0)
    # # src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # # cv.namedWindow(window_name)
    # # cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
    # # CannyThreshold(0)
    # # cv.waitKey()
    #
    # cv.imshow("src", img)
    # cv.imshow("blur", blur_img)
    # cv.imshow("threshold", threshold)
    #
    # cv.imshow("erosion", erosion)
    # cv.imshow("dilation", dilation)
    #
    # cv.imshow('mask', mask)
    # cv.imshow('filtered', res)
    #
    # # cv.imshow("grad", grad)
    # cv.imshow("blur", blur_img)
    #
    # k = cv.waitKey(0)
    # if k == ord("s"):
    #     cv.imwrite("starry_night.png", erosion)

