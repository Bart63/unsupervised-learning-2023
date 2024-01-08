import cv2
import numpy as np


def opening(img, kernel_size=(2,2)):
    kernel = np.ones(kernel_size,np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def is_whitespace(img, thr=22, kernel_size=(2,2)):
    img = img.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if 255 not in img:
        img *= 255
    opened_img = opening(img, kernel_size)
    op_val = np.sum(opened_img)//255
    return op_val < thr
