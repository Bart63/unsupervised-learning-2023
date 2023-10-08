from scipy.ndimage import rotate as rt
import cv2
import math
import numpy as np


def rotate(image, p=0.3, degree=15):
    if np.random.random() < p:
        degree = np.random.uniform(-degree, degree)
        image = rt(image, degree, reshape=False)
    return image


def scale(image, p=0.3, factor=0.08):
    random_number = np.random.random()
    shape = image.shape
    if random_number < p:
        factor = 1 + np.random.uniform(-factor, factor)
        if random_number < p/2:
            new_width = int(image.shape[0] * factor)
            image = cv2.resize(image, (new_width, shape[1]), interpolation=cv2.INTER_LINEAR)
        else:
            new_height = int(image.shape[1] * factor)
            image = cv2.resize(image, (shape[0], new_height),
                               interpolation=cv2.INTER_LINEAR)

        if factor < 1:
            pad_width = max(shape[0] - image.shape[0], 0)
            pad_height = max(shape[0] - image.shape[1], 0)
            image = cv2.copyMakeBorder(image, pad_width // 2, pad_width - pad_width // 2, pad_height // 2,
                                       pad_height - pad_height // 2, cv2.BORDER_CONSTANT, value=0)
        crop_x = (image.shape[1] - shape[1]) // 2
        crop_y = (image.shape[0] - shape[0]) // 2
        image = image[crop_y:crop_y + shape[0], crop_x:crop_x + shape[1]]
    return image


def one_hot(image):
    image = np.where(image > 255/2, 255, 0)
    return image


def salt_and_pepper_noise(image, p=0.01):
    salt_noise = np.random.rand(*image.shape) < p
    pepper_noise = np.random.rand(*image.shape) < p
    image[salt_noise] = 255
    image[pepper_noise] = 0
    return image


def folding_lines(image, num_lines):
    for _ in range(num_lines):
        line_length = np.random.randint(0, min(image.shape))
        line_angle_degrees = np.random.randint(0, 90)
        x1 = np.random.randint(0, image.shape[1] - 1)
        y1 = np.random.randint(0, image.shape[0] - 1)
        line_angle_radians = math.radians(line_angle_degrees)
        x2 = int(x1 + line_length * math.cos(line_angle_radians))
        y2 = int(y1 + line_length * math.sin(line_angle_radians))
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return image






