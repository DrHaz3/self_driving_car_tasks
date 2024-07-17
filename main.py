import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use('TkAgg')


def make_coordinates(_image, line_parameters):
    print(line_parameters.shape)
    slope, intercept = line_parameters
    y1 = _image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(_image, _lines):
    left_fit = []
    right_fit = []
    for line in _lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    print('this is left')
    left_line = make_coordinates(_image, left_fit_average)
    print('this is right')
    right_line = make_coordinates(_image, right_fit_average)
    return np.array([left_line, right_line])


def display_lines(_image, _lines):
    _line_image = np.zeros_like(_image)
    if _lines is not None:
        for x1, y1, x2, y2 in _lines:
            cv.line(_line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return _line_image


def canny(_image):
    gray = cv.cvtColor(_image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=7.5, tileGridSize=(3, 3))
    enhanced_gray = clahe.apply(gray)
    # blur = cv.GaussianBlur(enhanced_gray, (5, 5), 0)
    _canny = cv.Canny(enhanced_gray, 50, 150)
    return _canny


def region_of_interest(_image):
    height = _image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(_image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(_image, mask)
    return masked_image


# image = cv.imread('test_image.jpg')
# lane_image = np.copy(image)
# edges = canny(lane_image)
# cropped_image = region_of_interest(edges)
# lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combined_image = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv.imshow('result', combined_image)
# cv.waitKey(0)

start = time.time()
cap = cv.VideoCapture('test2.mp4')
while cap.isOpened():
    _, frame = cap.read()
    end = time.time()
    print(start - end, 'this is time')
    edges = canny(frame)
    cropped_image = region_of_interest(edges)
    lines = cv.HoughLinesP(cropped_image, 1, np.pi / 360, 100, np.array([]), minLineLength=30, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combined_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow('result', combined_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

"""this is a test for git training
hello guys!!!"""

cap.release()
cv.destroyAllWindows()
