import cv2
import numpy as np

def process_frame(frame):
    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur
    gray = cv2.GaussianBlur(gray, (11, 11), 15)

    # dilation - make all the crevice thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    edges = cv2.dilate(gray, kernel, iterations=1)

    #threshold
    binary = cv2.adaptiveThreshold(
        edges,
        255, #white color
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, #inverse: the light parts turn becomes black
        13, 
        3
    )

    #morphologies
    #reduce the small white noises
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    #connects the white lines to make them more fluent
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big)

    # edges drawn in white (100 and 200 are the thresholds)
    edges = cv2.Canny(binary, 100, 200)

    #return edges

    #Hough line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=80, maxLineGap=20)
    #draw on original frame
    out = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(out, (x1, y1), (x2, y2), (100, 255, 0), 3)
    return out

