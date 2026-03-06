import cv2
import numpy as np

def process_frame(frame):
    #hsv
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv_img[:,:,0]
    s = hsv_img[:,:,1]
    v = hsv_img[:,:,2]
    s_change = 1.6
    new_s = cv2.multiply(s, s_change)
    new_hsv = cv2.merge([h, new_s, v])
    hsv_image = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    # grayscale
    gray = cv2.cvtColor(new_hsv, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (9, 9), 10)

    # dilation - make all the crevice thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    edges = cv2.dilate(blur, kernel, iterations=1)
    #return edges

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

    # ROI
    h, w = frame.shape[:2]
    roi_pts = np.array([[
        (int(0.34*w), int(0.00*h)),
        (int(0.66*w), int(0.00*h)),
        (int(1.00*w), int(1.00*h)),
        (int(0.0*w), int(1.00*h)),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, roi_pts, 255)
    
    roi = cv2.bitwise_and(edges, edges, mask=mask)
    return roi
    
    #threshold
    #binary = cv2.adaptiveThreshold(
        #roi,
        #255, #white color
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #cv2.THRESH_BINARY_INV, #inverse: the light parts turn becomes black
        #13,
        #3
    #)
    #return binary
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
