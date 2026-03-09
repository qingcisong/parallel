import cv2
import numpy as np

def process_frame(frame):

    # preprocessing 
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv_img[:,:,0]
    s = hsv_img[:,:,1]
    v = hsv_img[:,:,2]

    new_s = cv2.multiply(s, 1.6)
    new_hsv = cv2.merge([h, new_s, v])

    gray = cv2.cvtColor(new_hsv, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9,9), 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    edges = cv2.dilate(blur,kernel,iterations=1)

    binary = cv2.adaptiveThreshold(
        edges,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        13,3
    )

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big)

    edges = cv2.Canny(binary,100,200)

    #  ROI 
    h, w = frame.shape[:2]

    roi_pts = np.array([[
        (int(0.34*w), int(0*h)),
        (int(0.66*w), int(0*h)),
        (int(1.0*w), int(1.0*h)),
        (0, int(1.0*h))
    ]], dtype=np.int32)

    mask = np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,roi_pts,255)

    roi = cv2.bitwise_and(edges,edges,mask=mask)

    #  Hough line detection 
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi/180,
        80,
        minLineLength=80,
        maxLineGap=40
    )

    out = frame.copy()

    left_lines = []
    right_lines = []

    if lines is not None:

        for x1,y1,x2,y2 in lines[:,0]:

            dx = x2-x1
            dy = y2-y1

            if dx == 0:
                continue

            slope = dy/dx

            # ignore horizontal noise
            if abs(slope) < 0.5:
                continue

            # left lane
            if slope < 0:
                left_lines.append((x1,y1,x2,y2))

            # right lane
            else:
                right_lines.append((x1,y1,x2,y2))

    # average lanes 
    def average_line(lines):

        if len(lines) == 0:
            return None

        xs = []
        ys = []

        for x1,y1,x2,y2 in lines:
            xs += [x1,x2]
            ys += [y1,y2]

        fit = np.polyfit(ys,xs,1)

        y1 = h
        y2 = int(h*0.5)

        x1 = int(fit[0]*y1 + fit[1])
        x2 = int(fit[0]*y2 + fit[1])

        return (x1,y1,x2,y2)

    left = average_line(left_lines)
    right = average_line(right_lines)

    #  draw left/right 
    if left is not None:
        cv2.line(out,(left[0],left[1]),(left[2],left[3]),(0,255,0),4)

    if right is not None:
        cv2.line(out,(right[0],right[1]),(right[2],right[3]),(0,255,0),4)

    #  center line 
    if left is not None and right is not None:

        cx1 = int((left[0]+right[0])/2)
        cy1 = int((left[1]+right[1])/2)

        cx2 = int((left[2]+right[2])/2)
        cy2 = int((left[3]+right[3])/2)

        cv2.line(out,(cx1,cy1),(cx2,cy2),(255,0,0),4)

    return out
