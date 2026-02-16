import cv2
from flask import Flask, Response

import numpy as np              # explicit numpy
import threading, time          # thread + sleep

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# frame buffer (stores numpy arrays)
frame_buffer = [None] * 5 #gives five spots for storing frames
buf_lock = threading.Lock() #makes the reading + writing independent from each other
buf_i = [0] #index

def camera_reader(): #calls the cap.read() only one time for 2 videos, avoid getting stuck
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame = np.asarray(frame)  # stores in an array

        with buf_lock:
            # write into current slot
            frame_buffer[buf_i[0]] = frame

            # move index forward (wrap around)
            buf_i[0] = buf_i[0] + 1
            if buf_i[0] == len(frame_buffer):
                buf_i[0] = 0


threading.Thread(target=camera_reader, daemon=True).start()  # CHANGED

#returns the most recent frame
def get_latest():
    with buf_lock:
        # newest frame is the one right before the next write position
        idx = buf_i[0] - 1
        if idx < 0:
            idx = len(frame_buffer) - 1

        f = frame_buffer[idx]
        if f is None:
            return None
        return f.copy()

#raw video stream
def gen():
    while True:
        frame = get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

#processed video stream
def processed_gen():
    while True:
        frame = get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        
        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # edges
        edges = cv2.Canny(gray, 60, 180)
        # make tape edges thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Hough line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                minLineLength=80, maxLineGap=20)
        # draw on original frame
        out = frame.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        ok, buf = cv2.imencode(".jpg", out)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/stream")
def stream():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_processed")
def stream_proc():
    return Response(processed_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
