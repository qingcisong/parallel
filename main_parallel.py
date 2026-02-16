import cv2
from flask import Flask, Response
            
import numpy as np              # numpy
import threading                # multi-threading
            
from processing_parallel import process_frame
        
app = Flask(__name__)

cap = cv2.VideoCapture(0)

# buffer (stores numpy arrays)
frame_buffer = [None] * 5 #five spots for storing frames
buf_lock = threading.Lock() #prevents the reading + writing at the same time
buf_i = [0] #index

#calls the cap.read() only one time for 2 videos, avoid getting stuck
def camera_reader(): 
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue   

        frame = np.asarray(frame)  # stores in an array

        with buf_lock:
            # write into current slot
            frame_buffer[buf_i[0]] = frame
            
            # move index forward
            buf_i[0] = buf_i[0] + 1
            if buf_i[0] == len(frame_buffer): #creates like a loop
                buf_i[0] = 0

#camera keeps reading frames, Flask keeps taking it out to process
threading.Thread(target=camera_reader, daemon=True).start()
               
#returns the most recent frame
def get_latest():
    with buf_lock:  
        # the camera_reader adds 1 after reading, to get back to the newest, you have to minus 1
        idx = buf_i[0] - 1  
        if idx < 0:
            idx = len(frame_buffer) - 1
            
        f = frame_buffer[idx] #gets out the newest frame
        if f is None:
            return None
        return f.copy()

#raw video stream
def gen():
    while True:
        frame = get_latest()
        if frame is None:
            continue

        ok, buf = cv2.imencode(".jpg", frame) #turns the numpy into a jpg image
        if not ok:
            continue

        #ontinuously sends, the boundary is each frame, /r/n is the standard line breaker
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

#processed video stream
def processed_gen():
    while True:   
        frame = get_latest()
        if frame is None:   
            continue
        
        out = process_frame(frame)
        
        ok, buf = cv2.imencode(".jpg", out)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

#when access /stream, the stream() is activated        
@app.route("/stream")
def stream():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_processed")
def stream_proc():
    return Response(processed_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":   
    app.run(host="0.0.0.0", port=5000, threaded=True)
  
