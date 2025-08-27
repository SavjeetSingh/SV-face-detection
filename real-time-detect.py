import cv2
import numpy as np


# loading the Caffe model
print("[INFO] loading model...")
prototxt = r"C:\Users\sam\Documents\GitHub\SV-face-detection\deploy.prototxt"
model= r"C:\Users\sam\Documents\GitHub\SV-face-detection\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    (h,w) = frame.shape[:2]

    # now preprocess the frame to create input blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

    # passing the blob through the network and get the detections
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loopingggggg
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            # creating the box and confidence level on the frame
            text = "{:.2f}%".format(confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0,255,0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()