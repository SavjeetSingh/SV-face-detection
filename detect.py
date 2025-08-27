import cv2
import numpy as np


# loading the Caffe model
print("[INFO] loading model...")
prototxt = r"C:\Users\sam\Documents\GitHub\SV-face-detection\deploy.prototxt"
model= r"C:\Users\sam\Documents\GitHub\SV-face-detection\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# now to load the image and its dimensions
image = cv2.imread(r"C:\Users\sam\Documents\GitHub\SV-face-detection\gojo-face.jpg")
(h,w) = image.shape[:2]

# now preprocess the image to create input blob
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

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

        # creating the box and confidence level on the image
        text = "{:.2f}%".format(confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0,255,0), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
