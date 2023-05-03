import jetson.inference
import jetson.utils
import cv2
import numpy as np

net = jetson.inference.detectNet(
    argv=["--model=/home/duburi/misa/jetson-inference/python/training/detection/ssd/models/sauvc/ssd-mobilenet.onnx",
          "--labels=/home/duburi/misa/jetson-inference/python/training/detection/ssd/models/sauvc/labels.txt",
          "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

bbox = None
tracker = None
detect_threshold = 0.9

while True:
    success, img = cap.read()
    imgCuda = jetson.utils.cudaFromNumpy(img)

    # If there is no tracker or the detection threshold is met, detect the object
    if not tracker or detections[0].Confidence > detect_threshold:
        # Detect objects in the frame
        detections = net.Detect(imgCuda, overlay="OVERLAY_NONE")

        # Get the first detection
        detection = detections[0]

        # Get the bounding box coordinates
        x1, y1, x2, y2 = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

        # Initialize the tracker with the bounding box
        bbox = (x1, y1, x2 - x1, y2 - y1)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(img, bbox)

    # If there is a tracker, update the bounding box
    else:
        success, bbox = tracker.update(img)
        x1, y1, w, h = [int(v) for v in bbox]
        x2, y2 = x1 + w, y1 + h

    # Draw the bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Display the image
    cv2.imshow("duburi", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
