
# fix for very slow cv2.VideoCapture startup time
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2


vid = cv2.VideoCapture(0)
vid.set(3, 200)
vid.set(4, 200)

while True:
    # inside infinity loop
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
