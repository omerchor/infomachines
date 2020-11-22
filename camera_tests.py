import time

import numpy as np
import cv2
from matplotlib import pyplot as plt
from contextlib import contextmanager

# Channel indices
BLUE = 0
GREEN = 1
RED = 2


cap = cv2.VideoCapture("25.1cm.mp4")
fgbg = cv2.createBackgroundSubtractorKNN(history=1000,
                                         dist2Threshold=100,
                                         detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorMOG2(#history=100,
#                                           #varThreshold=100,
#                                           detectShadows=True)

try:
    # cap.set(3, 640)
    # cap.set(4, 360)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            fgmask = fgbg.apply(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            shadows = np.where(fgmask == 127)
            fgmask[shadows] = 0

            foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
            # added_image = cv2.addWeighted(foreground, 0.5, gray, 1, 0)

            cv2.imshow('frame', foreground)

            # out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # out.release()
        pass
finally:
    cap.release()
    cv2.destroyAllWindows()

