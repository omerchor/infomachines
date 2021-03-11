import cv2
import time
import numpy as np


def main():
    cap = cv2.VideoCapture(r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\25.1cm.mp4")
    ret = True
    try:
        while ret:
            # ret, frame = cap.read(1)
            ret, frame = cap.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (40, 50, 0), (85, 255, 255))
                green = np.zeros_like(frame, np.uint8)
                imask = mask > 0
                green[imask] = frame[imask]

                mask = cv2.inRange(hsv, (150, 50, 100), (180, 255, 255)) + cv2.inRange(hsv, (0, 50, 100), (10, 255, 255))
                red = np.zeros_like(frame, np.uint8)
                imask = mask > 0
                red[imask] = frame[imask]

                gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 113, 255, 0)
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Extract boundaries of contours (just a technical line because output is a numpy object)

                for cnt in contours:
                    # Find moments of the contour
                    M = cv2.moments(cnt)
                    # Extract contour center (x,y)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                    cv2.circle(frame, center, 5, (255, 255, 255), cv2.FILLED)

                cv2.imshow("output", frame)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    ret = False
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # for i in range(10):
    #     generate_qrcodes(i)

