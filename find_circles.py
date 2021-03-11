import cv2
import time
import numpy as np


def detect_stickers(frame, hsv_min=(40, 50, 0), hsv_max=(85, 255, 255)):
    """
    Detects regions within HSV colors in range of hsv_min and hsv_max.
    Default values are for Green stickers. Try changing only hue (first index)
    for different colors.

    Parameters
    ----------
    frame image to look for objects in
    hsv_min minimal values of h, s, and v in a tuple (h,s,v).
    hsv_max maximal values of h, s, and v in a tuple (h,s,v).

    Returns
    -------
    List of object centers
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    foreground = np.zeros_like(frame, np.uint8)
    imask = mask > 0
    foreground[imask] = frame[imask]

    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 113, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = [cv2.moments(cnt) for cnt in contours]
    centers = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) for m in moments]
    return centers


def main():
    cap = cv2.VideoCapture(r"C:\Users\OmerChor\OneDrive - mail.tau.ac.il\University\Physics\Info_machines\project\25.1cm.mp4")
    ret = True
    try:
        while ret:
            # ret, frame = cap.read(1)
            ret, frame = cap.read()
            if ret:
                centers = detect_stickers(frame)

                for cnt in centers:
                    cv2.circle(frame, cnt, 5, (255, 255, 255), cv2.FILLED)

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

