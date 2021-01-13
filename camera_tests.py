import math
import time
from functools import partial

import imutils as imutils
import numpy as np
import cv2
import pims
import skimage
import trackpy as tp

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion, skeletonize, convex_hull, opening, remove_small_objects
from skimage.color import label2rgb

tp.enable_numba()

IS_360 = True
YMIN = 0 if IS_360 else 30
YMAX = 230 if IS_360 else 480
XMIN = 150 if IS_360 else 310
XMAX = 500 if IS_360 else 980

fgbg = cv2.createBackgroundSubtractorKNN(history=100,
                                         dist2Threshold=1000,
                                         detectShadows=True)
sift = cv2.SIFT_create(contrastThreshold=0.15,
                       edgeThreshold=0.1)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))


@pims.pipeline
def crop(img):
    """
    Crop the image to select the region of interest
    """
    x_min = XMIN
    x_max = XMAX
    y_min = YMIN
    y_max = YMAX
    return img[y_min:y_max,x_min:x_max]

# Channel indices
BLUE = 0
GREEN = 1
RED = 2


def evaluate_frame(frame):
    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)
    # gray = cv2.cvtColor(crop(frame), cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    shadows = np.where(fgmask == 127)
    fgmask[shadows] = 0

    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
    # threshold = 50
    # gray = cv2.blur(gray, (1, 1))
    # canny_output = cv2.Canny(gray, threshold, threshold * 2, apertureSize=3)
    #
    # contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # cnt = contours[4]
    # img = cv2.drawContours(gray, contours, 0, (0, 255, 0), 3)

    # cv2.imshow('foreground', foreground)
    # cv2.imshow('canny', canny_output)
    # cv2.imshow('contours', img)
    # green = frame[:,:,1]
    # f = tp.locate(green, 31, percentile=99)
    # for j in f.iterrows():
    #     cv2.circle(frame, (int(j[1][1]), int(j[1][0])), int(j[1][3]), 255, -1)

    # cv2.imshow('foreground', frame)

    # bw = closing(fgmask > 1, square(3))
    cleared = erosion(fgmask)
    label_img = label(cleared)
    label_img = remove_small_objects(label_img, min_size=100)
    regions = regionprops(label_img)

    frame = foreground
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        minr, minc, maxr, maxc = props.bbox
        cv2.circle(frame, (int(x0), int(y0)), 5, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0))
        cv2.line(frame, (int(x0), int(y0)), (int(x2), int(y2)), (0, 0, 255))
        cv2.rectangle(frame, (minc, minr), (minc + maxc - minc, minr + maxr - minr), (0, 255, 0), 2)

    return frame


def get_frames(path):
    cap = cv2.VideoCapture(path)
    ret = True
    try:
        while ret is True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            yield frame
    finally:
        cap.release()


def main():
    cap = cv2.VideoCapture("25.1cm.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\OmerChor\Pictures\Camera Roll\batch_1\WIN_20201209_14_36_21_Pro.mp4")
    # kernel = np.ones((3, 3), np.uint8)

    try:
        # cap.set(3, 640)
        # cap.set(4, 360)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.mov', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        try:
            i = 0
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                im = evaluate_frame(frame)
                i += 1
                # if i > 200:
                cv2.imshow("result", im)
                #     out.write(foreground)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            out.release()
    finally:
        cap.release()
        cv2.destroyAllWindows()


# def main_iterate():
#     fig = plt.figure()
#     data = np.zeros((640, 360))
#     im = plt.imshow(data)
#     frames_func = partial(get_frames, "25.1cm.mp4")
#     anim = animation.FuncAnimation(fig, evaluate_frame, frames=frames_func)
#     plt.show()


if __name__ == "__main__":
    main()
