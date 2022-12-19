# This is a sample Python script.
import cv2
import numpy as np
from PIL import ImageGrab


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def motion_detector():
    previous_frame = None
    while True:
        # load image
        # convert to RGB
        # Grab screenshot of the screen
        img_brg = np.array(ImageGrab.grab())

        # convert brg to rgb
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        # convert color to Gray
        prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        if previous_frame is None:
            # assign the previous frame with current prepared frame
            previous_frame = prepared_frame
            continue

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)

        previous_frame = prepared_frame

        # 2d array filled with ones
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # only take different areas that are different enough
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
        #                 lineType=cv2.LINE_AA)
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)

        cv2.imshow('Motion detector', img_rgb)

        if (cv2.waitKey(30) == 27):
            break

    cv2.destroyAllWindows()


motion_detector()
