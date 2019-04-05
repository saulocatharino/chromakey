import cv2
import numpy as np


def get_frame(cap, scaling_factor):

    ret, frame = cap.read()


    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 1



    while True:
        fundo = cv2.imread('fundo.jpg')
        frame = get_frame(cap, scaling_factor)
        frame = cv2.flip(frame,1)
        fundo = fundo[0:frame.shape[0],0:frame.shape[1]]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        lower = np.array([30,100,100])
        upper = np.array([255,255,255])
        gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)


        mask = cv2.inRange(hsv, lower, upper)

        fundo = cv2.bitwise_and(fundo, fundo, mask=mask)

        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, 5)


        norm = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY),50, 255, 1)[1]
        norm = np.invert(norm)
        norm = cv2.dilate(norm, None, iterations=1)

        edged = cv2.erode(norm, None, iterations=1)
        res2 = cv2.bitwise_xor(frame, cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))
        res2 = cv2.bitwise_or(frame, res2)

        final1 = cv2.hconcat([frame, hsv])
        final2 =  cv2.hconcat([fundo, fundo+res2])
        final3 = cv2.vconcat([final1,final2])

        cv2.imshow('Resultado', final3 )


        c = cv2.waitKey(1)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()


