import cv2
import mediapipe as mp
import time
import math
import os

from HandsTrackingModule import handDetector


def set_volume_mac(volume_percent):
    # volume_percent: 0–100
    volume_percent = max(0, min(100, int(volume_percent)))  # clamp to [0,100]
    os.system(f"osascript -e 'set volume output volume {volume_percent}'")


def get_volume_mac():
    out = os.popen("osascript -e 'output volume of (get volume settings)'").read()
    try:
        return int(out.strip())
    except ValueError:
        return 50


def main():
    wCam, hCam = 640, 480

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = handDetector(detectionCon=0.7)
    pTime = 0

    while True:

        success, img = cap.read()


        if not success:
            break

        # Find Hand
        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Tip of thumb: id=4, Tip of index finger: id=8

            #filter based on size

            #find distance between index and thumb

            #convert volume

            #reduce resolution to make it smoother

            #check fingersup

            #if pinky is down set volume

            #drawings

            #frame rate
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand range 50–300
            # Volume range 0–100
            vol = np.interp(length, [50, 300], [0, 100])
            set_volume_mac(vol)

            if length < 50:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np
    main()
