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
    vol = 0
    volBar = 400

    detector = handDetector(detectionCon=0.7)
    pTime = 0
    area = 0;

    while True:

        success, img = cap.read()


        if not success:
            break

        # Find Hand
        img = detector.findHands(img, draw=True)
        lmList,bbox = detector.findPosition(img, draw=True)


        if len(lmList) != 0:
            # Tip of thumb: id=4, Tip of index finger: id=8

            #filter based on size

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])//100
            print(area)

            if(250 <= area <= 1000):
                #find distance between index and thumb
                length,img,lineInfo = detector.findDistance(4,8,img)
                #print(length)

                #convert volume
                vol = np.interp(length, [50, 300], [0, 100])
                volBar =  np.interp(length, [50, 300], [400, 140])


                #reduce resolution to make it smoother


                #check fingersup
                fingers = detector.fingersUp()



                #if pinky is down set volume
                if(fingers[4] == 0):
                    set_volume_mac(vol)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

                #drawings

                #frame rate





                # Hand range 50–300
                # Volume range 0–100




        cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Vol: {int(vol)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)
        cVol = get_volume_mac()
        cv2.putText(img, f'Vol Set: {int(cVol)}%', (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)
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
