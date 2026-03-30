import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Volume control (Windows)
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Brightness
import screen_brightness_control as sbc


# -----------------------
# Volume Setup
# -----------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]


# -----------------------
# Mediapipe Setup
# -----------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0)

mode = "NONE"
last_switch = 0


def fingers_up(lmList):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if lmList[tips[0]][0] > lmList[tips[0]-1][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if lmList[tips[i]][1] < lmList[tips[i]-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # 👉 BLACK SCREEN
    img = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []
            h, w, c = frame.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            fingers = fingers_up(lmList)
            current_time = time.time()

            # -----------------------
            # MODE SWITCH
            # -----------------------

            if fingers == [0,1,1,0,0] and current_time - last_switch > 1:
                mode = "VOLUME"
                last_switch = current_time

            elif fingers == [0,1,1,1,0] and current_time - last_switch > 1:
                mode = "BRIGHTNESS"
                last_switch = current_time

            # Thumb & Index
            x1, y1 = lmList[4]
            x2, y2 = lmList[8]

            length = math.hypot(x2 - x1, y2 - y1)

            # Draw UI
            cv2.circle(img, (x1, y1), 10, (255,0,0), -1)
            cv2.circle(img, (x2, y2), 10, (255,0,0), -1)
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 3)

            # -----------------------
            # VOLUME MODE
            # -----------------------
            if mode == "VOLUME":

                vol = np.interp(length, [30,200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                bar = np.interp(length, [30,200], [400,150])
                per = np.interp(length, [30,200], [0,100])

                cv2.putText(img, "VOLUME MODE", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # -----------------------
            # BRIGHTNESS MODE
            # -----------------------
            elif mode == "BRIGHTNESS":

                brightness = np.interp(length, [30,200], [0,100])
                sbc.set_brightness(int(brightness))

                bar = np.interp(length, [30,200], [400,150])
                per = brightness

                cv2.putText(img, "BRIGHTNESS MODE", (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            else:
                bar = 400
                per = 0

            # Bar UI
            cv2.rectangle(img,(50,150),(85,400),(255,255,255),2)
            cv2.rectangle(img,(50,int(bar)),(85,400),(0,255,0),-1)

            cv2.putText(img,f'{int(per)} %',(40,450),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("AI Control System", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()