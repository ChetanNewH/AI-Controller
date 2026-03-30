import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append((cx, cy))

            # Thumb tip
            x1, y1 = lmList[4]

            # Index tip
            x2, y2 = lmList[8]

            # Draw points
            cv2.circle(img, (x1, y1), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255,0,0), cv2.FILLED)

            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

            # Calculate distance
            length = np.hypot(x2-x1, y2-y1)

            # Convert distance to brightness
            brightness = np.interp(length,[30,200],[0,100])

            sbc.set_brightness(int(brightness))

            # Brightness bar
            bar = np.interp(length,[30,200],[400,150])

            cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
            cv2.rectangle(img,(50,int(bar)),(85,400),(0,255,0),cv2.FILLED)

            cv2.putText(img,f'{int(brightness)} %',(40,450),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Brightness Controller", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()