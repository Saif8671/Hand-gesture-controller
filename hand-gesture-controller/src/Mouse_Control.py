import cv2
import mediapipe as mp
import pyautogui
import numpy as np

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm_list = []

        for id, lm in enumerate(hand_landmarks.landmark):
            lm_list.append((id, int(lm.x * w), int(lm.y * h)))

        # Determine which fingers are up
        fingers = []

        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for i in range(1, 5):
            if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Move mouse with index finger
        if fingers[1] == 1 and sum(fingers) == 1:
            x = np.interp(lm_list[8][0], [0, w], [0, screen_w])
            y = np.interp(lm_list[8][1], [0, h], [0, screen_h])
            pyautogui.moveTo(screen_w - x, y)

        # Left click with index + middle
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            pyautogui.click()

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
