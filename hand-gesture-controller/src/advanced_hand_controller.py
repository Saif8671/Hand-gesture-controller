import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

pyautogui.FAILSAFE = False

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]

# Smoothing
SMOOTHING = 0.2
prev_x, prev_y = 0, 0

# Gesture timing
last_action_time = 0
ACTION_COOLDOWN = 0.35

# Scroll state
prev_scroll_y = None

def fingers_up(lm, label):
    fingers = []

    # Thumb
    if label == "Right":
        fingers.append(lm[TIP_IDS[0]][0] > lm[TIP_IDS[0]-1][0])
    else:
        fingers.append(lm[TIP_IDS[0]][0] < lm[TIP_IDS[0]-1][0])

    # Other fingers
    for i in range(1, 5):
        fingers.append(lm[TIP_IDS[i]][1] < lm[TIP_IDS[i]-2][1])

    return fingers

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    gesture_text = "Idle"
    now = time.time()

    if res.multi_hand_landmarks:
        for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handedness.classification[0].label

            lm = [(int(pt.x*w), int(pt.y*h)) for pt in hand_lm.landmark]
            fingers = fingers_up(lm, label)
            count = sum(fingers)

            # =========================
            # RIGHT HAND → MOUSE
            # =========================
            if label == "Right":

                # Move mouse (Index only)
                if fingers[1] and count == 1:
                    x_cam, y_cam = lm[8]

                    x_scr = np.interp(x_cam, (0, w), (0, SCREEN_W))
                    y_scr = np.interp(y_cam, (0, h), (0, SCREEN_H))

                    curr_x = prev_x + (x_scr - prev_x) * SMOOTHING
                    curr_y = prev_y + (y_scr - prev_y) * SMOOTHING

                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    gesture_text = "Move"

                # Left click
                if fingers[1] and fingers[2] and count == 2 and now - last_action_time > ACTION_COOLDOWN:
                    pyautogui.click()
                    last_action_time = now
                    gesture_text = "Left Click"

                # Right click
                if fingers[1] and fingers[2] and fingers[3] and count == 3 and now - last_action_time > ACTION_COOLDOWN:
                    pyautogui.rightClick()
                    last_action_time = now
                    gesture_text = "Right Click"

            # =========================
            # LEFT HAND → SCROLL & VOLUME
            # =========================
            if label == "Left":

                # Scroll (open palm)
                if count == 5:
                    y = lm[8][1]
                    if prev_scroll_y:
                        delta = prev_scroll_y - y
                        pyautogui.scroll(int(delta * 1.5))
                    prev_scroll_y = y
                    gesture_text = "Scroll"

                else:
                    prev_scroll_y = None

                # Volume (thumb + index distance)
                if fingers[0] and fingers[1]:
                    dist = distance(lm[4], lm[8])
                    if dist > 70:
                        pyautogui.press("volumeup")
                        gesture_text = "Volume Up"
                    elif dist < 40:
                        pyautogui.press("volumedown")
                        gesture_text = "Volume Down"

            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Professional Hand Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
