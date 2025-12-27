import cv2
import mediapipe as mp

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm_list = []

        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = frame.shape
            lm_list.append((id, int(lm.x * w), int(lm.y * h)))

        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:
            finger_count += 1

        # Other fingers
        for i in range(1, 5):
            if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i]-2][2]:
                finger_count += 1

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Fingers: {finger_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    