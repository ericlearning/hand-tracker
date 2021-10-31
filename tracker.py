import cv2
import numpy as np
import mediapipe as mp

mp_utils = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cv2.namedWindow('image')

cap = cv2.VideoCapture(1)

def detect_motion(ps):
    p_max, p_min = ps.max(0), ps.min(0)
    ps = (ps - p_min) / (p_max - p_min)
    dist = ((ps[4] - ps[8])**2).sum()
    print(dist < 0.2)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        retval, img = cap.read()

        img.flags.writeable = False
        out = hands.process(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img.flags.writeable = True
        if out.multi_hand_landmarks:
            for cur_landmark in out.multi_hand_landmarks:
                ps = [(p.x, p.y, p.z) for p in cur_landmark.landmark]
                ps = np.array(ps)

                detect_motion(ps)

                mp_utils.draw_landmarks(
                    img,
                    cur_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        cv2.imshow('image', img[:, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
