"""
scripts/preview_landmarks.py

Live webcam preview with MediaPipe hand landmark overlay.
Use this before recording to verify hand visibility and detection quality.
Press Q to quit.
"""

import argparse
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Preview running — press Q to quit.")
    print("Check: hand skeleton visible, no jitter at frame edges, good lighting.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(
                    frame,
                    "DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "NO DETECTION",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 220),
                    2,
                )

            cv2.imshow("AirSketch — landmark preview (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    main(p.parse_args().camera)
