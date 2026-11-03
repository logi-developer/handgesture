import cv2
import mediapipe as mp
import math

# ---------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices (MediaPipe Hands topology)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks, handedness_label):
    """
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    True if the finger is extended (up), False if curled (down).
    """
    fingers = []

    # Thumb: compare x position of tip vs ip joint, direction depends on
    # which hand (left/right) is in frame, since thumb moves sideways.
    if handedness_label == "Right":
        fingers.append(landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x)
    else:
        fingers.append(landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x)

    # Other four fingers: tip above (smaller y) than pip joint = extended
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers


def classify_gesture(landmarks, handedness_label):
    """Classify a gesture from landmark positions. Returns a label string."""
    thumb, index, middle, ring, pinky = fingers_up(landmarks, handedness_label)
    up_count = sum([thumb, index, middle, ring, pinky])

    # OK sign: thumb tip close to index tip, other three fingers extended
    thumb_index_dist = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    if thumb_index_dist < 0.05 and middle and ring and pinky:
        return "OK"

    # Thumbs up / down: only thumb extended, hand mostly vertical
    if thumb and not index and not middle and not ring and not pinky:
        if landmarks[THUMB_TIP].y < landmarks[WRIST].y:
            return "Thumbs Up"
        else:
            return "Thumbs Down"

    if up_count == 0:
        return "Fist"

    if up_count == 5:
        return "Open Palm"

    if index and middle and not ring and not pinky and not thumb:
        return "Peace / Victory"

    if index and not middle and not ring and not pinky:
        return "Pointing"

    if index and pinky and not middle and not ring:
        return "Rock On"

    if thumb and pinky and not index and not middle and not ring:
        return "Call Me"

    return "Unknown"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip for a natural "mirror" view, convert BGR -> RGB for MediaPipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" / "Right"

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    gesture = classify_gesture(hand_landmarks.landmark, label)

                    # Position text near the wrist of this hand
                    h, w, _ = frame.shape
                    wrist_x = int(hand_landmarks.landmark[WRIST].x * w)
                    wrist_y = int(hand_landmarks.landmark[WRIST].y * h)

                    cv2.putText(
                        frame,
                        f"{label}: {gesture}",
                        (max(wrist_x - 50, 0), max(wrist_y - 20, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Gesture Detection", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
