import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import math
import threading
import pyttsx3

# =======================================================================
# CONFIG - change these to customize behavior
# =======================================================================
DATA_FILE = "custom_gestures.json"
SAMPLES_PER_RECORDING = 25       # frames captured per recording session
MATCH_THRESHOLD = 0.20           # lower = stricter custom-gesture matching
CAPTURE_DELAY = 0.03             # seconds between captured samples

TRIGGER_GESTURE = "help"         # matched case-insensitively
SPOKEN_MESSAGE = "Help! I need assistance."
COOLDOWN_SECONDS = 4             # min seconds between repeated voice alerts
ALERT_DISPLAY_SECONDS = 2        # how long the on-screen banner stays up

# =======================================================================
# MediaPipe setup
# =======================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


# =======================================================================
# PART 1: Built-in gesture classification (rule-based on finger state)
# =======================================================================
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks, handedness_label):
    fingers = []
    if handedness_label == "Right":
        fingers.append(landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x)
    else:
        fingers.append(landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x)
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    return fingers

def classify_builtin_gesture(landmarks, handedness_label):
    thumb, index, middle, ring, pinky = fingers_up(landmarks, handedness_label)
    up_count = sum([thumb, index, middle, ring, pinky])

    thumb_index_dist = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    if thumb_index_dist < 0.05 and middle and ring and pinky:
        return "OK"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up" if landmarks[THUMB_TIP].y < landmarks[WRIST].y else "Thumbs Down"
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



# =======================================================================
# PART 2: Custom gesture recording / recognition
# =======================================================================
def normalize_landmarks(landmarks):
    """Scale- and position-invariant vector representing the hand shape."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()


def load_gestures():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            raw = json.load(f)
        return {name: [np.array(s, dtype=np.float32) for s in samples]
                for name, samples in raw.items()}
    return {}


def save_gestures(gestures):
    serializable = {name: [s.tolist() for s in samples]
                     for name, samples in gestures.items()}
    with open(DATA_FILE, "w") as f:
        json.dump(serializable, f)


def classify_custom_gesture(vector, gestures):
    """Returns (name, distance) for the closest saved custom gesture, or
    ("Unknown", threshold) if nothing is close enough."""
    best_name, best_dist = "Unknown", MATCH_THRESHOLD
    for name, samples in gestures.items():
        for sample in samples:
            dist = np.linalg.norm(vector - sample)
            if dist < best_dist:
                best_name, best_dist = name, dist
    return best_name, best_dist


def record_gesture(cap, hands, gestures, existing_name=None):
    if existing_name:
        name = existing_name
        print(f"\nAdding more samples to existing gesture '{name}'.")
    else:
        name = input("\nType a name for this new gesture and press Enter: ").strip()
        if not name:
            print("No name entered, cancelling.")
            return

    print(f"Get ready to hold the gesture for '{name}'...")
    for i in range(3, 0, -1):
        print(f"  starting in {i}...")
        time.sleep(0.6)
    print("Capturing now - hold your pose steady!")

    captured = []
    while len(captured) < SAMPLES_PER_RECORDING:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            vector = normalize_landmarks(hand_landmarks.landmark)
            captured.append(vector)

        cv2.putText(frame, f"Recording '{name}': {len(captured)}/{SAMPLES_PER_RECORDING}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("All-in-One Hand Gesture System", frame)
        cv2.waitKey(1)
        time.sleep(CAPTURE_DELAY)

    if name in gestures:
        gestures[name].extend(captured)
    else:
        gestures[name] = captured

    save_gestures(gestures)
    print(f"Saved {len(captured)} samples for gesture '{name}'. "
          f"Total samples now: {len(gestures[name])}.")


def delete_gesture(gestures):
    if not gestures:
        print("\nNo saved gestures to delete.")
        return
    print("\nSaved gestures:", ", ".join(gestures.keys()))
    name = input("Type the exact name to delete: ").strip()
    if name in gestures:
        del gestures[name]
        save_gestures(gestures)
        print(f"Deleted '{name}'.")
    else:
        print("No gesture with that name found.")


# =======================================================================
# PART 3: Voice alert
# =======================================================================
def speak_async(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


# =======================================================================
# Main loop
# =======================================================================
def main():
    gestures = load_gestures()
    print(f"Loaded {len(gestures)} custom gesture(s): {', '.join(gestures.keys()) or 'none yet'}")
    print("Controls: n = new gesture | a = add samples | d = delete gesture | q = quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    last_trigger_time = 0
    alert_until = 0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            custom_name = "Unknown"
            builtin_name = "No hand detected"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # Built-in rule-based gesture
                builtin_name = classify_builtin_gesture(hand_landmarks.landmark, handedness_label)

                # Custom trained gesture
                vector = normalize_landmarks(hand_landmarks.landmark)
                custom_name, custom_dist = classify_custom_gesture(vector, gestures)

                cv2.putText(frame, f"Built-in: {builtin_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Custom: {custom_name} ({custom_dist:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            now = time.time()

            # Trigger voice alert if the custom gesture matches TRIGGER_GESTURE
            # (case-insensitive) and the cooldown period has passed
            if custom_name.lower() == TRIGGER_GESTURE.lower() and (now - last_trigger_time) > COOLDOWN_SECONDS:
                last_trigger_time = now
                alert_until = now + ALERT_DISPLAY_SECONDS
                speak_async(SPOKEN_MESSAGE)
                print(f"[ALERT] '{TRIGGER_GESTURE}' gesture detected -> speaking message.")

            if now < alert_until:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 255), -1)
                cv2.putText(frame, "HELP DETECTED", (30, h // 2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.putText(frame, "n=new  a=add  d=delete  q=quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("All-in-One Hand Gesture System", frame)
            key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                record_gesture(cap, hands, gestures)
            elif key == ord("a"):
                if not gestures:
                    print("\nNo gestures saved yet - press 'n' to create one first.")
                else:
                    print("\nSaved gestures:", ", ".join(gestures.keys()))
                    name = input("Type the exact name to add samples to: ").strip()
                    if name in gestures:
                        record_gesture(cap, hands, gestures, existing_name=name)
                    else:
                        print("No gesture with that name found.")
            elif key == ord("d"):
                delete_gesture(gestures)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()