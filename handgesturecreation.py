import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_FILE = "custom_gestures.json"
SAMPLES_PER_RECORDING = 100     # how many frames to capture per recording session
MATCH_THRESHOLD = 0.50         # lower = stricter matching, higher = more lenient
CAPTURE_DELAY = 0.03            # seconds between captured samples

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WRIST = 0
MIDDLE_MCP = 9


# ---------------------------------------------------------------------
# Landmark normalization
# ---------------------------------------------------------------------
def normalize_landmarks(landmarks):
    """
    Convert MediaPipe landmarks into a scale- and position-invariant
    vector so the same gesture matches regardless of hand distance
    from the camera or position in frame.

    Returns a flat numpy array of length 63 (21 points x, y, z).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Translate so the wrist is the origin
    pts -= pts[WRIST]

    # Scale so that wrist-to-middle_mcp distance = 1 (roughly hand size)
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.flatten()


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------
def classify(vector, gestures):
    best_name = "Unknown"
    best_dist = float("inf")

    for name, samples in gestures.items():
        mean_sample = np.mean(samples, axis=0)
        dist = np.linalg.norm(vector - mean_sample)

        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist > MATCH_THRESHOLD:
        return "Unknown", best_dist

    return best_name, best_dist


# ---------------------------------------------------------------------
# Recording flow
# ---------------------------------------------------------------------
def record_gesture(cap, hands, gestures, existing_name=None):
    """
    Prompts for a name (unless adding to an existing one), then captures
    SAMPLES_PER_RECORDING frames of hand landmarks while showing a live
    preview so the user can hold the pose steady.
    """
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
            progress = len(captured)
        else:
            progress = len(captured)

        cv2.putText(frame, f"Recording '{name}': {progress}/{SAMPLES_PER_RECORDING}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hand Gesture Detection", frame)
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


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    gestures = load_gestures()
    print(f"Loaded {len(gestures)} saved gesture(s): {', '.join(gestures.keys()) or 'none yet'}")
    print("\nControls: n = new gesture | a = add samples | d = delete gesture | q = quit\n")

    cap = cv2.VideoCapture(0)

    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

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

            label = "No hand detected"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                vector = normalize_landmarks(hand_landmarks.landmark)
                name, dist = classify(vector, gestures)
                label = f"{name} ({dist:.2f})" if gestures else "No gestures saved yet"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "n=new  a=add  d=delete  q=quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Detection", frame)
            cv2.imshow("Hand Gesture Detection", frame)
            key = cv2.waitKeyEx(1)

            if key != -1:
             print("Key:", key)

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