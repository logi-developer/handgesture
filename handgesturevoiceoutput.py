import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import pyttsx3
import threading
 
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_FILE = "custom_gestures.json"
MATCH_THRESHOLD = 0.50
COOLDOWN_SECONDS = 5
ALERT_DISPLAY_SECONDS = 10

# Gesture name -> Voice message
GESTURE_MESSAGES = {
    "HELP": "Help! I need assistance.",
    "EMERGENCY": "Emergency! Please help me.",
    "WATER": "I need water.",
    "TABLETS": "I need my tablets."
}
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
 
WRIST = 0
MIDDLE_MCP = 9
 
 
# ---------------------------------------------------------------------
# Text-to-speech (runs in a background thread so video doesn't freeze)
# ---------------------------------------------------------------------
def speak_async(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()
 
 
# ---------------------------------------------------------------------
# Landmark normalization (must match custom_gesture_creator.py)
# ---------------------------------------------------------------------
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()
 
 
def load_gestures():
    if not os.path.exists(DATA_FILE):
        print(f"Warning: '{DATA_FILE}' not found. Record gestures first "
              f"with custom_gesture_creator.py.")
        return {}
    with open(DATA_FILE, "r") as f:
        raw = json.load(f)
    return {name: [np.array(s, dtype=np.float32) for s in samples]
            for name, samples in raw.items()}
 
 
def classify(vector, gestures):
    best_name, best_dist = "Unknown", MATCH_THRESHOLD
    for name, samples in gestures.items():
        for sample in samples:
            dist = np.linalg.norm(vector - sample)
            if dist < best_dist:
                best_name, best_dist = name, dist
    return best_name, best_dist
 
 
# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    gestures = load_gestures()
    for gesture in GESTURE_MESSAGES:
     if gesture not in gestures:
        print(f"Warning: Gesture '{gesture}' has not been recorded yet.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return
 
    last_trigger_time = 0
    alert_until = 0
    alert_gesture = ""
 
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
 
            detected_name = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                vector = normalize_landmarks(hand_landmarks.landmark)
                name, dist = classify(vector, gestures)
                detected_name = name
 
                cv2.putText(frame, f"Detected: {name} ({dist:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
 
            now = time.time()
 
            # Trigger the alert if the target gesture is seen and cooldown has passed
            if detected_name in GESTURE_MESSAGES and (now - last_trigger_time) > COOLDOWN_SECONDS:
             last_trigger_time = now
             alert_until = now + ALERT_DISPLAY_SECONDS
             alert_gesture = detected_name

             speak_async(GESTURE_MESSAGES[detected_name])

             print(f"[ALERT] {detected_name}")
 
            # Draw the on-screen banner while within its display window
            if now < alert_until:
                h, w, _ = frame.shape
                cv2.rectangle(
                    frame, 
                    (0, h // 2 - 40), 
                    (w, h // 2 + 40), 
                    (0, 0, 255), 
                    -1,
                )
                cv2.putText(
                    frame, 
                    f"{alert_gesture} DETECTED", 
                    (30, h // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.4, 
                    (255, 255, 255), 
                    3,
                )
 
            cv2.putText(frame, "Press 'q' to quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
 
            cv2.imshow("Gesture Voice Alert", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()