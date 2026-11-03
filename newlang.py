import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
import time
import threading
import tempfile

CONFIG_FILE = "config.json"
GESTURES_FILE = "custom_gestures.json"

SAMPLES_PER_RECORDING = 100
MATCH_THRESHOLD = 0.50
CAPTURE_DELAY = 0.03
COOLDOWN_SECONDS = 5
ALERT_DISPLAY_SECONDS = 5

LANGUAGE_HELP = """
Common language codes:
  en = English      hi = Hindi        ta = Tamil       te = Telugu
  kn = Kannada      ml = Malayalam    mr = Marathi     bn = Bengali
  gu = Gujarati     pa = Punjabi      ur = Urdu        es = Spanish
  fr = French       de = German       ar = Arabic      zh-CN = Chinese
  ja = Japanese     ko = Korean       ru = Russian     pt = Portuguese
"""

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# =======================================================================
# Custom gesture storage / recognition (no built-in poses anywhere)
# =======================================================================
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()


def load_gestures():
    if os.path.exists(GESTURES_FILE):
        with open(GESTURES_FILE, "r") as f:
            raw = json.load(f)
        return {name: [np.array(s, dtype=np.float32) for s in samples]
                for name, samples in raw.items()}
    return {}


def save_gestures(gestures):
    serializable = {name: [s.tolist() for s in samples]
                     for name, samples in gestures.items()}
    with open(GESTURES_FILE, "w") as f:
        json.dump(serializable, f)


def classify_custom_gesture(vector, gestures):
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
        cv2.imshow("Custom Gesture Speech", frame)
        cv2.waitKey(1)
        time.sleep(CAPTURE_DELAY)

    if name in gestures:
        gestures[name].extend(captured)
    else:
        gestures[name] = captured
    save_gestures(gestures)
    print(f"Saved {len(captured)} samples for gesture '{name}'. "
          f"Total samples now: {len(gestures[name])}.")
    return name


def delete_gesture(gestures, config):
    if not gestures:
        print("\nNo saved gestures to delete.")
        return
    print("\nSaved gestures:", ", ".join(gestures.keys()))
    name = input("Type the exact name to delete: ").strip()
    if name in gestures:
        del gestures[name]
        save_gestures(gestures)
        config.get("messages", {}).pop(name, None)
        save_config(config)
        print(f"Deleted '{name}'.")
    else:
        print("No gesture with that name found.")


# =======================================================================
# CONFIG (languages + per-gesture messages)
# =======================================================================
def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return None


def prompt_message_for_gesture(config, gesture_name):
    """Ask the user for the sentence to speak for a given gesture name."""
    current = config.get("messages", {}).get(gesture_name, "")
    prompt = f"Sentence to speak for gesture '{gesture_name}'"
    prompt += f" (current: '{current}')" if current else ""
    prompt += ": "
    text = input(prompt).strip()
    if text:
        config.setdefault("messages", {})[gesture_name] = text
    elif current:
        config.setdefault("messages", {})[gesture_name] = current
    else:
        print(f"  (no message set for '{gesture_name}' - it will stay silent for now)")


def run_setup_wizard(existing_config, gestures):
    print("\n" + "=" * 60)
    print("  SETUP WIZARD")
    print("=" * 60)
    config = existing_config or {}
    config["languages"] = ["en"]
    

    

    if gestures:
        print(f"\nFound {len(gestures)} saved custom gesture(s): {', '.join(gestures.keys())}")
        print("Enter a sentence to speak for each one (press Enter to skip a gesture):\n")
        for name in gestures.keys():
            prompt_message_for_gesture(config, name)
    else:
        print("\nNo custom gestures recorded yet. Press 'n' while the app is running "
              "to record your first one, then 'm' to give it a spoken sentence.")

    save_config(config)
    print(f"\nSetup complete! Saved to {CONFIG_FILE}.\n")
    return config


# =======================================================================
# TRANSLATION + MULTI-LANGUAGE VOICE (background thread)
# =======================================================================
def translate_message(message, target_lang):
    if target_lang == "en":
        return message
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target_lang).translate(message)
    except Exception as e:
        print(f"[Translation error for '{target_lang}']: {e}. Speaking English instead.")
        return message


def speak_multilang_async(message, languages):
    def _speak_all():
        from gtts import gTTS
        from playsound import playsound

        for lang in languages:
            try:
                text = translate_message(message, lang)
                print(f"  Speaking ({lang}): {text}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp_path = tmp.name

                tts = gTTS(text=text, lang=lang)
                tts.save(tmp_path)
                playsound(tmp_path)
                os.remove(tmp_path)
                time.sleep(0.3)
            except Exception as e:
                print(f"[Voice error for '{lang}']: {e}")

    threading.Thread(target=_speak_all, daemon=True).start()

def choose_language():
    print("\n==============================")
    print("      SELECT LANGUAGE")
    print("==============================")
    print("1. English")
    print("2. Tamil")
    print("3. Hindi")
    print("4. Telugu")
    print("5. Kannada")
    print("6. Malayalam")
    print("7. Marathi")
    print("8. Bengali")
    print("9. Gujarati")
    print("10. Punjabi")

    language_map = {
        "1": ("English", "en"),
        "2": ("Tamil", "ta"),
        "3": ("Hindi", "hi"),
        "4": ("Telugu", "te"),
        "5": ("Kannada", "kn"),
        "6": ("Malayalam", "ml"),
        "7": ("Marathi", "mr"),
        "8": ("Bengali", "bn"),
        "9": ("Gujarati", "gu"),
        "10": ("Punjabi", "pa")
    }

    while True:
        choice = input("\nSelect language (1-10): ").strip()

        if choice in language_map:
            name, code = language_map[choice]
            print(f"\nSelected Language : {name}\n")
            return code

        print("Invalid choice. Try again.") 

# =======================================================================
# Startup menu - ask n/m/a/d BEFORE asking for language
# =======================================================================
def startup_menu(cap, hands, gestures, config):
    """
    Runs before language selection. Lets the user create/edit/delete
    gestures and messages first. Keeps looping until they choose to
    continue (blank input or 'c').
    """
    while True:
        print("\n==============================")
        print("   BEFORE WE START")
        print("==============================")
        print("n = record a NEW custom gesture")
        print("m = set/change a gesture's spoken message")
        print("a = add more samples to an existing gesture")
        print("d = delete a gesture")
        print("c = continue to language selection (or just press Enter)")

        choice = input("\nChoose an option (n/m/a/d/c): ").strip().lower()

        if choice == "" or choice == "c":
            break

        elif choice == "n":
            new_name = record_gesture(cap, hands, gestures)
            if new_name:
                prompt_message_for_gesture(config, new_name)
                save_config(config)

        elif choice == "m":
            if not gestures:
                print("\nNo gestures saved yet - choose 'n' to create one first.")
            else:
                messages = config.get("messages", {})
                print("\nSaved gestures and their current messages:")
                for gname in gestures.keys():
                    current_msg = messages.get(gname, "(no message set)")
                    print(f"  {gname}: {current_msg}")
                name = input("\nType the exact gesture name to set/change its message: ").strip()
                if name in gestures:
                    prompt_message_for_gesture(config, name)
                    save_config(config)
                else:
                    print("No gesture with that name found.")

        elif choice == "a":
            if not gestures:
                print("\nNo gestures saved yet - choose 'n' to create one first.")
            else:
                print("\nSaved gestures:", ", ".join(gestures.keys()))
                name = input("Type the exact name to add samples to: ").strip()
                if name in gestures:
                    record_gesture(cap, hands, gestures, existing_name=name)
                else:
                    print("No gesture with that name found.")

        elif choice == "d":
            delete_gesture(gestures, config)

        else:
            print("Not a recognized option, try again.")


# =======================================================================
# Main loop
# =======================================================================
def main():
    gestures = load_gestures()
    config = load_config()
    if config is None:
        config = run_setup_wizard(config, gestures)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    last_trigger_time = 0
    alert_until = 0
    last_spoken_gesture = None

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        # Ask n/m/a/d FIRST, then language, before the live preview loop
        startup_menu(cap, hands, gestures, config)

        selected_language = choose_language()
        languages = [selected_language]
        messages = config.get("messages", {})

        print(f"\nLoaded {len(gestures)} custom gesture(s): {', '.join(gestures.keys()) or 'none yet'}")
        print(f"Languages: {', '.join(languages)}")
        print("Controls: n=new gesture | a=add samples | d=delete gesture | "
              "m=set message | t=test | q=quit\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected_name = "Unknown"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                vector = normalize_landmarks(hand_landmarks.landmark)
                detected_name, dist = classify_custom_gesture(vector, gestures)

                cv2.putText(frame, f"Gesture: {detected_name} ({dist:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            now = time.time()
            has_message = detected_name in messages and messages[detected_name].strip()

            # Only speak again for the same gesture after cooldown, or
            # immediately if it's a DIFFERENT recognized gesture than last time
            should_trigger = (
                detected_name != "Unknown" and has_message and
                (detected_name != last_spoken_gesture or (now - last_trigger_time) > COOLDOWN_SECONDS)
            )

            if now < alert_until:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 120, 0), -1)
                cv2.putText(frame, "SPEAKING...", (30, h // 2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.putText(frame, "n=new  a=add  d=delete  m=message  t=test  q=quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Custom Gesture Speech", frame)
            key = cv2.waitKey(5) & 0xFF

            if should_trigger:
                last_trigger_time = now
                last_spoken_gesture = detected_name
                alert_until = now + ALERT_DISPLAY_SECONDS
                print(f"\n[TRIGGER] Gesture '{detected_name}' detected -> speaking in "
                      f"{len(languages)} language(s).")
                speak_multilang_async(messages[detected_name], languages)

            if key == ord("q"):
                break
            elif key == ord("n"):
                new_name = record_gesture(cap, hands, gestures)
                if new_name:
                    prompt_message_for_gesture(config, new_name)
                    save_config(config)
                    messages = config.get("messages", {})
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
                delete_gesture(gestures, config)
                messages = config.get("messages", {})
            elif key == ord("m"):
                if not gestures:
                    print("\nNo gestures saved yet - press 'n' to create one first.")
                else:
                    print("\nSaved gestures and their current messages:")
                    for gname in gestures.keys():
                        current_msg = messages.get(gname, "(no message set)")
                        print(f"  {gname}: {current_msg}")
                    name = input("\nType the exact gesture name to set/change its message: ").strip()
                    if name in gestures:
                        prompt_message_for_gesture(config, name)
                        save_config(config)
                        messages = config.get("messages", {})
                    else:
                        print("No gesture with that name found.")
            elif key == ord("t"):
                if not messages:
                    print("\nNo messages configured yet - press 'm' to set one first.")
                else:
                    print("\nGestures with messages:", ", ".join(messages.keys()))
                    name = input("Type the exact gesture name to test: ").strip()
                    if name in messages:
                        alert_until = time.time() + ALERT_DISPLAY_SECONDS
                        speak_multilang_async(messages[name], languages)
                    else:
                        print("No message configured for that gesture.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()