# -*- coding: utf-8 -*-
"""
Eye Gaze Mouse Control - High Precision & No Calibration Needed
"""

import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk

pyautogui.FAILSAFE = False

# Get accurate screen dimensions
SCREEN_W, SCREEN_H = pyautogui.size()

# ----------------------------------------------------------------------
# Configurable Settings (Adjusted for stability and lower sensitivity)
# ----------------------------------------------------------------------
EAR_THRESHOLD = 0.21            
WINK_GAP_RATIO = 1.15           
BLINK_DOUBLE_WINDOW = 0.7       
CLICK_COOLDOWN = 1.0            
SMOOTHING_WINDOW = 8            # increased smoothing to stop jumping
GAZE_GAIN_X = 2.8               # balanced horizontal sensitivity
GAZE_GAIN_Y = 2.8               # balanced vertical sensitivity
DWELL_TIME = 0.8                
CAM_INDEX = 0                   

# ----------------------------------------------------------------------
# MediaPipe FaceMesh
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def euclidean(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_ear(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4) * 2
    return vertical / horizontal if horizontal != 0 else 0.0


def iris_center(landmarks, iris_idx, w, h):
    xs = [landmarks[i].x * w for i in iris_idx]
    ys = [landmarks[i].y * h for i in iris_idx]
    return np.mean(xs), np.mean(ys)


def get_eye_gaze_ratio(landmarks, eye_idx, iris_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    ix, iy = iris_center(landmarks, iris_idx, w, h)

    eye_left_x = min(p1[0], p4[0])
    eye_right_x = max(p1[0], p4[0])
    eye_top_y = min(p2[1], p3[1])
    eye_bottom_y = max(p5[1], p6[1])

    width = (eye_right_x - eye_left_x) or 1e-6
    height = (eye_bottom_y - eye_top_y) or 1e-6

    # Normalized position of iris inside the eye frame (centered around 0.5)
    hx = (ix - eye_left_x) / width
    hy = (iy - eye_top_y) / height
    return hx, hy


def get_combined_gaze(landmarks, w, h):
    lhx, lhy = get_eye_gaze_ratio(landmarks, LEFT_EYE, LEFT_IRIS, w, h)
    rhx, rhy = get_eye_gaze_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS, w, h)
    return (lhx + rhx) / 2.0, (lhy + rhy) / 2.0


def map_gaze_to_screen(gx, gy):
    # Direct ratio mapping without manual calibration for stable tracking
    # Assuming neutral gaze ratio is roughly 0.5 with standard limits [0.35, 0.65]
    nx = (gx - 0.42) / 0.16
    ny = (gy - 0.38) / 0.24

    nx = 0.5 + (nx - 0.5) * GAZE_GAIN_X
    ny = 0.5 + (ny - 0.5) * GAZE_GAIN_Y

    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    
    screen_x = int(nx * SCREEN_W)
    screen_y = int(ny * SCREEN_H)
    return screen_x, screen_y


class VirtualKeyboard:
    def __init__(self):
        self.root = None
        self.visible = False
        self.buttons = []
        self.hover_key = None
        self.hover_start = 0
        self._lock = threading.Lock()

    def _build(self):
        self.root = tk.Tk()
        self.root.title("Eye Virtual Keyboard")
        self.root.attributes("-topmost", True)

        win_w = int(SCREEN_W * 0.6)
        win_h = int(SCREEN_H * 0.3)
        x = (SCREEN_W - win_w) // 2
        y = SCREEN_H - win_h - 50
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        rows = [
            "1234567890",
            "QWERTYUIOP",
            "ASDFGHJKL",
            "ZXCVBNM",
        ]

        for row in rows:
            frame = tk.Frame(self.root)
            frame.pack(fill="both", expand=True, padx=2, pady=2)
            for ch in row:
                b = tk.Button(frame, text=ch, font=("Arial", 12, "bold"),
                              command=lambda c=ch: self._press(c))
                b.pack(side="left", fill="both", expand=True, padx=1, pady=1)
                self.buttons.append(b)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="both", expand=True, padx=2, pady=2)
        
        for text, cmd in [("Space", " "), ("Backspace", "BACKSPACE"), ("Enter", "ENTER")]:
            b = tk.Button(bottom, text=text, font=("Arial", 11, "bold"),
                          command=lambda c=cmd: self._press(c))
            b.pack(side="left", fill="both", expand=True, padx=1, pady=1)
            self.buttons.append(b)

        self.root.protocol("WM_DELETE_WINDOW", self.hide)

    def _press(self, ch):
        if ch == "BACKSPACE":
            pyautogui.press("backspace")
        elif ch == "ENTER":
            pyautogui.press("enter")
        else:
            pyautogui.typewrite(ch)

    def show(self):
        with self._lock:
            if self.root is None:
                self._build()
            self.root.deiconify()
            self.visible = True

    def hide(self):
        with self._lock:
            if self.root is not None:
                self.root.withdraw()
            self.visible = False

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()

    def update_gui(self):
        if self.root is not None:
            try:
                self.root.update()
            except tk.TclError:
                pass

    def check_dwell_click(self, screen_x, screen_y):
        if not self.visible or self.root is None:
            self.hover_key = None
            return

        found = None
        for b in self.buttons:
            try:
                bx = b.winfo_rootx()
                by = b.winfo_rooty()
                bw = b.winfo_width()
                bh = b.winfo_height()
            except tk.TclError:
                continue
            if bx <= screen_x <= bx + bw and by <= screen_y <= by + bh:
                found = b
                break

        now = time.time()
        if found is not None:
            if found is self.hover_key:
                if now - self.hover_start >= DWELL_TIME:
                    found.invoke()
                    self.hover_key = None
            else:
                self.hover_key = found
                self.hover_start = now
        else:
            self.hover_key = None


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    keyboard = VirtualKeyboard()
    hist_x, hist_y = [], []

    state = {
        "left_closed": False,
        "right_closed": False,
        "both_closed": False,
        "left_blink_times": [],
        "right_blink_times": [],
        "both_blink_times": [],
        "last_left_click": 0,
        "last_right_click": 0,
        "last_kb_toggle": 0,
    }

    print("Started. Press 'q' in the camera window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            gx, gy = get_combined_gaze(lm, w, h)

            hist_x.append(gx)
            hist_y.append(gy)
            if len(hist_x) > SMOOTHING_WINDOW:
                hist_x.pop(0)
                hist_y.pop(0)
            avg_gx = sum(hist_x) / len(hist_x)
            avg_gy = sum(hist_y) / len(hist_y)

            screen_x, screen_y = map_gaze_to_screen(avg_gx, avg_gy)
            pyautogui.moveTo(screen_x, screen_y, duration=0)

            left_ear = get_ear(lm, LEFT_EYE, w, h)
            right_ear = get_ear(lm, RIGHT_EYE, w, h)

            left_closed_now = left_ear < EAR_THRESHOLD
            right_closed_now = right_ear < EAR_THRESHOLD
            both_closed_now = left_closed_now and right_closed_now

            wink_left_now = left_closed_now and right_ear >= EAR_THRESHOLD * WINK_GAP_RATIO
            wink_right_now = right_closed_now and left_ear >= EAR_THRESHOLD * WINK_GAP_RATIO

            now = time.time()

            if wink_left_now and not state["left_closed"]:
                state["left_blink_times"].append(now)
            state["left_closed"] = wink_left_now

            if wink_right_now and not state["right_closed"]:
                state["right_blink_times"].append(now)
            state["right_closed"] = wink_right_now

            if both_closed_now and not state["both_closed"]:
                state["both_blink_times"].append(now)
            state["both_closed"] = both_closed_now

            state["left_blink_times"] = [t for t in state["left_blink_times"] if now - t <= BLINK_DOUBLE_WINDOW]
            state["right_blink_times"] = [t for t in state["right_blink_times"] if now - t <= BLINK_DOUBLE_WINDOW]
            state["both_blink_times"] = [t for t in state["both_blink_times"] if now - t <= BLINK_DOUBLE_WINDOW]

            if len(state["left_blink_times"]) >= 2 and now - state["last_left_click"] > CLICK_COOLDOWN:
                pyautogui.click(button="left")
                state["last_left_click"] = now
                state["left_blink_times"] = []
                print("Left Click")

            if len(state["right_blink_times"]) >= 2 and now - state["last_right_click"] > CLICK_COOLDOWN:
                pyautogui.click(button="right")
                state["last_right_click"] = now
                state["right_blink_times"] = []
                print("Right Click")

            if len(state["both_blink_times"]) >= 2 and now - state["last_kb_toggle"] > CLICK_COOLDOWN:
                keyboard.toggle()
                state["last_kb_toggle"] = now
                state["both_blink_times"] = []
                print("Keyboard Toggle")

            keyboard.check_dwell_click(screen_x, screen_y)

            cv2.putText(frame, f"L_EAR:{left_ear:.2f} R_EAR:{right_ear:.2f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        keyboard.update_gui()

        cv2.imshow("Eye Mouse Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if keyboard.root is not None:
        keyboard.root.destroy()


if __name__ == "__main__":
    main()