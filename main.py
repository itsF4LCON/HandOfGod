import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import pyautogui
import math
import time
from pynput.mouse import Controller, Button

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Done!")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

cap = cv2.VideoCapture(1)
cam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

MARGIN = 0.35
SMOOTHING = 0.35
CLICK_COOLDOWN = 0.4

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=lambda result, img, timestamp: globals().update({"latest_result": result})
)

latest_result = None
landmarker = vision.HandLandmarker.create_from_options(options)

is_dragging = False
pinch_start_time = 0
DRAG_THRESHOLD = 0.5  
mouse = Controller()

smooth_x, smooth_y = 0, 0
last_click = 0
prev_time = 0
gesture_label = "none"

last_hand_y = None
SCROLL_SENSITIVITY = 25.5

def finger_up(points):
    fingers = []
    fingers.append(1 if points[4][0] > points[3][0] else 0)
    for tip, mid in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(1 if points[tip][1] < points[mid][1] else 0)
    return fingers

def get_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 0.001))
    prev_time = curr_time

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if latest_result and latest_result.hand_landmarks:
        for hand in latest_result.hand_landmarks:
            h, w, _ = frame.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            index_tip = points[8]
            x_min = cam_width * MARGIN
            x_max = cam_width * (1 - MARGIN)
            y_min = cam_height * MARGIN
            y_max = cam_height * (1 - MARGIN)
            screen_x = int((index_tip[0] - x_min) / (x_max - x_min) * screen_w)
            screen_y = int((index_tip[1] - y_min) / (y_max - y_min) * screen_h)
            screen_x = max(0, min(screen_w, screen_x))
            screen_y = max(0, min(screen_h, screen_y))

        
            pinch_dist = get_distance(points[4], points[8])


            fingers = finger_up(points)
            is_fist = sum(fingers[1:]) == 0


            if pinch_dist < 40:
                gesture_label = "DRAG" if is_dragging else "HOLD"
                last_hand_y = None  # reset scroll

                if pinch_start_time == 0:
                    pinch_start_time = time.time()

                hold_duration = time.time() - pinch_start_time

                if hold_duration > DRAG_THRESHOLD and not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True

                if is_dragging:
                    smooth_x = smooth_x * 0.6 + screen_x * 0.4
                    smooth_y = smooth_y * 0.6 + screen_y * 0.4
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))

            elif is_fist:
                gesture_label = "SCROLL"
                current_y = points[9][1]
                if last_hand_y is not None:
                    delta_y = last_hand_y - current_y
                    if abs(delta_y) > 2:
                        mouse.scroll(0, delta_y / SCROLL_SENSITIVITY)
                last_hand_y = current_y

            else:
                last_hand_y = None
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                elif pinch_start_time != 0:
                    hold_duration = time.time() - pinch_start_time
                    if hold_duration < DRAG_THRESHOLD:
                        pyautogui.click()

                pinch_start_time = 0
                gesture_label = "MOVE"

                smooth_x = smooth_x * SMOOTHING + screen_x * (1 - SMOOTHING)
                smooth_y = smooth_y * SMOOTHING + screen_y * (1 - SMOOTHING)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

            for pt in points:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255, 255, 255), 2)


    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Air Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()