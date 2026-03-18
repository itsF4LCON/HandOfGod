# Air Mouse (Hand Tracking Mouse Control)

Control your mouse using just your hand via your webcam.
This project uses MediaPipe hand tracking to move the cursor, click, drag, and scroll based on gestures.

## Features

* Move cursor with index finger
* Click by pinching (thumb + index)
* Drag by holding pinch
* Scroll using fist gesture
* Smooth mouse movement
* Real-time hand tracking with OpenCV

## Requirements

Install dependencies:

pip install opencv-python mediapipe pyautogui pynput

## How to Run

python main.py

The script will automatically download the required hand tracking model if it's not found.

## Controls

* Move Mouse → Move your index finger
* Click → Quick pinch (thumb + index)
* Drag → Hold pinch
* Scroll → Make a fist and move hand up/down

## Notes

* Make sure your webcam is working
* Lighting affects detection accuracy
* Press ESC to exit

## File Structure

.
├── main.py
├── hand_landmarker.task (auto-downloaded)

## Tech Used

* OpenCV
* MediaPipe
* PyAutoGUI
* Pynput

## License

Free to use and modify
