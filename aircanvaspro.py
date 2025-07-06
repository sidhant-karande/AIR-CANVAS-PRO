import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tkinter as tk
from tkinter import messagebox, filedialog
import sys
import os
import json
import time

class AirCanvasApp:
    def __init__(self):
        # Initialize drawing variables
        self.bpoints = [deque(maxlen=2000)]
        self.gpoints = [deque(maxlen=2000)]
        self.rpoints = [deque(maxlen=2000)]
        self.ypoints = [deque(maxlen=2000)]
        self.color_indices = {"blue": 0, "green": 1, "red": 2, "yellow": 3}
        self.colorIndex = 0
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.min_brush_size = 5
        self.max_brush_size = 30
        self.brushSize = 8
        self.pointerSize = 10

        # Canvas setup
        self.canvas_width = 636
        self.canvas_height = 471
        self.menu_height = 67
        self.paintWindow = np.zeros((self.canvas_height, self.canvas_width, 3)) + 255
        self._setup_menu_area()

        # MediaPipe Hands setup
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

        # Webcam
        self.cap = None
        self.is_running = False

        # Gesture recognition
        self.last_save_time = 0
        self.save_cooldown = 1.5
        self.two_fingers_start_time = None
        self.two_fingers_hold_duration = 0.5
        self.thumbs_down_start_time = None
        self.thumbs_down_hold_duration = 0.5
        self.drawing_mode = False
        self.saving_enabled = True
        self.clear_enabled = True

        # Create main window
        self.root = tk.Tk()
        self._setup_gui()

    def _setup_menu_area(self):
        """Setup the menu area on the paint window"""
        button_config = [
            ("CLEAR", (40, 1), (140, 65), (0, 0, 0)),
            ("BLUE", (160, 1), (255, 65), (255, 0, 0)),
            ("GREEN", (275, 1), (370, 65), (0, 255, 0)),
            ("RED", (390, 1), (485, 65), (0, 0, 255)),
            ("YELLOW", (505, 1), (600, 65), (0, 255, 255)),
        ]
        for text, start, end, color in button_config:
            cv2.rectangle(self.paintWindow, start, end, (0, 0, 0), 2)
            text_color = (0, 0, 0) if text == "CLEAR" else (0, 0, 0)
            cv2.putText(self.paintWindow, text, (start[0] + 9, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

    def _setup_gui(self):
        """Setup the main GUI window"""
        self.root.title("Air Canvas - Main Menu")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)

        bg_color = "#2c3e50"
        button_color = "#3498db"
        button_hover = "#2980b9"
        text_color = "#ecf0f1"

        self.root.configure(bg=bg_color)

        title_label = tk.Label(self.root, text="Air Canvas", font=("Helvetica", 32, "bold"),
                               bg=bg_color, fg=text_color, pady=20)
        title_label.pack()

        subtitle_label = tk.Label(self.root, text="Draw in the air with your fingers!",
                                  font=("Helvetica", 14), bg=bg_color, fg=text_color)
        subtitle_label.pack(pady=(0, 40))

        button_frame = tk.Frame(self.root, bg=bg_color)
        button_frame.pack()

        start_button = tk.Button(button_frame, text="Start Drawing", font=("Helvetica", 14),
                                 bg=button_color, fg=text_color, activebackground=button_hover,
                                 activeforeground=text_color, relief="flat",
                                 command=self.start_program, width=15, height=2)
        start_button.pack(pady=10)

        load_button = tk.Button(button_frame, text="Load Drawing", font=("Helvetica", 14),
                                bg=button_color, fg=text_color, activebackground=button_hover,
                                activeforeground=text_color, relief="flat",
                                command=self.load_drawing, width=15, height=2)
        load_button.pack(pady=10)

        how_to_button = tk.Button(button_frame, text="How to Use", font=("Helvetica", 14),
                                  bg=button_color, fg=text_color, activebackground=button_hover,
                                  activeforeground=text_color, relief="flat",
                                  command=self.show_instructions, width=15, height=2)
        how_to_button.pack(pady=10)

        exit_button = tk.Button(button_frame, text="Exit", font=("Helvetica", 14),
                                bg="#e74c3c", fg=text_color, activebackground="#c0392b",
                                activeforeground=text_color, relief="flat",
                                command=self.exit_program, width=15, height=2)
        exit_button.pack(pady=10)

    def clear_canvas(self):
        """Clear the drawing canvas"""
        if not self.clear_enabled:
            return
        self.bpoints = [deque(maxlen=2000)]
        self.gpoints = [deque(maxlen=2000)]
        self.rpoints = [deque(maxlen=2000)]
        self.ypoints = [deque(maxlen=2000)]
        self.paintWindow[self.menu_height:, :, :] = 255
        self._setup_menu_area()

    def show_instructions(self):
        """Show instructions dialog"""
        instructions = """
        Air Canvas - How to Use:

        1. Show your hand in front of the camera
        2. Use your index finger to point and draw
        3. To draw, pinch your thumb and index finger together
        4. To change color, point at the color buttons at the top
        5. To clear the canvas, point at the CLEAR button
        6. Adjust brush size by spreading/closing index and middle fingers
        7. Press 'q' in the window to return to main menu

        Gestures:
        - Show Two Fingers: Save the current drawing
        - Hold Thumbs Down: Clear the canvas

        Color Buttons:
        - BLUE: First color option
        - GREEN: Second color option
        - RED: Third color option
        - YELLOW: Fourth color option

        Menu Options:
        - Load Drawing: Load a previously saved drawing

        Enjoy drawing in the air!
        """
        messagebox.showinfo("How to Use Air Canvas", instructions)

    def _calculate_finger_distance(self, landmarks, frame_shape):
        """Calculate distance between index and middle finger tips"""
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]

        index_pos = (int(index_tip.x * frame_shape[1]), int(index_tip.y * frame_shape[0]))
        middle_pos = (int(middle_tip.x * frame_shape[1]), int(middle_tip.y * frame_shape[0]))

        distance = ((middle_pos[0] - index_pos[0])**2 + (middle_pos[1] - index_pos[1])**2)**0.5
        return distance, index_pos, middle_pos

    def _update_brush_size(self, distance):
        """Update brush size based on finger distance"""
        min_distance = 20
        max_distance = 120

        distance = max(min_distance, min(distance, max_distance))
        normalized = (distance - min_distance) / (max_distance - min_distance)
        self.brushSize = int(self.min_brush_size + normalized * (self.max_brush_size - self.min_brush_size))

    def _detect_two_fingers(self, landmarks):
        """Detect if two fingers (index and middle) are extended"""
        index_finger_tip = landmarks.landmark[8]
        middle_finger_tip = landmarks.landmark[12]
        ring_finger_tip = landmarks.landmark[16]
        pinky_finger_tip = landmarks.landmark[20]
        thumb_tip = landmarks.landmark[4]
        finger_base = landmarks.landmark[5]

        index_above_base = index_finger_tip.y < finger_base.y
        middle_above_base = middle_finger_tip.y < finger_base.y
        ring_below_base = ring_finger_tip.y > finger_base.y
        pinky_below_base = pinky_finger_tip.y > finger_base.y
        thumb_below_base = thumb_tip.y > finger_base.y

        return (index_above_base and middle_above_base and ring_below_base
                and pinky_below_base and thumb_below_base)

    def _detect_thumbs_down(self, landmarks):
        """Detect if the thumb is clearly pointing downwards"""
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        return thumb_tip.y > thumb_mcp.y + 0.05

    def save_drawing(self):
        """Save the current drawing to a file"""
        if not self.saving_enabled:
            return
        current_time = time.time()
        if current_time - self.last_save_time < self.save_cooldown:
            return

        self.last_save_time = current_time

        save_data = {
            'bpoints': [list(p) for p in self.bpoints if p],
            'gpoints': [list(p) for p in self.gpoints if p],
            'rpoints': [list(p) for p in self.rpoints if p],
            'ypoints': [list(p) for p in self.ypoints if p],
            'colorIndex': self.colorIndex,
            'brushSize': self.brushSize
        }

        os.makedirs('saves', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"saves/drawing_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f)
            self._show_temp_message("Drawing saved!", (int(self.canvas_width * 0.3), 50), 1.5)
            print(f"Drawing saved to {filename}")
        except Exception as e:
            print(f"Error saving drawing: {e}")

    def load_drawing(self):
        """Load a drawing from file"""
        if self.is_running:
            messagebox.showwarning("Warning", "Please exit drawing mode first")
            return

        filename = filedialog.askopenfilename(
            initialdir="saves",
            title="Select drawing file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )

        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.bpoints = [deque(p, maxlen=2000) for p in data.get('bpoints', [])]
            self.gpoints = [deque(p, maxlen=2000) for p in data.get('gpoints', [])]
            self.rpoints = [deque(p, maxlen=2000) for p in data.get('rpoints', [])]
            self.ypoints = [deque(p, maxlen=2000) for p in data.get('ypoints', [])]
            self.colorIndex = data.get('colorIndex', 0)
            self.brushSize = data.get('brushSize', 8)

            self.paintWindow = np.zeros((self.canvas_height, self.canvas_width, 3)) + 255
            self._setup_menu_area()

            points_data = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
            for i, color_points in enumerate(points_data):
                for stroke in color_points:
                    for j in range(1, len(stroke)):
                        if stroke[j - 1] is not None and stroke[j] is not None:
                            cv2.line(self.paintWindow, stroke[j - 1], stroke[j], self.colors[i], self.brushSize)

            messagebox.showinfo("Success", "Drawing loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load drawing: {e}")

    def _show_temp_message(self, message, position, duration):
        """Show a temporary message on the canvas"""
        temp_window = self.paintWindow.copy()
        cv2.putText(temp_window, message, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Paint", temp_window)
        cv2.waitKey(int(duration * 1000))
        cv2.imshow("Paint", self.paintWindow)

    def start_program(self):
        """Start the air canvas program"""
        if self.is_running:
            return
        self.is_running = True
        self.root.withdraw()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            self.root.deiconify()
            self.is_running = False
            return

        cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]

            # Draw menu on the camera frame
            frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
            cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
            cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
            cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
            cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
            cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            result = self.hands.process(framergb)
            center = None
            current_two_fingers = False
            current_thumbs_down = False

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Get finger positions
                    index_finger = hand_landmarks.landmark[8]
                    center = (int(index_finger.x * frame_width), int(index_finger.y * frame_height))
                    thumb = hand_landmarks.landmark[4]
                    thumb_pos = (int(thumb.x * frame_width), int(thumb.y * frame_height))

                    # Calculate brush size
                    finger_distance, index_pos, middle_pos = self._calculate_finger_distance(hand_landmarks, frame.shape)
                    self._update_brush_size(finger_distance)

                    # Draw landmarks and connections
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                    # Draw finger distance line
                    cv2.line(frame, index_pos, middle_pos, (255, 255, 255), 2)
                    cv2.circle(frame, center, self.pointerSize, self.colors[self.colorIndex], -1)

                    # Check for gestures
                    current_two_fingers = self._detect_two_fingers(hand_landmarks)
                    current_thumbs_down = self._detect_thumbs_down(hand_landmarks)

                    # Handle Two Fingers Gesture
                    if current_two_fingers:
                        if self.two_fingers_start_time is None:
                            self.two_fingers_start_time = time.time()
                        elif time.time() - self.two_fingers_start_time >= self.two_fingers_hold_duration:
                            self.save_drawing()
                            self.two_fingers_start_time = None
                            self.saving_enabled = False
                    else:
                        self.two_fingers_start_time = None
                        self.saving_enabled = True

                    # Handle Thumbs Down Gesture
                    if current_thumbs_down:
                        if self.thumbs_down_start_time is None:
                            self.thumbs_down_start_time = time.time()
                        elif time.time() - self.thumbs_down_start_time >= self.thumbs_down_hold_duration:
                            self.clear_canvas()
                            self.thumbs_down_start_time = None
                            self.clear_enabled = False
                    else:
                        self.thumbs_down_start_time = None
                        self.clear_enabled = True

                    # Check for pinch gesture (drawing mode)
                    pinch_distance = ((thumb_pos[0] - center[0])**2 + (thumb_pos[1] - center[1])**2)**0.5

                    if pinch_distance < 40:  # Pinch threshold
                        self.drawing_mode = True
                        if center[1] <= self.menu_height:  # Menu area
                            if 40 <= center[0] <= 140:
                                self.clear_canvas()
                            elif 160 <= center[0] <= 255:
                                self.colorIndex = 0
                            elif 275 <= center[0] <= 370:
                                self.colorIndex = 1
                            elif 390 <= center[0] <= 485:
                                self.colorIndex = 2
                            elif 505 <= center[0] <= 600:
                                self.colorIndex = 3
                        else:  # Drawing area
                            if self.drawing_mode:
                                if self.colorIndex == 0:
                                    self.bpoints[len(self.bpoints) - 1].append(center)
                                elif self.colorIndex == 1:
                                    self.gpoints[len(self.gpoints) - 1].append(center)
                                elif self.colorIndex == 2:
                                    self.rpoints[len(self.rpoints) - 1].append(center)
                                elif self.colorIndex == 3:
                                    self.ypoints[len(self.ypoints) - 1].append(center)
                    else:
                        self.drawing_mode = False
                        #start new line.
                        self.bpoints.append(deque(maxlen=2000))
                        self.gpoints.append(deque(maxlen=2000))
                        self.rpoints.append(deque(maxlen=2000))
                        self.ypoints.append(deque(maxlen=2000))

            # Draw all the points
            points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
            for i, color_points in enumerate(points):
                for stroke in color_points:
                    for j in range(1, len(stroke)):
                        if stroke[j - 1] is not None and stroke[j] is not None:
                            cv2.line(self.paintWindow, stroke[j - 1], stroke[j], self.colors[i], self.brushSize)
                            cv2.line(frame, stroke[j - 1], stroke[j], self.colors[i], 2)

            # Show current color and brush size
            cv2.rectangle(frame, (10, 10), (30, 30), self.colors[self.colorIndex], -1)
            cv2.putText(frame, f"Size: {self.brushSize}", (40, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Output", frame)
            cv2.imshow("Paint", self.paintWindow)

            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Cleanup
        self.cleanup()
        self.root.deiconify()

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def exit_program(self):
        """Exit the application"""
        self.cleanup()
        self.root.quit()
        sys.exit()

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = AirCanvasApp()
    app.run()
