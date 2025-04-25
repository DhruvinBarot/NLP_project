import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import pyttsx3
import time
import numpy as np
import mediapipe as mp
from Function import persons_input  # Assuming this is your custom function 

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignABC - Gesture Recognition")
        self.root.resizable(False, False)  # Fixing GUI, not zoomable

        # New color scheme
        self.bg_color = "#141414"
        self.text_color = "white"
        self.highlight_bg_color = "#F0E130"  # Summer yellow
        self.font_style = "Orbitron"
        self.root.focus_force()

        # Main frame with summer yellow border
        self.main_frame = tk.Frame(root, bg=self.bg_color, padx=0, pady=0, highlightbackground=self.highlight_bg_color, highlightthickness=2)
        self.main_frame.grid(row=1, column=0, sticky="nsew")

        # Load logo images during initialization
        self.load_images()

        # Logo labels
        self.logo1_label = tk.Label(self.main_frame, image=self.logo1_tk, bg=self.bg_color)
        self.logo1_label.grid(row=0, column=0, padx=(0, 15), pady=0, sticky="nw")

        self.logo2_label = tk.Label(self.main_frame, image=self.logo2_tk, bg=self.bg_color)
        self.logo2_label.grid(row=0, column=1, padx=(0, 0), pady=15, sticky="nw")

        # Camera frame on the left
        self.camera_frame = tk.Frame(self.main_frame, bg=self.bg_color, highlightbackground="white", highlightthickness=2)
        self.camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.initialize_camera()

        self.buttons_frame = tk.Frame(self.camera_frame, bg=self.bg_color)
        self.buttons_frame.pack(side="top", pady=10)

        # Load button images during initialization
        self.load_button_images()

        # Place images on the buttons frame
        self.place_buttons()

        # Gesture frame on the right
        self.initialize_gesture_frame()

        # Initialize speech engine and hand detection
        self.initialize_engine_and_hands()

        self.gesture_text = ""
        self.gesture_history = []
        self.last_gesture_time = time.time()

        self.start_detection()

        # Adding introduction speech
        self.add_intro_speech()

        # Bind the backspace key to a method
        self.root.bind("<BackSpace>", self.handle_backspace)
        # Bind the space key to a method
        self.root.bind("<space>", self.handle_space)

    def load_images(self):
        try:
            # Load logo images
            logo1_path = "C:/Users/Aloof/Sign-Language/IMAGES/tip-logo.png"
            logo1_img = Image.open(logo1_path)
            logo1_img = logo1_img.resize((452, 89), Image.LANCZOS)
            self.logo1_tk = ImageTk.PhotoImage(logo1_img)

            logo2_path = "C:/Users/Aloof/Sign-Language/IMAGES/Logo.png"
            logo2_img = Image.open(logo2_path)
            logo2_img = logo2_img.resize((452, 59), Image.LANCZOS)
            self.logo2_tk = ImageTk.PhotoImage(logo2_img)
        except Exception as e:
            print("Error loading image:", e)
            self.logo1_tk = None
            self.logo2_tk = None

    def initialize_camera(self):
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

    def load_button_images(self):
        try:
            # Resize images to 28% smaller size
            save_img = Image.open("C:/Users/Aloof/Sign-Language/IMAGES/save.png")
            save_img = save_img.resize((int(504 * 0.28), int(125 * 0.28)), Image.LANCZOS)
            self.save_img_tk = ImageTk.PhotoImage(save_img)

            speech_img = Image.open("C:/Users/Aloof/Sign-Language/IMAGES/speech.png")
            speech_img = speech_img.resize((int(504 * 0.28), int(125 * 0.28)), Image.LANCZOS)
            self.speech_img_tk = ImageTk.PhotoImage(speech_img)

            clear_img = Image.open("C:/Users/Aloof/Sign-Language/IMAGES/clear.png")
            clear_img = clear_img.resize((int(504 * 0.28), int(125 * 0.28)), Image.LANCZOS)
            self.clear_img_tk = ImageTk.PhotoImage(clear_img)
        except Exception as e:
            print("Error loading button images:", e)
            self.save_img_tk = None
            self.speech_img_tk = None
            self.clear_img_tk = None

    def place_buttons(self):
        self.save_button = tk.Label(self.buttons_frame, image=self.save_img_tk, bg=self.bg_color)
        self.save_button.pack(side="left", padx=10)
        self.save_button.bind("<Button-1>", self.save_text)

        self.speech_button = tk.Label(self.buttons_frame, image=self.speech_img_tk, bg=self.bg_color)
        self.speech_button.pack(side="left", padx=10)
        self.speech_button.bind("<Button-1>", self.speech)

        self.clear_button = tk.Label(self.buttons_frame, image=self.clear_img_tk, bg=self.bg_color)
        self.clear_button.pack(side="left", padx=10)
        self.clear_button.bind("<Button-1>", self.clear_text)

    def initialize_gesture_frame(self):
        # Gesture frame on the right
        self.gesture_frame = tk.Frame(self.main_frame, bg=self.bg_color, highlightbackground="white", highlightthickness=2)
        self.gesture_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Load the new logo image as background
        new_logo_path = "C:/Users/Aloof/Sign-Language/IMAGES/_BIG LOGO.png"
        try:
            bg_logo_image = Image.open(new_logo_path)
            bg_logo_image = bg_logo_image.resize((600, 600), Image.LANCZOS)
            self.bg_logo_tk = ImageTk.PhotoImage(bg_logo_image)
        except Exception as e:
            print("Error loading background image:", e)
            self.bg_logo_tk = None

        # Create a label for the background image
        self.bg_logo_label = tk.Label(self.gesture_frame, image=self.bg_logo_tk, bg=self.bg_color)
        self.bg_logo_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Image path modification
        new_title_image_path = "C:/Users/Aloof/Sign-Language/IMAGES/_HR.png"  # New image path

        # Load the new image
        try:
            title_image = Image.open(new_title_image_path)
            title_image = title_image.resize((452, 59), Image.LANCZOS)
            self.title_image_tk = ImageTk.PhotoImage(title_image)
        except Exception as e:
            print("Error loading new image:", e)
            self.title_image_tk = None

        # Title image label
        self.title_image_label = tk.Label(self.gesture_frame, image=self.title_image_tk, bg=self.bg_color)
        self.title_image_label.pack(side="top", pady=5)

        # Text label
        self.gesture_text_label = tk.Label(self.gesture_frame, text="", font=(self.font_style, 20), wraplength=400, justify="left", fg=self.text_color, bg=self.bg_color)
        self.gesture_text_label.pack(side="top", pady=5)

        # Gesture history below the text
        self.history_frame = tk.Frame(self.gesture_frame, bg=self.bg_color, highlightbackground="white", highlightthickness=2)
        self.history_frame.pack(side="bottom", pady=0)

        self.history_title_label = tk.Label(self.history_frame, text="GESTURE HISTORY:", font=(self.font_style, 16, "bold"), fg=self.text_color, bg=self.bg_color)
        self.history_title_label.pack()

        self.history_listbox = tk.Listbox(self.history_frame, width=50, height=5, font=(self.font_style, 12), fg=self.text_color, bg=self.bg_color)
        self.history_listbox.pack(side="left", fill="both", expand=True)

        self.history_scrollbar_y = tk.Scrollbar(self.history_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_scrollbar_y.pack(side="right", fill="y")

        self.history_listbox.config(yscrollcommand=self.history_scrollbar_y.set)

        self.history_listbox.bind('<ButtonRelease-1>', self.speak_from_history)

    def initialize_engine_and_hands(self):
        self.engine = pyttsx3.init()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def start_detection(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.configure(image=img_tk)
            self.camera_label.image = img_tk

            imgH, imgW, _ = frame.shape
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_coordinates = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for index, landmark in enumerate(hand_landmarks.landmark):
                        x_coordinate, y_coordinate = int(landmark.x * imgW), int(landmark.y * imgH)
                        hand_coordinates.append([index, x_coordinate, y_coordinate])
                    
                    mp_drawing = mp.solutions.drawing_utils
                    annotated_image = frame.copy()
                    is_gray = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        if is_gray:
                            color = (128, 128, 128)  # Gray
                        else:
                            color = (255, 255, 255)  # White
                        mp_drawing.draw_landmarks(
                            annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))
                        is_gray = not is_gray
                
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    annotated_image_pil = Image.fromarray(annotated_image_rgb)
                    annotated_image_tk = ImageTk.PhotoImage(image=annotated_image_pil)
                    self.camera_label.configure(image=annotated_image_tk)
                    self.camera_label.image = annotated_image_tk
                
                hand_coordinates = np.array(hand_coordinates)
                string = persons_input(hand_coordinates)

                if string:
                    current_time = time.time()
                    if current_time - self.last_gesture_time >= 2:
                        self.gesture_text += string.strip().replace(" ", "")
                        self.last_gesture_time = current_time
                        self.gesture_text_label.config(text=self.gesture_text)

        self.root.after(10, self.start_detection)

    def add_intro_speech(self):
        introduction_text = "hi"
        self.engine.say(introduction_text)
        self.engine.runAndWait()

    def speech(self, event):
        if self.gesture_text:
            self.engine.say(self.gesture_text)
            self.engine.runAndWait()
        self.gesture_frame.config(bg="#F0E130")
        self.gesture_frame.config(highlightbackground="#F0E130")

    def clear_text(self, event):
        self.gesture_text = ""
        self.gesture_text_label.config(text="")
        self.gesture_frame.config(bg="red")
        self.gesture_frame.config(highlightbackground="red")

    def save_text(self, event):
        self.gesture_frame.config(bg="blue")
        self.gesture_frame.config(highlightbackground="blue")
        if  self.gesture_text:
            timestamp = time.time()
            self.gesture_history.append((timestamp, self.gesture_text))
            self.update_history_listbox()

    def speak_from_history(self, event):
        self.gesture_frame.config(bg="white")
        self.gesture_frame.config(highlightbackground="white")
        selected_index = self.history_listbox.curselection()
        if selected_index:
            selected_text = self.history_listbox.get(selected_index)
            _, gesture = selected_text.split(" - ", 1)
            self.engine.say(gesture)
            self.engine.runAndWait()

    def update_history_listbox(self):
        self.history_listbox.delete(0, tk.END)
        for timestamp, gesture in self.gesture_history:
            formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
            self.history_listbox.insert(tk.END, f"TIME: {formatted_time} - {gesture}")

    def handle_backspace(self, event):
        if self.gesture_text:
            self.gesture_frame.config(bg="white")
            self.gesture_frame.config(highlightbackground="white")
            self.gesture_text = self.gesture_text[:-1]
            self.gesture_text_label.config(text=self.gesture_text)
            self.update_history_listbox()

    def handle_space(self, event):
        self.main_frame.focus_set()
        self.gesture_frame.config(bg="purple")
        self.gesture_frame.config(highlightbackground="purple")
        self.gesture_text += " "
        self.gesture_text_label.config(text=self.gesture_text)
        self.main_frame.focus_set()

    def on_closing(self):
        self.cap.release()
        self.root.destroy()
        

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
