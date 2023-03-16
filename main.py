import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

from predict_emotion import add_box_to_frame


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.show_webcam()

    def create_widgets(self):
        self.webcam_button = tk.Button(self, text="Webcam", command=self.show_webcam)
        self.webcam_button.pack(side="left")
        self.image_button = tk.Button(self, text="Imagem", command=self.show_image)
        self.image_button.pack(side="left")
        self.quit_button = tk.Button(self, text="Sair", command=self.master.destroy)
        self.quit_button.pack(side="right")

        self.image_label = tk.Label(self)
        self.image_label.pack()

    def show_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.show_webcam_frame()

    def show_webcam_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            self.master.after(10, self.show_webcam_frame)

    def show_image(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.show_frame(img)

    def show_frame(self, frame):
        frame = add_box_to_frame(frame)
        h, w, c = frame.shape
        if h > 700:
            ratio = 700 / h
            frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(master=self, image=image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo


root = tk.Tk()
app = Application(master=root)
root.title("Metaforando")
app.mainloop()
