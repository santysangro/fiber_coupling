import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

from configuration import STS_IDS
from controller.servos import Servos


class ReadServosFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.label = ctk.CTkLabel(
            self, text="Read Servo Positions", font=("Helvetica", 14, "bold"))
        self.label.pack(padx=5, pady=5)

        self.button = ctk.CTkButton(
            self, text="Read Positions", command=self.read_servo_positions)
        self.button.pack(padx=5, pady=5)

        self.output_text = ctk.CTkTextbox(
            self, width=30, height=10)
        self.output_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def read_servo_positions(self):
        try:
            with Servos() as servos:
                positions = servos.read()
                positions_text = ", ".join(map(str, positions))
                self.output_text.insert("0.0", positions_text + "\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return



class WriteServosFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # UI options
        paddings = {"padx": 5, "pady": 6}

        self.label = ctk.CTkLabel(
            self, text="Write Servo Positions", font=("Helvetica", 14, "bold"))
        self.label.pack(**paddings)

        self.description = ctk.CTkLabel(
            self, text="Enter the positions to which the servos should be moved, in the format 'pos_1, pos_2, pos_3, pos_4'", wraplength=300)
        self.description.pack(**paddings)

        self.entry = ctk.CTkEntry(self)
        self.entry.pack(**paddings, fill=tk.X)

        self.button = ctk.CTkButton(
            self, text="Write Positions", command=self.write_servo_positions)
        self.button.pack(**paddings)

        self.output_text = ctk.CTkTextbox(
            self, width=30, height=10)
        self.output_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def write_servo_positions(self):
        positions = self.entry.get().split(', ')

        try:
            positions = [int(pos) for pos in positions]

            if len(positions) != len(STS_IDS):
                raise ValueError(
                    "Number of positions does not match the number of servos")
            if any(pos < 0 for pos in positions):
                raise ValueError("Negative values are not allowed")

            with Servos() as servos:
                servos.precise_write(positions)

            self.output_text.insert(
                tk.END, f"Servo positions set to {positions}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
