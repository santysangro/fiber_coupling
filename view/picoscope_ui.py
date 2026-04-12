import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

from controller.picoscope import Picoscope


class ReadPicoscopeFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.label = ctk.CTkLabel(
            self, text="Read Picoscope Signal", font=("Helvetica", 14, "bold")
        )
        self.label.pack(padx=5, pady=5)

        # --- Voltage range selector ---
        self.range_label = ctk.CTkLabel(self, text="Select Range:")
        self.range_label.pack(padx=5, pady=(5, 0))

        self.range_var = tk.StringVar(value="1 V")

        self.range_menu = ctk.CTkOptionMenu(
            self,
            values=["50 mV", "1 V", "2 V", "5 V"],
            variable=self.range_var
        )
        self.range_menu.pack(padx=5, pady=5)

        # --- Read button ---
        self.button = ctk.CTkButton(
            self, text="Read Signal", command=self.read_signal
        )
        self.button.pack(padx=5, pady=5)

        # --- Output box ---
        self.output_text = ctk.CTkTextbox(self, width=300, height=200)
        self.output_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def read_signal(self):
        try:
            selected_range = self.range_var.get()
            p_range  = f'PS2000_{selected_range}'
            # Create picoscope instance with range
            picoscope = Picoscope(voltage_range=p_range)

            voltage, _ = picoscope.get_voltage()

            self.output_text.insert(tk.END, f"{voltage} mV\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


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
