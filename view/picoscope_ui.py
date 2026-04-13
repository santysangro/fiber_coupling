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

