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

        # ======================
        # Channel selector
        # ======================
        self.channel_label = ctk.CTkLabel(self, text="Select Channel:")
        self.channel_label.pack(padx=5, pady=(5, 0))

        self.channel_var = tk.StringVar(value="A")

        self.channel_menu = ctk.CTkOptionMenu(
            self,
            values=["A", "B"],
            variable=self.channel_var
        )
        self.channel_menu.pack(padx=5, pady=5)

        # ======================
        # Voltage range selector
        # ======================
        self.range_label = ctk.CTkLabel(self, text="Select Range:")
        self.range_label.pack(padx=5, pady=(5, 0))

        self.range_var = tk.StringVar(value="1V")

        self.range_menu = ctk.CTkOptionMenu(
            self,
            values=["20MV", "50MV", "100MV", "200MV", "500MV","1V", "2V", "5V","10V""5V"],
            variable=self.range_var
        )
        self.range_menu.pack(padx=5, pady=5)

        # ======================
        # Read button
        # ======================
        self.button = ctk.CTkButton(
            self, text="Read Signal", command=self.read_signal
        )
        self.button.pack(padx=5, pady=5)

        # ======================
        # Output box
        # ======================
        self.output_text = ctk.CTkTextbox(self, width=300, height=200)
        self.output_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

    def read_signal(self):
        try:
            selected_range = self.range_var.get()
            selected_channel = self.channel_var.get()

            p_range = f'PS2000_{selected_range}'

            # Create picoscope instance
            picoscope = Picoscope(
                voltage_range=p_range,
            )

            voltage, _ = picoscope.get_voltage(CHANNEL=selected_channel)
            picoscope.close_device()
            self.output_text.insert(
                tk.END,
                f"Channel {selected_channel}: {voltage} mV\n"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))