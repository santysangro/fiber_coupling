import customtkinter as ctk
from view.servos_ui import ReadServosFrame, WriteServosFrame
from view.picoscope_ui import ReadPicoscopeFrame
from view.fiber_coupling_ui import FiberCouplingFrame
from view.fine_tuning_ui import FineTuningFrame


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Fiber Coupling Control Panel")

        # Configure grid (Title row + content row)
        self.root.grid_rowconfigure(0, weight=0)  # title
        self.root.grid_rowconfigure(1, weight=1)  # main content

        for col in range(3):
            self.root.grid_columnconfigure(col, weight=1)

        paddings = {"padx": 10, "pady": 10}

        # ======================
        # Title
        # ======================
        self.title_label = ctk.CTkLabel(
            root,
            text="Fiber Coupling",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=3, pady=15)

        # ======================
        # Column 1: Servos
        # ======================
        self.servos_frame = ctk.CTkFrame(root)
        self.servos_frame.grid(row=1, column=0, sticky="nsew", **paddings)

        self.servos_frame.grid_rowconfigure((0, 1), weight=1)
        self.servos_frame.grid_columnconfigure(0, weight=1)

        self.read_servos_frame = ReadServosFrame(self.servos_frame)
        self.read_servos_frame.grid(row=0, column=0, sticky="nsew", **paddings)

        self.write_servos_frame = WriteServosFrame(self.servos_frame)
        self.write_servos_frame.grid(row=1, column=0, sticky="nsew", **paddings)

        # ======================
        # Column 2: Picoscope
        # ======================
        self.pico_frame = ctk.CTkFrame(root)
        self.pico_frame.grid(row=1, column=1, sticky="nsew", **paddings)

        self.pico_frame.grid_rowconfigure(0, weight=1)
        self.pico_frame.grid_columnconfigure(0, weight=1)

        self.red_signal_frame = ReadPicoscopeFrame(self.pico_frame)
        self.red_signal_frame.grid(row=0, column=0, sticky="nsew", **paddings)
        # ======================
        # Column 3: Fiber Coupling
        # ======================
        self.fiber_frame = ctk.CTkFrame(root)
        self.fiber_frame.grid(row=1, column=2, sticky="nsew", **paddings)

        self.fiber_frame.grid_rowconfigure((0, 1), weight=1)
        self.fiber_frame.grid_columnconfigure(0, weight=1)

        # ----------------------
        # Scratch alignment (coarse)
        # ----------------------
        self.scratch_frame = FiberCouplingFrame(self.fiber_frame)
        self.scratch_frame.grid(row=0, column=0, sticky="nsew", **paddings)

        # ----------------------
        # Fine tuning (optimization)
        # ----------------------
        self.tuning_frame = FineTuningFrame(self.fiber_frame)
        self.tuning_frame.grid(row=1, column=0, sticky="nsew", **paddings)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    app = App(root)
    root.mainloop()