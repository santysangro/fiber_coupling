import customtkinter as ctk
from view.servos_ui import ReadServosFrame, WriteServosFrame
from view.picoscope_ui import ReadPicoscopeFrame

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Fiber Coupling Control Panel")

        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(4, weight=1)

        # UI options
        paddings = {"padx": 10, "pady": 10}

        # Read Servo Positions Section
        self.read_servos_frame = ReadServosFrame(root)
        self.read_servos_frame.grid(
            row=0, column=0, rowspan=1, sticky="nsew", **paddings)

        # Write Servo Positions Section
        self.write_servos_frame = WriteServosFrame(root)
        self.write_servos_frame.grid(
            row=0, column=1, rowspan=2, sticky="nsew", **paddings)

        # Read Voltage

        self.red_signal_frame = ReadPicoscopeFrame(root)
        self.read_servos_frame.grid(
            row=2, column=2, rowspan=1, sticky="nsew", **paddings)
        """
        # Write Servo Positions Section
        self.write_servos_frame = WriteServosFrame(root)
        self.write_servos_frame.grid(
            row=0, column=1, rowspan=2, sticky="nsew", **paddings)
        """

if __name__ == "__main__":
    # Modes: system (default), light, dark
    ctk.set_appearance_mode("dark")
    # Themes: blue (default), dark-blue, green
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    app = App(root)
    root.mainloop()
