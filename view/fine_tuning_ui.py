import customtkinter as ctk


class FineTuningFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)

        # Title
        self.title = ctk.CTkLabel(
            self,
            text="Fine Tuning",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title.grid(row=0, column=0, pady=(10, 5))

        # Status
        self.status = ctk.CTkLabel(
            self,
            text="Status: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.status.grid(row=1, column=0, pady=5)

        # Step size control (important for fine tuning)
        self.step_label = ctk.CTkLabel(self, text="Step Size")
        self.step_label.grid(row=2, column=0, pady=(10, 2))

        self.step_slider = ctk.CTkSlider(
            self,
            from_=0.1,
            to=10,
            number_of_steps=100
        )
        self.step_slider.set(1.0)
        self.step_slider.grid(row=3, column=0, pady=5)

        # Optimize button
        self.optimize_button = ctk.CTkButton(
            self,
            text="Optimize Signal",
            command=self.start_optimization
        )
        self.optimize_button.grid(row=4, column=0, pady=10)

        # Stop button (important for control loops)
        self.stop_button = ctk.CTkButton(
            self,
            text="Stop",
            fg_color="darkred",
            command=self.stop_optimization
        )
        self.stop_button.grid(row=5, column=0, pady=5)

        self.running = False

    def start_optimization(self):
        self.running = True
        print("Starting fine tuning optimization...")

        self.status.configure(text="Status: Optimizing")

        # TODO later:
        # - read picoscope signal
        # - small servo adjustments
        # - gradient/ascent or hill-climb

    def stop_optimization(self):
        self.running = False
        print("Stopping fine tuning...")

        self.status.configure(text="Status: Stopped")