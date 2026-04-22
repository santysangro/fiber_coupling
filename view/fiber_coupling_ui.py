import customtkinter as ctk
from controller.fiber_coupling import FiberCoupling

class FiberCouplingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.fiberCoupling = FiberCoupling()
        # Title
        self.title = ctk.CTkLabel(
            self,
            text="Fiber Coupling\nFrom Scratch",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title.grid(row=0, column=0, pady=(10, 5))

        # Status label (future: signal / progress)
        self.status = ctk.CTkLabel(
            self,
            text="Status: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.status.grid(row=1, column=0, pady=5)

        # Start button
        self.start_button = ctk.CTkButton(
            self,
            text="Start Alignment",
            command=self.start_alignment
        )
        self.start_button.grid(row=2, column=0, pady=10)

    def start_alignment(self):
        print("Starting scratch alignment...")
        self.fiberCoupling.run_optimization()
        # TODO: later connect:
        # - servo scan
        # - pico signal read
        # - optimization loop

        self.status.configure(text="Status: Running")