import customtkinter as ctk
from controller.fiber_coupling import FiberCoupling
import numpy as np
from configuration import SERVOS_TEST_POS
class FiberCouplingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        min_bound = np.subtract(SERVOS_TEST_POS, [200, 200, 200, 200, 100])
        max_bound = np.add(SERVOS_TEST_POS, [200, 200, 200, 200, 100])
        self.fiberCoupling = FiberCoupling(min_boundary=min_bound, max_boundary=max_bound)
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