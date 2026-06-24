import customtkinter as ctk
import threading
from run_blind_fc import run_blind_coupling

class FiberCouplingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)

        self.title = ctk.CTkLabel(
            self,
            text="Blind Fiber Coupling",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title.grid(row=0, column=0, pady=(10, 5))

        self.status = ctk.CTkLabel(
            self,
            text="Status: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.status.grid(row=1, column=0, pady=5)

        self.start_button = ctk.CTkButton(
            self,
            text="Start Blind Coupling",
            command=self.start_alignment
        )
        self.start_button.grid(row=2, column=0, pady=10)

    def start_alignment(self):
        self.status.configure(text="Status: Running")
        self.start_button.configure(state="disabled")

        thread = threading.Thread(target=self.run_alignment_thread, daemon=True)
        thread.start()

    def run_alignment_thread(self):
        try:
            run_blind_coupling()
            self.status.configure(text="Status: Finished")
        except Exception as e:
            self.status.configure(text=f"Status: Error")
            print("Blind coupling failed:", e)
        finally:
            self.start_button.configure(state="normal")