import customtkinter as ctk
import threading

from run_fine_tune import run_fine_tune


class FineTuningFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.running = False

        self.title = ctk.CTkLabel(
            self,
            text="Fine Tuning",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title.grid(row=0, column=0, pady=(10, 5))

        self.status = ctk.CTkLabel(
            self,
            text="Status: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.status.grid(row=1, column=0, pady=5)

        self.optimize_button = ctk.CTkButton(
            self,
            text="Optimize Signal",
            command=self.start_optimization
        )
        self.optimize_button.grid(row=4, column=0, pady=10)

        self.stop_button = ctk.CTkButton(
            self,
            text="Stop",
            fg_color="darkred",
            command=self.stop_optimization
        )
        self.stop_button.grid(row=5, column=0, pady=5)

    def start_optimization(self):
        if self.running:
            return

        self.running = True
        self.status.configure(text="Status: Optimizing")
        self.optimize_button.configure(state="disabled")

        thread = threading.Thread(
            target=self.run_optimization_thread,
            daemon=True
        )
        thread.start()

    def run_optimization_thread(self):
        try:
            run_fine_tune()
            self.status.configure(text="Status: Finished")
        except Exception as e:
            self.status.configure(text="Status: Error")
            print("Fine tuning failed:", e)
        finally:
            self.running = False
            self.optimize_button.configure(state="normal")

    def stop_optimization(self):
        self.running = False
        self.status.configure(text="Status: Stopped")
        print("Stopping fine tuning...")