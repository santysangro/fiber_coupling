import customtkinter as ctk
from test_nn_corrected_ui import TestNnCorrectedFrame
from generate_actuator_ui import GenerateActuatorFrame
from generate_data_ui import GenerateDataFrame
from run_prediction_ui import RunPredictionFrame
from test_actuator_ui import TestActuatorFrame
from test_nn_actuator_ui import TestNnActuatorFrame
from test_nn_ui import TestNnFrame
from servos_ui import ReadServosFrame, WriteServosFrame
from model_training_ui import TrainModelFrame
from model_finetuning_ui import FinetuneModelFrame
from cameras_ui import CamerasFrame


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment Control Panel")

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
            row=0, column=0, rowspan=2, sticky="nsew", **paddings)

        # Write Servo Positions Section
        self.write_servos_frame = WriteServosFrame(root)
        self.write_servos_frame.grid(
            row=0, column=1, rowspan=2, sticky="nsew", **paddings)

        # Train base model
        self.train_model_frame = TrainModelFrame(root)
        self.train_model_frame.grid(
            row=0, column=2, rowspan=2, sticky="nsew", **paddings)

        # Finetune base model
        self.finetune_model_frame = FinetuneModelFrame(root)
        self.finetune_model_frame.grid(
            row=0, column=3, rowspan=3, sticky="nsew", **paddings)

        # Run prediction
        self.test_nn_frame = RunPredictionFrame(root)
        self.test_nn_frame.grid(
            row=0, column=4, rowspan=4, sticky="nsew", **paddings)

        # Generate actuator
        self.generate_actuator_frame = GenerateActuatorFrame(root)
        self.generate_actuator_frame.grid(
            row=2, column=0, sticky="nsew", **paddings)

        # Test actuator
        self.test_actuator_frame = TestActuatorFrame(root)
        self.test_actuator_frame.grid(
            row=2, column=1, rowspan=2, sticky="nsew", **paddings)
        
        # Generate training data
        self.generate_data_frame = GenerateDataFrame(root)
        self.generate_data_frame.grid(
            row=2, column=2, rowspan=2, sticky="nsew", **paddings)
        
        # # Test neural network
        # self.test_nn_frame = TestNnFrame(root)
        # self.test_nn_frame.grid(
        #     row=3, column=0, sticky="nsew", **paddings)

        # Cameras frame
        self.cameras_frame = CamerasFrame(root)
        self.cameras_frame.grid(
            row=3, column=0, sticky="nsew", **paddings)

        # # Test neural network with neural network correction
        # self.test_nn_frame = TestNnCorrectedFrame(root)
        # self.test_nn_frame.grid(
        #     row=3, column=3, sticky="nsew", **paddings)

        # Test neural network with closed loop correction
        self.test_nn_actuator_frame = TestNnActuatorFrame(root)
        self.test_nn_actuator_frame.grid(
            row=3, column=3, sticky="nsew", **paddings)


if __name__ == "__main__":
    # Modes: system (default), light, dark
    ctk.set_appearance_mode("dark")
    # Themes: blue (default), dark-blue, green
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    app = App(root)
    root.mainloop()
