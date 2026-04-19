from configuration import M_LOOP_ITERATIONS, SERVOS_TEST_POS
from run_experiment_fiber_coupling import run_experiment
from controller.servos import Servos
from controller.picoscope import Picoscope
import time

# Imports for M-LOOP
import mloop.interfaces as mli  # type: ignore
import mloop.controllers as mlc  # type: ignore
import mloop.visualizations as mlv  # type: ignore


# Declare your custom class that inherits from the Interface class



class CustomInterface(mli.Interface):

    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()

        # Attributes of the interface can be added here
        # If you want to precalculate any variables etc. this is the place to do it
        self.picoscope = Picoscope(voltage_range='PS2000_2V')

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):

        # Get parameters from the provided dictionary
        params = params_dict['params']

        # args = [params, self.pc_connection]
        # if self.target_position is not None:
        #     args.append(self.target_position)

        cost = run_experiment(params, self.picoscope)
        # The cost, uncertainty and bad boolean must all be returned as a dictionary
        # You can include other variables you want to record as well if you want
        cost_dict = {'cost': cost}
        time.sleep(0.1)

        return cost_dict


def run_mloop():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface()
    with Servos() as servos:
        initial_positions = servos.read()
        init_positions = [pos[1] for pos in initial_positions]
        print("Initial positions: ", init_positions)

    # Next create the controller. Provide it with your interface and any options you want to set
    num_servos = len(init_positions)
    min_boundary = [0,0]#, 794,736] #[max(2547, 0) for pos in init_positions] #pos - MLOOP_RANGE
    max_boundary = [4090, 4090]#, 1194, 936] #[min(2847, 4095) for pos in init_positions]

    controller = mlc.create_controller(interface,
                                        controller_type= 'gaussian_process',
                                       max_num_runs=100, #M_LOOP_ITERATIONS,
                                       num_params=2,#num_servos,
                                       min_boundary=min_boundary,
                                       max_boundary=max_boundary,
                                       cost_has_noise=True,
                                       #first_params =[1860, 1625, 1039, 769],              #first parameters to try
                                        #trust_region=0.1,                     #maximum % move distance from best params
                                        )

    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)

    # You can also run the default sets of visualizations for the controller with one command
    # mlv.show_all_default_visualizations(controller)

    print("Finished MLOOP. Closing...")

    interface.picoscope.close_device()
    mlv.show_all_default_visualizations(controller)
    return controller.best_params


# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    run_mloop()
