# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
from omegaconf import OmegaConf
import carla

from opencda.version import __version__
from CARLA.version import check_version
from CARLA.map.carla_map import check_map
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description='OpenCDA scenario runner.')

    # add arguments to the parser
    parser.add_argument('-v', '--version',
                        type=check_version, default='0.9.14',
                        help='Specify CARLA version (must be between 0.9.11 and 0.9.15)')
    parser.add_argument('-s', '--test_scenario', required=True, type=str, default='simulation',
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument('-t', '--number_ticks', type=int, default=10,
                        help='Specify number of ticks for the simulation. Carla server runs in Synchronous mode.'
                             'number_ticks records the number of transition from state t to state t + 1.')
    parser.add_argument('-m', '--map', type=check_map, default='Town06',
                        help='Specify the desired Carla map in the simulation.')
    parser.add_argument("-c", "--number_cavs",
                        type=lambda v: int(v) if int(v) > 1 else
                        (_ for _ in ()).throw(argparse.ArgumentTypeError(f"{v} must be > 1")),
                        default=3, help="Specify the number of CAVs in the simulation (must be > 1)."
    )
    parser.add_argument('-n', '--number_vehicles', type=int, default=10,
                        help='Specify the number of vehicles (not connected) in the simulation.')
    parser.add_argument('-p', '--number_pedestrians', type=int, default=10,
                        help='Specify the number of pedestrians in the simulation.')
    parser.add_argument('--apply_ml',
                        action='store_true', default=False,
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('--record', action='store_true',
                        help='whether to record and save the simulation process to .log file')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def main() -> None:

    # parse the arguments
    opt = arg_parse()

    # print the version of OpenCDA
    print('OpenCDA Version: %s' % __version__)

    # set the default yaml file
    default_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')

    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)

    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)


    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # add cavs
    for cav in range(1, opt.number_cavs):
        scene_dict['scenario']['single_cav_list'].append({'name': f'cav{cav}'})

    run_scenario(opt, scene_dict)


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town=opt.map,
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder(f'{opt.test_scenario}.log', True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name=f'{opt.test_scenario}',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()

        # run steps
        number_ticks = 0
        while number_ticks < opt.number_ticks:
            print(f"{number_ticks + 1} / {opt.number_ticks}", end='\r')
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=50),
                carla.Rotation(
                    pitch=-
                    90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            number_ticks += 1
    finally:
        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.destroy_actors()
        scenario_manager.close()

        eval_manager.evaluate()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')

    # Command: python simulation.py -s simulation -t 100 -m Town03