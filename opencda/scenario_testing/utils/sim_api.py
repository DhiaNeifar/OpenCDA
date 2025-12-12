# -*- coding: utf-8 -*-
"""
Utilize scenario manager to manage CARLA simulation construction. This script
is used for carla simulation only, and if you want to manage the Co-simulation,
please use cosim_api.py.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import time
import math
import random
import sys
import json
from random import shuffle
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import carla
import numpy as np

from opencda.core.common.vehicle_manager import VehicleManager
from opencda.core.application.platooning.platooning_manager import \
    PlatooningManager
from opencda.core.common.rsu_manager import RSUManager
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.utils.customized_map_api import \
    load_customized_world, bcolors
from CARLA.manager import CarlaManager


def car_blueprint_filter(blueprint_library, carla_version='0.9.11'):
    """
    Exclude the uncommon vehicles from the default CARLA blueprint library
    (i.e., isetta, carlacola, cybertruck, t2).

    Parameters
    ----------
    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    carla_version : str
        CARLA simulator version, currently support 0.9.11 and 0.9.12. We need
        this as since CARLA 0.9.12 the blueprint name has been changed a lot.

    Returns
    -------
    blueprints : list
        The list of suitable blueprints for vehicles.
    """

    if carla_version == '0.9.11':
        print('old version')
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge_charger.police'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooperst'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes-benz.coupe'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.mustang.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz2017'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    else:
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge.charger_police'),
            blueprint_library.find('vehicle.dodge.charger_police_2020'),
            blueprint_library.find('vehicle.dodge.charger_2020'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooper_s'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes.coupe'),
            blueprint_library.find('vehicle.mercedes.coupe_2020'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.ford.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz_2017'),
            blueprint_library.find('vehicle.lincoln.mkz_2020'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    return blueprints


def multi_class_vehicle_blueprint_filter(label, blueprint_library, bp_meta):
    """
    Get a list of blueprints that have the class equals the specified label.

    Parameters
    ----------
    label : str
        Specified blueprint.

    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    bp_meta : dict
        Dictionary of {blueprint name: blueprint class}.

    Returns
    -------
    blueprints : list
        List of blueprints that have the class equals the specified label.

    """
    blueprints = [
        blueprint_library.find(k)
        for k, v in bp_meta.items() if v["class"] == label
    ]
    return blueprints


class ScenarioManager:
    """
    The manager that controls simulation construction, background traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Car;a HD Map.

    """

    def __init__(self, scenario_params,
                 apply_ml,
                 carla_version,
                 xodr_path=None,
                 town=None,
                 cav_world=None,
                 save_path=None):
        self.scenario_params = scenario_params
        self.carla_version = carla_version

        simulation_config = scenario_params['world']

        # set random seed if stated
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        self.carla_manager = CarlaManager()
        self.client = self.carla_manager.ensure_running()

        if xodr_path:
            self.world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                sleep_time = 5
                print(f"Loading map, sleeping for {sleep_time} seconds")
                self.world = self.client.load_world(town)
                time.sleep(sleep_time)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % town)
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        if simulation_config['sync_mode']:
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = \
                simulation_config['fixed_delta_seconds']
        else:
            sys.exit(
                'ERROR: Current version only supports sync simulation mode')

        self.world.apply_settings(new_settings)

        # set weather
        weather = self.set_weather(simulation_config['weather'])
        self.world.set_weather(weather)

        # Define probabilities for each type of blueprint
        self.use_multi_class_bp = scenario_params["blueprint"][
            'use_multi_class_bp'] if 'blueprint' in scenario_params else False
        if self.use_multi_class_bp:
            # bbx/blueprint meta
            with open(scenario_params['blueprint']['bp_meta_path']) as f:
                self.bp_meta = json.load(f)
            self.bp_class_sample_prob = scenario_params['blueprint'][
                'bp_class_sample_prob']

            # normalize probability
            self.bp_class_sample_prob = {
                k: v / sum(self.bp_class_sample_prob.values()) for k, v in
                self.bp_class_sample_prob.items()}

        self.cav_world = cav_world
        self.carla_map = self.world.get_map()
        self.apply_ml = apply_ml
        self.spawn_points = self.carla_map.get_spawn_points()
        self.save_path = save_path
        self.vehicles, self.pedestrians = None, None

    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather


    def spawn_vehicle(self, cav_vehicle_bp) -> carla.Actor:
        vehicle = None

        while self.spawn_points and vehicle is None:
            random_spawn_point = random.choice(self.spawn_points)
            vehicle = self.world.try_spawn_actor(cav_vehicle_bp, random_spawn_point)
            self.spawn_points.remove(random_spawn_point)

        if vehicle is None:
            print("No available spawn points to spawn the vehicle.")
            sys.exit(0)

        return vehicle

    def create_vehicle_manager(self, application,
                               map_helper=None,
                               data_dump=True):
        cav_list = []
        single_cav_list = random.sample(self.vehicles, len(self.scenario_params['scenario']['single_cav_list']))
        for i, cav_config in enumerate(
                self.scenario_params['scenario']['single_cav_list']):
            # in case the cav wants to join a platoon later
            # it will be empty dictionary for single cav application
            platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base', {})})
            cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                         platoon_base,
                                         cav_config)
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform

            # create vehicle manager for each cav
            vehicle = single_cav_list[i]
            vehicle_manager = VehicleManager(
                vehicle, cav_config, application,
                self.carla_map, self.cav_world,
                current_time=self.scenario_params['current_time'],
                data_dumping=data_dump,
                save_path=self.save_path)

            self.world.tick()

            vehicle_manager.v2x_manager.set_platoon(None)


            vehicle_manager.update_info()


            cav_list.append(vehicle_manager)

        return cav_list


    def create_vehicle_manager_old(self, application,
                               map_helper=None,
                               data_dump=True):
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating single CAVs.')
        # By default, we use lincoln as our cav model.
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)
        single_cav_list = []

        for i, cav_config in enumerate(
                self.scenario_params['scenario']['single_cav_list']):
            # in case the cav wants to join a platoon later
            # it will be empty dictionary for single cav application
            platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base',{})})
            cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                         platoon_base,
                                         cav_config)
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform

            cav_vehicle_bp.set_attribute('color', '0, 0, 255')
            vehicle = self.spawn_vehicle(cav_vehicle_bp)

            # create vehicle manager for each cav
            vehicle_manager = VehicleManager(
                vehicle, cav_config, application,
                self.carla_map, self.cav_world,
                current_time=self.scenario_params['current_time'],
                data_dumping=data_dump,
                save_path=self.save_path)

            self.world.tick()

            vehicle_manager.v2x_manager.set_platoon(None)

            destination = random.choice(self.spawn_points)
            destination = carla.Location(x=destination.location.x,
                                         y=destination.location.y,
                                         z=destination.location.z)
            vehicle_manager.update_info()
            vehicle_manager.set_destination(
                vehicle_manager.vehicle.get_location(),
                destination,
                clean=True)

            single_cav_list.append(vehicle_manager)

        return single_cav_list

    def create_vehicle_manager_from_scenario_runner(self, vehicle):
        """
        Create a single CAV with a loaded ego vehicle from SR.
        Different from the create_vehicle_manager API creating Carla vehicle from scratch,
        SR creates on its own only supports 'single' vehicle.

        Parameters
        ----------
        vehicle:
            The Carla ego vehicle created by ScenarioRunner.

        Returns
        -------
        single_cav_list : list
            A list contains the single CAV derived from the ego vehicle.
        """
        single_cav_params = self.scenario_params['scenario']['single_cav_list']
        if len(single_cav_params) != 1:
            raise ValueError('Only support one ego vehicle for ScenarioRunner')

        cav_config = single_cav_params[0]
        platoon_base = OmegaConf.create(
            {'platoon': self.scenario_params.get('platoon_base', {})})
        cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                     platoon_base,
                                     cav_config)
        vehicle_manager = VehicleManager(
            vehicle, cav_config, ['single'], self.carla_map, self.cav_world, save_path=self.save_path)

        self.world.tick()

        vehicle_manager.v2x_manager.set_platoon(None)

        destination = carla.Location(x=cav_config['destination'][0],
                                     y=cav_config['destination'][1],
                                     z=cav_config['destination'][2])
        vehicle_manager.update_info()
        vehicle_manager.set_destination(
            vehicle_manager.vehicle.get_location(),
            destination,
            clean=True)

        return [vehicle_manager]

    def create_platoon_manager(self, map_helper=None, data_dump=True):
        """
        Create a list of platoons.

        Parameters
        ----------
        map_helper : function
            A function to help spawn vehicle on a specific position in a
            specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating platoons/')
        platoon_list = []
        self.cav_world = CavWorld(self.apply_ml)

        # we use lincoln as default choice since our UCLA mobility lab use the
        # same car
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)

        # create platoons
        for i, platoon in enumerate(
                self.scenario_params['scenario']['platoon_list']):
            platoon = OmegaConf.merge(self.scenario_params['platoon_base'],
                                      platoon)
            platoon_manager = PlatooningManager(platoon, self.cav_world)
            for j, cav in enumerate(platoon['members']):
                platton_base = OmegaConf.create({'platoon': platoon})
                cav = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                      platton_base,
                                      cav
                                      )
                if 'spawn_special' not in cav:
                    spawn_transform = carla.Transform(
                        carla.Location(
                            x=cav['spawn_position'][0],
                            y=cav['spawn_position'][1],
                            z=cav['spawn_position'][2]),
                        carla.Rotation(
                            pitch=cav['spawn_position'][5],
                            yaw=cav['spawn_position'][4],
                            roll=cav['spawn_position'][3]))
                else:
                    spawn_transform = map_helper(self.carla_version,
                                                 *cav['spawn_special'])

                cav_vehicle_bp.set_attribute('color', '0, 0, 255')
                vehicle = self.world.spawn_actor(cav_vehicle_bp,
                                                 spawn_transform)

                # create vehicle manager for each cav
                vehicle_manager = VehicleManager(
                    vehicle, cav, ['platooning'],
                    self.carla_map, self.cav_world,
                    current_time=self.scenario_params['current_time'],
                    data_dumping=data_dump, save_path=self.save_path)

                # add the vehicle manager to platoon
                if j == 0:
                    platoon_manager.set_lead(vehicle_manager)
                else:
                    platoon_manager.add_member(vehicle_manager, leader=False)

            self.world.tick()
            destination = carla.Location(x=platoon['destination'][0],
                                         y=platoon['destination'][1],
                                         z=platoon['destination'][2])

            platoon_manager.set_destination(destination)
            platoon_manager.update_member_order()
            platoon_list.append(platoon_manager)

        return platoon_list

    def create_rsu_manager(self, data_dump):
        """
        Create a list of RSU.

        Parameters
        ----------
        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        rsu_list : list
            A list contains all rsu managers..
        """
        print('Creating RSU.')
        rsu_list = []
        for i, rsu_config in enumerate(
                self.scenario_params['scenario']['rsu_list']):
            rsu_config = OmegaConf.merge(self.scenario_params['rsu_base'],
                                         rsu_config)
            rsu_manager = RSUManager(self.world, rsu_config,
                                     self.carla_map,
                                     self.cav_world,
                                     self.scenario_params['current_time'],
                                     data_dump, save_path=self.save_path)

            rsu_list.append(rsu_manager)

        return rsu_list

    def spawn_vehicles_by_list(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given list.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """

        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '0, 255, 0'
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        ego_vehicle_bp = blueprint_library.find(default_model)

        for i in range(traffic_config['vehicle_list']):

            if not traffic_config['random']:
                ego_vehicle_bp.set_attribute('color', color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)

                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            vehicle = self.spawn_vehicle(ego_vehicle_bp)
            vehicle.set_autopilot(True, 8000)

            # if 'vehicle_speed_perc' in vehicle_config:
            #     tm.vehicle_percentage_speed_difference(
            #         vehicle, vehicle_config['vehicle_speed_perc'])
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            bg_list.append(vehicle)

        return bg_list

    def spawn_vehicle_by_range(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '0, 255, 0'
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        ego_vehicle_bp = blueprint_library.find(default_model)

        spawn_ranges = traffic_config['range']
        spawn_set = set()
        spawn_num = 0

        for spawn_range in spawn_ranges:
            spawn_num += spawn_range[6]
            x_min, x_max, y_min, y_max = \
                math.floor(spawn_range[0]), math.ceil(spawn_range[1]), \
                math.floor(spawn_range[2]), math.ceil(spawn_range[3])

            for x in range(x_min, x_max, int(spawn_range[4])):
                for y in range(y_min, y_max, int(spawn_range[5])):
                    location = carla.Location(x=x, y=y, z=0.3)
                    way_point = self.carla_map.get_waypoint(location).transform

                    spawn_set.add((way_point.location.x,
                                   way_point.location.y,
                                   way_point.location.z,
                                   way_point.rotation.roll,
                                   way_point.rotation.yaw,
                                   way_point.rotation.pitch))
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        while count < spawn_num:
            if len(spawn_list) == 0:
                break

            coordinates = spawn_list[0]
            spawn_list.pop(0)

            spawn_transform = carla.Transform(carla.Location(x=coordinates[0],
                                                             y=coordinates[1],
                                                             z=coordinates[2] + 0.3),
                carla.Rotation(
                roll=coordinates[3],
                yaw=coordinates[4],
                pitch=coordinates[5]))
            if not traffic_config['random']:
                ego_vehicle_bp.set_attribute('color', color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)
                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            vehicle = \
                self.world.try_spawn_actor(ego_vehicle_bp, spawn_transform)

            if not vehicle:
                continue

            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            if 'ignore_lights_percentage' in traffic_config:
                tm.ignore_lights_percentage(vehicle,
                                            traffic_config[
                                                'ignore_lights_percentage'])

            # each vehicle have slight different speed
            tm.vehicle_percentage_speed_difference(
                vehicle,
                traffic_config['global_speed_perc'] + random.randint(-30, 30))

            bg_list.append(vehicle)
            count += 1

        return bg_list

    def create_traffic_carla(self):
        """
        Create traffic flow (vehicles + pedestrians).

        Returns
        -------
        tm : carla.traffic_manager
            Carla traffic manager.

        bg_list : list
            The list that contains all background traffic vehicles and pedestrians.
        """
        print('Spawning CARLA traffic flow.\n')
        traffic_config = self.scenario_params['carla_traffic_manager']
        tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        vehicles, pedestrians = [], []

        # Vehicles
        if isinstance(traffic_config['vehicle_list'], int) or \
                isinstance(traffic_config['vehicle_list'], ListConfig):
            vehicles = self.spawn_vehicles_by_list(tm, traffic_config, vehicles)
        else:
            vehicles = self.spawn_vehicle_by_range(tm, traffic_config, vehicles)

        pedestrians = self.spawn_pedestrians(traffic_config['pedestrian_list'], pedestrians)
        # # Pedestrians
        # if 'pedestrian_list' in traffic_config and traffic_config['pedestrian_list'] > 0:
        #

        print('CARLA traffic flow generated.')
        self.vehicles = vehicles
        self.pedestrians = pedestrians
        return tm

    def spawn_pedestrians(self, num_pedestrians, bg_list):
        """
        Spawn pedestrians with AI controllers using batch operations.

        Parameters
        ----------
        num_pedestrians : int
            Number of pedestrians to spawn.

        bg_list : list
            Background traffic list to extend.

        Returns
        -------
        bg_list : list
            Updated list including pedestrians + controllers.
        """
        if num_pedestrians <= 0:
            return bg_list

        blueprint_library = self.world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        # Import CARLA command classes
        SpawnActor = carla.command.SpawnActor

        # Generate spawn points
        spawn_points = []
        for i in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(loc))

        print(f"Found {len(spawn_points)} valid spawn points for pedestrians")

        # 1. Spawn walkers using batch operations
        batch = []
        walker_speeds = []

        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bps)

            # Configure walker attributes
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # Set walker speed
            if walker_bp.has_attribute('speed'):
                # Use walking speed (index 1) - you can change this logic
                speed = walker_bp.get_attribute('speed').recommended_values[1]
                walker_speeds.append(float(speed))
            else:
                walker_speeds.append(1.4)  # default walking speed

            batch.append(SpawnActor(walker_bp, spawn_point))

        # Execute batch spawn for walkers
        walker_results = self.client.apply_batch_sync(batch, True)

        # Collect successfully spawned walkers
        walker_ids = []
        final_speeds = []

        for i, result in enumerate(walker_results):
            if result.error:
                print(f"Failed to spawn walker: {result.error}")
            else:
                walker_ids.append(result.actor_id)
                final_speeds.append(walker_speeds[i])
                bg_list.append(result.actor_id)  # Add walker ID to bg_list

        print(f"Successfully spawned {len(walker_ids)} walkers")

        # 2. Spawn controllers using batch operations
        batch = []
        for walker_id in walker_ids:
            batch.append(SpawnActor(controller_bp, carla.Transform(), walker_id))

        # Execute batch spawn for controllers
        controller_results = self.client.apply_batch_sync(batch, True)

        # Collect controller IDs
        controller_ids = []
        for i, result in enumerate(controller_results):
            if result.error:
                print(f"Failed to spawn controller: {result.error}")
            else:
                controller_ids.append(result.actor_id)
                bg_list.append(result.actor_id)  # Add controller ID to bg_list

        print(f"Successfully spawned {len(controller_ids)} controllers")

        # 3. Wait for tick to ensure all actors are ready
        try:
            self.world.tick()
        except:
            self.world.wait_for_tick()

        # 4. Configure controller behavior
        all_actors = self.world.get_actors(controller_ids)

        for i, controller in enumerate(all_actors):
            try:
                # Start the controller
                controller.start()

                # Set destination
                destination = self.world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)

                # Set speed (use index to get corresponding speed)
                if i < len(final_speeds):
                    controller.set_max_speed(final_speeds[i])
                else:
                    controller.set_max_speed(1.4)  # fallback speed

            except Exception as e:
                print(f"Error configuring controller {i}: {e}")

        print(f"Configured {len(all_actors)} pedestrian controllers")

        return bg_list

    def tick(self):
        """
        Tick the server.
        """
        self.world.tick()

    def destroy_actors(self):
        """
        Destroy only vehicles, pedestrians, and sensors.
        """
        self.client.set_timeout(0.1)
        actor_list = self.world.get_actors()
        v, p = 0, 0
        for actor in actor_list:
            if not actor.is_alive:
                continue

            tid = actor.type_id
            # Only allow vehicles, pedestrians, and sensors
            if tid.startswith("vehicle."):
                v += 1
            if tid.startswith("walker."):
                p += 1
            if tid.startswith("vehicle.") or tid.startswith("walker.") or tid.startswith("sensor.") or tid.startswith("controller."):
                actor.destroy()
        print(f"Number vehicles {v}")
        print(f"Number pedestrians {p}")

    def close(self):
        """
        Simulation close.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)
