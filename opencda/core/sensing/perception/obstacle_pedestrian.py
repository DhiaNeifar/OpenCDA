# -*- coding: utf-8 -*-
"""
Obstacle pedestrian class to save object detection.
"""

import carla
import numpy as np
import open3d as o3d

import opencda.core.sensing.perception.sensor_transformation as st


class ObstaclePedestrian(object):
    """
    A class for obstacle pedestrian. The attributes are designed to match
    with carla.Walker class.

    Parameters
    ----------
    corners : nd.nparray
        Eight corners of the bounding box. shape:(8, 3).

    o3d_bbx : open3d.AlignedBoundingBox
        The bounding box object in Open3d. Mainly used for visualization.

    pedestrian : carla.Walker
        The carla.Walker object.

    lidar : carla.sensor.lidar
        The lidar sensor (optional).

    Attributes
    ----------
    bounding_box : carla.BoundingBox
        Bounding box of the pedestrian.

    location : carla.Location
        Location of the pedestrian.

    velocity : carla.Vector3D
        Velocity of the pedestrian.

    carla_id : int
        The pedestrian's id. Matches carla.Walker id. -1 if not set.
    """

    def __init__(self, corners=None, o3d_bbx=None,
                 pedestrian=None, lidar=None):

        if not pedestrian:
            # from raw bounding box corners
            from opencda.core.sensing.perception.bounding_box import BoundingBox
            self.bounding_box = BoundingBox(corners)
            self.location = self.bounding_box.location
            self.transform = None
            self.o3d_bbx = o3d_bbx
            self.carla_id = -1
            self.velocity = carla.Vector3D(0.0, 0.0, 0.0)
        else:
            self.set_pedestrian(pedestrian, lidar)

    def get_transform(self):
        return self.transform

    def get_location(self):
        return self.location

    def get_velocity(self):
        return self.velocity

    def set_carla_id(self, id):
        self.carla_id = id

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_pedestrian(self, pedestrian, lidar=None):
        """
        Assign the attributes from carla.Walker to ObstaclePedestrian.
        """
        self.location = pedestrian.get_location()
        self.transform = pedestrian.get_transform()
        self.bounding_box = pedestrian.bounding_box
        self.carla_id = pedestrian.id
        self.type_id = pedestrian.type_id

        self.set_velocity(pedestrian.get_velocity())

        # compute bounding box in lidar coords if lidar exists
        if lidar is None:
            return

        min_boundary = np.array([self.location.x - self.bounding_box.extent.x,
                                 self.location.y - self.bounding_box.extent.y,
                                 self.location.z + self.bounding_box.location.z
                                 - self.bounding_box.extent.z,
                                 1])
        max_boundary = np.array([self.location.x + self.bounding_box.extent.x,
                                 self.location.y + self.bounding_box.extent.y,
                                 self.location.z + self.bounding_box.location.z
                                 + self.bounding_box.extent.z,
                                 1])

        min_boundary = min_boundary.reshape((4, 1))
        max_boundary = max_boundary.reshape((4, 1))
        stack_boundary = np.hstack((min_boundary, max_boundary))

        # transform to lidar coordinates
        stack_boundary_sensor_cords = st.world_to_sensor(
            stack_boundary, lidar.get_transform())
        stack_boundary_sensor_cords[:1, :] = -stack_boundary_sensor_cords[:1, :]
        stack_boundary_sensor_cords = stack_boundary_sensor_cords[:-1, :]

        min_boundary_sensor = np.min(stack_boundary_sensor_cords, axis=1)
        max_boundary_sensor = np.max(stack_boundary_sensor_cords, axis=1)

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_boundary_sensor,
                                                   max_bound=max_boundary_sensor)
        aabb.color = (0, 1, 0)  # green for pedestrians
        self.o3d_bbx = aabb
