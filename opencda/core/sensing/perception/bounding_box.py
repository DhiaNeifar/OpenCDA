import numpy as np
import carla

class BoundingBox(object):
    """
    Bounding box class for obstacle vehicle.

    Params:
    -corners : nd.nparray
        Eight corners of the bounding box. (shape:(8, 3))
    Attributes:
    -location : carla.location
        The location of the object.
    -extent : carla.vector3D
        The extent of  the object.
    """

    def __init__(self, corners):
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        center_z = np.mean(corners[:, 2])

        extent_x = (np.max(corners[:, 0]) - np.min(corners[:, 0])) / 2
        extent_y = (np.max(corners[:, 1]) - np.min(corners[:, 1])) / 2
        extent_z = (np.max(corners[:, 2]) - np.min(corners[:, 2])) / 2

        self.location = carla.Location(x=center_x, y=center_y, z=center_z)
        self.extent = carla.Vector3D(x=extent_x, y=extent_y, z=extent_z)
