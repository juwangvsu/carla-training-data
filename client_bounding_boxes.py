#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
from populate import populate_veh, populate_walkers, populate_walkers_fromfile
import numpy
import numpy.random
from config import clean_vehs
from carla import ColorConverter as cc
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_BACKSPACE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#stuff from carla-training-data
from dataexport_913 import * #save_lidar_data...
town5_pos1 =carla.Transform(carla.Location(x=-77.074196, y=103.549164, z=40.434952), carla.Rotation(pitch=-90.999847, yaw=0.770454, roll=0.000098))
town5_pos2 =carla.Transform(carla.Location(x=-75.074196, y=145.549164, z=0.434952), carla.Rotation(pitch=0, yaw=0, roll=0.000098))
#town5_pos2 =carla.Transform(carla.Location(x=-75.074196, y=90.549164, z=0.434952), carla.Rotation(pitch=0, yaw=0, roll=0.000098))
town1_pos2 =carla.Transform(carla.Location(x=213.88, y=59.90, z=0.434952), carla.Rotation(pitch=0, yaw=0, roll=0.000098))

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, width, height):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((width, height))
        #bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self, args):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.lidar_img = None
        self.lidar_surface = None
        self.surface = None
        self.args = args 
        self.capturelidar = True
        self.capturecam = True


    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.args.width))
        camera_bp.set_attribute('image_size_y', str(self.args.height))
        #camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        #camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        #camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.ford.mustang')[0]
        #car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        spawn_point.location.x = 213.88
        spawn_point.location.y = 59.90
        spawn_point.location.z = 0.3
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        spawn_point.rotation.yaw = 0
        spawn_point = town1_pos2
        self.car = self.world.spawn_actor(car_bp, spawn_point)
        return spawn_point

    def setup_lidar(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '50')
        bp.set_attribute('points_per_second', str(self.args.lidar_pps))
        #bp.set_attribute('points_per_second', '560000')
        #bp = world.get_blueprint_library().find('sensor.other.imu')
        bound_x = 0.5 + self.car.bounding_box.extent.x
        bound_y = 0.5 + self.car.bounding_box.extent.y
        bound_z = 0.5 + self.car.bounding_box.extent.z

        #lidar_trans=carla.Transform()
        #lidar_trans.location.x=0
        #lidar_trans.location.z=3
        lidar_trans = carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0))
        attch_t = carla.AttachmentType.SpringArm
        self.lidar_sensor = self.world.spawn_actor(
            bp,
            lidar_trans,
            attach_to=self.car,
            attachment_type=attch_t)
        weak_self = weakref.ref(self)
        self.lidar_sensor.listen(lambda sensor_data: weak_self()._parse_lidar(weak_self, sensor_data))

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car, attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        #self.camera.listen(lambda image: weak_self()._parse_image(weak_self, image))
        self.camera.listen(lambda image: BasicSynchronousClient._parse_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = self.args.width / 2.0
        calibration[1, 2] = self.args.height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.args.width / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        #calibration[0, 2] = VIEW_WIDTH / 2.0
        #calibration[1, 2] = VIEW_HEIGHT / 2.0
        #calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
    #################################################################33
    ## another way to process key and pygame events
    def parse_events(self, car):
        control = car.get_control()
        control.throttle = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_BACKSPACE:
                    car.set_autopilot(False)
                    #world.restart()
                    car.set_autopilot(True)
                elif event.key == K_w:
                    print('w ', K_w)
                    control.throttle = 1
                    control.reverse = False
                elif event.key == K_s:
                    print('s ', K_s)
                    control.throttle = 1
                    control.reverse = True
        car.apply_control(control)

    #################################################################33
    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True
        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            print('w ', keys[K_w])
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            print('s ', keys[K_w])
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            print('a ', keys[K_w])
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            print('d ', keys[K_w])
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False
    #def _parse_lidar(weak_self, img):
    #    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    #    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    #    lidar_data = np.array(points[:, :2])
    #    lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
    #    lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
    #    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    #    lidar_data = lidar_data.astype(np.int32)
    #    lidar_data = np.reshape(lidar_data, (-1, 2))
    #    lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
    #    lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
    #    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    #    self.lidar_surface = pygame.surfarray.make_surface(lidar_img)

    @staticmethod
    def _parse_lidar(weak_self, img):
        #return
        self = weak_self()
        #print('lidar callback')
        #t_start = self.timer.time()

        disp_size = [int(self.args.width), int(self.args.height)]
        
        lidar_range = 50
        #lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(img.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / (2*lidar_range)
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        self.lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        self.lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        self.lidar_surface = pygame.surfarray.make_surface(self.lidar_img)
        if self.args.save_cam and self.capturelidar:
            print('lidar save')
            self.capturelidar = False
            lidar_fname='_out/velodyne/%08d.bin' % img.frame
            lidar_pcdfname='_out/velodyne/pcd/%08d.pcd' % img.frame
            save_lidar_data(lidar_fname, lidar_pcdfname, points, "bin")
        return

    @staticmethod
    def _parse_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if not self:
            return
        if not img.frame%5==0:
            return
        print('camera callback ', img.frame)
        self.image = img
        img.convert(cc.Raw)
        array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.image.height, self.image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.args.save_cam and self.capturecam:
            print('camera save')
            self.capturecam = False
            print('save camera image: _out/%08d.png' % img.frame)
            img.save_to_disk('_out/%08d' % img.frame)

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.surface is not None:
            print('render cam...')
            display.blit(self.surface, (0, 0))
            self.surface=None

        if self.lidar_surface is not None:
            print('render lidar...')
        #    self.lidar_surface = pygame.surfarray.make_surface(self.lidar_img)
            display.blit(self.lidar_surface, (0, int(self.args.height)))

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()
            pygame.font.init()
            print('host is" ', self.args.host)
            self.client = carla.Client(self.args.host, 2000)
            self.client.set_timeout(10.0)
            print('load map %r.' % self.args.map)
            #self.world = self.client.get_world()
            self.world = self.client.load_world(self.args.map) #load map take 10 secs
            if self.args.map is not None:
                print('load map %r.' % self.args.map)
                self.world = self.client.load_world(self.args.map)

            if self.args.ic:
                clean_vehs(self.args, self.client)

            self.traffic_manager = self.client.get_trafficmanager()

            if self.args.weather is not None:
                if not hasattr(carla.WeatherParameters, self.args.weather):
                    print('ERROR: weather preset %r not found.' % self.args.weather)
                else:
                    print('set weather preset %r.' % self.args.weather)
                    self.world.set_weather(getattr(carla.WeatherParameters, self.args.weather))

            spawn_point=self.setup_car()
            self.setup_camera()
            self.setup_lidar()
            populate_walkers_fromfile(self.client, self.world, self.traffic_manager, self.args.filterw, self.args.generationw, self.args.asynch, 10, self.args.seedw, False, playerlocation=spawn_point)

            self.display = pygame.display.set_mode((self.args.width, 2*self.args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            #self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0,0,0))
            pygame.display.flip()

            spectator = self.world.get_spectator()
            print('spectator: ', spectator)
            spec_trans = self.car.get_transform()
            spec_trans.location.z += 5
            spectator.set_transform(spec_trans)

            if self.args.fps is not None:
                settings = self.world.get_settings()
                settings.fixed_delta_seconds = (1.0 / self.args.fps) if self.args.fps > 0.0 else 0.0
                self.world.apply_settings(settings)

            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(False) # if set true, this require the main loop to call world.tick() to drive the game
            vehicles = self.world.get_actors().filter('vehicle.*')
            if self.args.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()
            self._server_clock = pygame.time.Clock()

            while True:
                if self.args.sync:
                    print('args.sync= ', self.args.sync)
                    self.world.tick()

                self.capturecam = True
                self.capturelidar = True
                pygame_clock.tick_busy_loop(20)

                if self.control(self.car):
                    return
                self.render(self.display)
                #bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                #ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes, self.args.width, self.args.height)

                pygame.display.flip()
            #    self._server_clock = pygame.time.Clock()
            #    self._server_clock.tick()
                pygame.event.pump()
                #if self.control(self.car):
                #    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

import parse_arg

def main():
    """
    Initializes the client-side bounding box demo.
    """
    args=parse_arg.parse_arg()
    print(args)
    try:
        client = BasicSynchronousClient(args)
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
