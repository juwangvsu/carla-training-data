#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.

Log:
    9/9/22 still trying to figure out the time/tick related that affect sensor data rate
    settings.fixed_delta_seconds = 0.1, simulator runs 10hz, if lidar rotation_frequency=10,
    we will get a full scan, if fixed_delta_seconds = 0.05, we will get half scan.
    lidar resolution is set by points_per_second

"""

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

import carla
import argparse
import random
import time
import numpy as np
from config import clean_vehs
from populate import populate_veh, populate_walkers, populate_walkers_fromfile


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_i
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
    from pygame.locals import K_z
    from pygame.locals import K_BACKSPACE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
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

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, args):
        self.surface = None
        self.world = world
        self.args = args 
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0
        self.savedata=False

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(self.args.camera_resx))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))
            camera_bp.set_attribute('fov', str(self.args.fov))
            #verify camera proj matrix: f_u==f_v
            #f  = im_size_x /(2.0 * tan(fov * pi / 360))
            #Cx = im_size_x / 2.0
            #Cy = im_size_y / 2.0
            camera_bp.set_attribute('sensor_tick', str(0.1)) # set time btw camera captures, suppose to only affect camera, but ...

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)
            print('camera transform: ', camera.get_transform())

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            #lidar_bp.set_attribute('sensor_tick', str(0.1))

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            print('lidar transform: ', lidar.get_transform())

            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        print('camera callback ',image.frame, self.sensor.get_transform(), 'fov ',image.fov)
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.args.save_cam and self.savedata:
            print('camera save')
            self.capturecam = False
            print('save camera image: _out/image_2/%08d.png' % image.frame)
            image.save_to_disk('_out/image_2/%08d' % image.frame)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):

        #print('lidar callback', image.frame, self.sensor.get_transform(), 'point count ', image.get_point_count(1))
        totalpts=0
        for i in range(64):
            totalpts = totalpts + image.get_point_count(i)
        print('point count ', totalpts)

        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        if self.args.save_cam and self.savedata:
            print('lidar save')
            self.capturelidar = False
            lidar_fname='_out/velodyne/%08d.bin' % image.frame
            lidar_pcdfname='_out/velodyne/pcd/%08d.pcd' % image.frame
            save_lidar_data(lidar_fname, lidar_pcdfname, points, "bin")

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

class BasicController(object):
    def __init__(self, args, client, car):
        self.client =client 
        self.world = None
        self.camera = None
        self.car = car 
        self.args = args 
        self.info = False 
        self.savedata = False #toggle by 'z'

        self.display = None
        self.image = None
        self.capture = True

    def control2(self):
        #print('control2')
        control = self.car.get_control()
        control.throttle = 0
        keys = pygame.key.get_pressed()
        #print('key w ' , keys[K_w])
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE or event.key == K_q:
                    return True
                if event.key == K_z:
                    self.savedata = not self.savedata 
                    print('z save data ', self.savedata)
                if event.key == K_w:
                    print('w ')
                    control.throttle = 1
                    control.reverse = False
                elif event.key == K_s:
                    control.throttle = 1
                    control.reverse = True
                if event.key == K_SPACE:
                    control.throttle = 0
                if event.key == K_a:
                    control.steer = max(-1., min(control.steer - 0.05, 0))
                elif event.key == K_d:
                    control.steer = min(1., max(control.steer + 0.05, 0))
                else:
                    control.steer = 0
                if event.key == K_i:
                    self.info=True
                    control.steer = max(-1., min(control.steer - 0.05, 0))
        #control.hand_brake = K_SPACE
        self.car.apply_control(control)
        return False

    def control(self):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True
        print('control')
        control = self.car.get_control()
        control.throttle = 0
        if keys[K_w]:
            print('w ', keys[K_w])
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        self.car.apply_control(control)
        return False

def set_spectator(world, car):
        spectator = world.get_spectator()
        print('spectator: ', spectator)
        spec_trans = car.get_transform()
        spec_trans.location.z += 5
        spectator.set_transform(spec_trans)

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()
        settings = world.get_settings()

        if args.map is not None:
            print('load map %r.' % args.map)
            world = client.load_world(args.map)

        if args.ic:
            clean_vehs(args, client)

        if args.weather is not None:
            if not hasattr(carla.WeatherParameters, args.weather):
                print('ERROR: weather preset %r not found.' % args.weather)
            else:
                print('set weather preset %r.' % args.weather)
                world.set_weather(getattr(carla.WeatherParameters, args.weather))

        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = False
        traffic_manager = client.get_trafficmanager()
        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        spawn_point = town1_pos2
        populate_walkers_fromfile(client, world, traffic_manager, args.filterw, args.generationw, args.asynch, 20, args.seedw, False, playerlocation=spawn_point)

        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        #vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle = world.spawn_actor(bp, spawn_point)
        set_spectator(world, vehicle)

        vehicle_list.append(vehicle)
        #vehicle.set_autopilot(True)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height*2])
        controller = BasicController(args, client, vehicle)
        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)), vehicle, {}, display_pos=[0, 0])
        cam_sensor=SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=1.7), carla.Rotation(yaw=+00)), 
                      vehicle, {}, [0, 0], args)
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)), vehicle, {}, display_pos=[0, 2])
        #SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), vehicle, {}, display_pos=[1, 1])

        lidar_sensor=SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=1.7)), 
                      vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '1900000', 'rotation_frequency': '10'}, [1, 0], args)
        #SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0, z=2.4)), vehicle, {'channels' : '64', 'range' : '100', 'points_per_second': '100000', 'rotation_frequency': '20'}, display_pos=[1, 2])


        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        pygame_clock = pygame.time.Clock()
        while True:
            #print('\n\nmain loop tick')
            # Carla Tick
            if args.sync:
                world.tick()
            pygame_clock.tick_busy_loop(20)
            
            if controller.info:
                print('veh loc ', vehicle.get_transform())
                print('cam loc ', cam_sensor.sensor.get_transform())
                print('lidar loc ', lidar_sensor.sensor.get_transform())
                controller.info=False

            #controller.savedata toggle it value when key 'z'
            cam_sensor.savedata=controller.savedata
            lidar_sensor.savedata=controller.savedata

            # Render received data
            display_manager.render()

#            pygame.event.pump()
            if controller.control2():
                    return

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)


import parse_arg
def main():
    args=parse_arg.parse_arg()
    print(args)

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

import os
if __name__ == '__main__':
    os.system('rm _out/image_2/*')
    os.system('rm _out/velodyne/*')
    os.system('rm _out/velodyne/pcd/*')
    main()
    os.system('tar cvf _out.tar _out/')
