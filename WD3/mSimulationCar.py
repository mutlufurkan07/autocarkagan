# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:47:50 2020
@author: Ahmet Furkan Tavşancı
"""
import airsim
import time
import numpy as np
import os
import random
import threading
import cv2
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image


class mSimulationCar:
    def __init__(self, lidar_dim, api_control=True):

        self.client = airsim.CarClient()
        # ground_truth_env = self.client.simGetGroundTruthEnvironment(vehicle_name="PhysXCar")

        self.client.simSetObjectScale("PhysXCar", airsim.Vector3r(1, 1.4, 1))

        self.client.confirmConnection()
        self.client.enableApiControl(api_control)
        self.car_controls = airsim.CarControls()

        print("mCar is initializied and API Control enabled: %s" % self.client.isApiControlEnabled())
        self.current_movement_state = []
        self.init_time = time.time()
        self.current_pos_orientation = [0, 0, 0, 0, 0, 0]  # px py pz ox oy oz
        self.current_speed = 0
        self.max_throttle = 5
        self.max_brake = 1
        self.max_speed = 3  # 3m/s
        self.mFrame = np.zeros((256, 256, 1), dtype="uint8")
        self.mutex_Flag = False
        self.lidar_range = 32
        self.mCar_state = []
        self.RGB_image = []
        self.mCar_pos = []
        self.mCar_orientation = []
        self.current_LidarData = None
        self.lidar_data_buffer = []
        self.flatten_lidar_data = []
        self.timeStampsOfPosandOrientation = []
        self.mcurrentSimOutput = []
        self.visualizeLidar_flatten = True
        self.starting_time = time.time()
        self.current_time = self.starting_time
        self.data_dir = "data"
        self.iscurrentState_acollision = 0
        self.has_collided_flag = 0
        self.car_api_control_steer_flag = False
        self.car_api_steering_angle = 0
        self.car_steering_time = 0
        self.lidar_dim = lidar_dim
        self.arraytobesaved = np.zeros(self.lidar_dim + 5)
        self.target_pointer_array = np.zeros((1, 50))
        self.possible_car_states = None
        self.target_location = [0, 0]
        self.initial_x = 0
        self.initial_y = 0
        self.textFile = None
        self.currpos = [0, 0]

    def car_api_control_steer(self, normalized_steering, steering_time):
        self.car_api_steering_angle = normalized_steering
        self.car_steering_time = steering_time
        self.car_api_control_steer_flag = True

    def createFolder(self):
        file_index = 0
        current_dir = self.data_dir + "/" + str(file_index) + ".txt"

        while os.path.isfile(current_dir):
            file_index += 1
            current_dir = self.data_dir + "/" + str(file_index) + ".txt"

        self.textFile = open(self.data_dir + "/" + str(file_index) + ".txt", "w")

    def save_current_state_TXT(self, m_arr):
        np.savetxt(self.textFile, m_arr.reshape(1, 185))
        # print("Saved current state......")

    def neural_network_output(self, current_t, pos_x_value, pos_y_value, orientation_euler_z, collision_Flag):
        curr_timestamp = np.round(current_t - self.starting_time, 6)
        curr_pos_x = np.round(pos_x_value, 3)
        curr_pos_y = np.round(pos_y_value, 3)
        curr_euler_orientation_z = np.round(orientation_euler_z / np.pi * 180, 2)


        self.arraytobesaved[0] = curr_timestamp
        self.arraytobesaved[1] = curr_pos_x
        self.arraytobesaved[2] = curr_pos_y
        self.arraytobesaved[3] = curr_euler_orientation_z
        self.arraytobesaved[4:len(self.arraytobesaved) - 1] = (self.flatten_lidar_data / 255)
        self.arraytobesaved[len(self.arraytobesaved) - 1] = collision_Flag
        return self.arraytobesaved

    def setVehiclePose(self, pos_x, pos_y, rotation_z):
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(pos_x, pos_y, -0.5), airsim.utils.to_quaternion(0, 0, rotation_z / 57.2957795)),
            True)
        # print("Pose is set to x: %.2f meters y: %.2f meters  theta: %.2f degrees" % (pos_x , pos_y,rotation_z))

    def read_states(self):
        states = list(set(open("data/states.txt", "r").readlines()))
        for i in range(len(states)):
            states[i] = states[i].split(",")
        self.possible_car_states = np.array(states).astype(float)

    def randomly_initiate_states(self):
        random_index = random.randint(0, len(self.possible_car_states) - 1)
        self.setVehiclePose(self.possible_car_states[random_index, 0], self.possible_car_states[random_index, 1],
                            self.possible_car_states[random_index, 2])
        # self.setVehiclePose(0, 0, 90)
        self.initial_x = self.possible_car_states[random_index, 0]
        self.initial_y = self.possible_car_states[random_index, 1]
        # print(f"Vehicle Position : {self.possible_car_states[random_index, 0]} ,"
        #     f" {self.possible_car_states[random_index, 1]} ")
        target_random_index = random.randint(0, len(self.possible_car_states) - 1)
        if random_index == target_random_index:
            target_random_index -= 1
        self.target_location = [self.possible_car_states[target_random_index, 0],
                                self.possible_car_states[target_random_index, 1]]

    def collect_peripheralData(self):
        cv2.waitKey(1)

        # request current lidar data from unreal airsim and process the raw data
        self.current_LidarData = self.client.getLidarData()

        # self.depth_data = self.getScreenDepthVis()

        # self.get_semantics()
        # responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])
        # response = responses[0]
        # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # img_rgb = img1d.reshape(response.height, response.width, 3)
        # cv2.imshow("ksldkas", img_rgb)

        self.flatten_lidar_data = self.processLidarTo180Deg(self.visualizeLidar_flatten)

        # get current car state from unreal and airsim and convert them to proper metrics(pos,speed,heading etc.)
        self.mCar_state = self.__get_carState(False)
        pos = self.mCar_state.kinematics_estimated.position

        orientation_quaternion = self.mCar_state.kinematics_estimated.orientation
        self.mCar_pos.append([pos.x_val, pos.y_val, pos.z_val])
        orientation_euler_x, orientation_euler_y, orientation_euler_z = self.quaternion_to_eularian_angles(
            orientation_quaternion)

        self.mCar_orientation.append([orientation_euler_z / np.pi * 180])

        if self.client.simGetCollisionInfo().has_collided:
            self.has_collided_flag = 1

        # check a loop time for getting data
        t = time.time()
        self.timeStampsOfPosandOrientation.append(t - self.starting_time)

        # print current network input raw data

        # print("Current time elapsed msec:           %.2f msecs" % ((t - self.current_time) * 1000))
        # print("Current timestamp:                   %.2f sec  "  % (t - self.starting_time ))
        # print("Current position in xyz in meters:   %.2f , %.2f" %  (pos.x_val, pos.y_val))
        # print("Current Orientation:                 %.1f degrees" % (orientation_euler_z / np.pi * 180))
        # print("has_collided:                       " , self.client.simGetCollisionInfo().has_collided)
        self.mcurrentSimOutput = self.neural_network_output(t, pos.x_val, pos.y_val, orientation_euler_z,
                                                            self.has_collided_flag)

        self.current_time = t

        if self.car_api_control_steer_flag:
            if not self.mutex_Flag:
                self.go_steer(self.car_api_steering_angle, self.car_steering_time)
                self.car_api_control_steer_flag = False

    def write_Pose_OrientTXT(self):
        np.savetxt("Positions.txt", np.array(self.mCar_pos), fmt="%.1f")
        np.savetxt("Orientations.txt", np.array(self.mCar_orientation), fmt="%.1f")
        np.savetxt("Timestamps.txt", np.array(self.timeStampsOfPosandOrientation), fmt="%.6f")

    # plt.plot(np.array(self.timeStampsOfPosandOrientation)[:179] , diff(np.array(self.mCar_pos)[:180,0]) / diff(
    # np.array(self.timeStampsOfPosandOrientation)[:180] )) plt.show()

    def plot_car_pos(self):
        pos_values = np.array(self.mCar_pos)
        pos_values = pos_values[:, :2]
        plt.scatter(pos_values[:, 1], pos_values[:, 0])
        plt.title("Car Position")
        plt.show()
        return pos_values

    def quaternion_to_eularian_angles(self, q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)

        return pitch, roll, yaw

    # dont call from outside class
    def __get_carState(self, isPrint):
        self.mutex_Flag = True
        # time.sleep(1)

        car_state = self.client.getCarState()
        pos = car_state.kinematics_estimated.position
        self.current_speed = car_state.speed
        orientation_x, orientation_y, orientation_z = self.quaternion_to_eularian_angles(
            car_state.kinematics_estimated.orientation)

        if isPrint:
            print("Speed,PX,PY,OX,OY")
            print("%d   ,%f.2,%f.2,%f.2,%f.2" % \
                  (car_state.speed, pos.x_val, pos.y_val,
                   orientation_x, orientation_y))

        self.mutex_Flag = False
        return car_state

    def get_semantics(self):
        response = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])[0]

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array H X W X 3

        cv2.imshow("segmentation", img_rgb)
        return img_rgb

    def getScreenDepthVis(self):

        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        img2d = img2d[5 * responses[0].height//10: 6 * responses[0].height//10, :]

        image = cv2.equalizeHist(img2d.astype(np.uint8))
        mean_image = np.mean(img2d, 0)


        # image1 = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        # factor = 2
        # maxIntensity = 255.0  # depends on dtype of image data
        #
        # # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark
        # newImage1 = (maxIntensity) * (image1 / maxIntensity) ** factor
        # newImage1 = np.array(newImage1, dtype=np.uint8)
        # cv2.imshow("Other", newImage1)
        # cv2.waitKey(0)

        return mean_image

    def reset_car_controls(self):
        self.car_controls.throttle = 0
        self.car_controls.brake = 0
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)

    def go_steer(self, normalized_steering_angle, mtime):
        self.mutex_Flag = True

        self.car_controls.steering = normalized_steering_angle

        if self.current_speed < self.max_speed:
            self.car_controls.throttle = self.max_throttle
            self.client.setCarControls(self.car_controls)
        else:
            self.car_controls.throttle = 0
            self.client.setCarControls(self.car_controls)

        # self.reset_car_controls()
        self.mutex_Flag = False

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))

        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        points = points[:, :2]
        mask = np.where(points[:, 0] > 0)[0]
        all_Y = points[:, 0][mask]
        all_X = points[:, 1][mask]

        return np.append(all_Y.reshape((len(all_Y), 1)), all_X.reshape((len(all_X), 1)), axis=1)

    def processLidarTo180Deg(self, visualize):

        if len(self.current_LidarData.point_cloud) < 3:
            print("\tNo points received from Lidar data")
            return np.ones(self.lidar_dim) * self.lidar_range + np.random.normal(0, 0.08, (self.lidar_dim))
        else:
            points = self.parse_lidarData(self.current_LidarData)
            all_X = points[:, 1]
            all_Y = points[:, 0]
            dist = np.sqrt(all_X * all_X + all_Y * all_Y)
            at2 = np.arctan2(all_Y, all_X) * 180 / np.pi
            at2 = np.around(at2).astype(int)
            vector = np.ones(180) * self.lidar_range
            vector[at2 - 1] = dist
            ret_arr = np.flip(vector)
            ret_arr = 255 - ((ret_arr / self.lidar_range) * 255)
            ret_arr += np.random.normal(0, 0.1, ret_arr.shape)
            ret_arr = ret_arr[(180-self.lidar_dim)//2:180-(180-self.lidar_dim)//2]

            if visualize:
                cv_arr = np.zeros((80, self.lidar_dim), np.uint8)

                for ii in range(80):
                    cv_arr[ii, :] = ret_arr
                cv_arr = cv2.resize(cv_arr, (360, 80))
                cv2.imshow("Flatten lidar", cv_arr)
                # cv2.waitKey(1)

            return ret_arr

    def request_current_lidar(self, visualize):
        if len(self.current_LidarData.point_cloud) < 3:
            print("\tNo points received from Lidar data")
        else:
            points = self.parse_lidarData(self.current_LidarData)
            # print("Received Lidar Points" , points.shape)
            points = points * 100

            points[:, 1] = points[:, 1] + 24000
            points = points / 40

            points = points.astype(int)
            if (visualize):
                frame = np.zeros((410, 810))
                for ii in range(len(points)):
                    point_y = points[ii, 0]
                    if point_y > 200:
                        dist = point_y - 200
                        point_y = point_y - 2 * dist
                    else:
                        dist = 200 - point_y
                        point_y = point_y + 2 * dist

                    if point_y < 400:
                        frame = cv2.circle(frame, (points[ii, 1], point_y), 5, (127, 0, 0), -1)
                frame = cv2.circle(frame, (400, 400), 5, (50, 0, 0), 2)
                # cv2.imshow("Car Lidar_opencv", frame)
                # cv2.waitKey(1)

        return self.current_LidarData

    def createTargetArray(self, target_pos_x, target_pos_y, curr_heading, distance_to_target):
        target_positional_orientation = np.arctan2(target_pos_y, target_pos_x)*180/np.pi
        target_relative_theta = (target_positional_orientation - curr_heading)

        if target_relative_theta < -180:
            target_relative_theta += 360
        if target_relative_theta > 180:
            target_relative_theta -= 360

        target_array = np.zeros((60,))
        ratio1 = np.sqrt(target_pos_x**2 + target_pos_y**2)
        ratio2 = np.sqrt((self.target_location[0] - self.initial_x)**2 + (self.target_location[1] - self.initial_y)**2 )
        ratio = ratio1 / (ratio2 + 1e-8)
        ratio = np.clip(1 - ratio, 0.5, 1)
        if -180 < int(target_relative_theta) < -83:
            target_array[0:5] = np.ones((5,)) * ratio
        elif 81 < int(target_relative_theta) < 181:
            target_array[-5:] = np.ones((5,)) * ratio
        else:
            target_array[int((target_relative_theta) // 3) + 28:int((target_relative_theta) // 3) + 33] = np.ones(
                (5,)) * ratio


        # target_array += np.random.normal(0, 0.03, target_array.shape)
        target_array[np.where(target_array > 1)[0]] = 1
        cv_arr = target_array.reshape((60, 1))
        cv_arr = cv2.resize(cv_arr, (80, 360))
        cv2.imshow("Pos target array", cv_arr.T)

        return target_array

        # target_positional_orientation = np.arctan2(target_pos_y, target_pos_x) * 180 / np.pi
        # target_relative_theta = (target_positional_orientation - curr_heading)
        # if target_relative_theta < -180:
        #     target_relative_theta += 360
        # if target_relative_theta > 180:
        #     target_relative_theta -= 360
        # print("positional_orientation " , target_positional_orientation)
        # target_array = np.zeros((60,))
        # x_goal = target_pos_x + self.currpos[0]
        # y_goal = target_pos_y + self.currpos[1]

        # print("Target relative data" , target_relative_theta)
        # ratio = (np.sqrt(target_pos_x**2 + target_pos_y**2)) / (np.sqrt((x_goal - self.xx)**2 + (y_goal - self.yy)**2))
        # ratio = 1 - ratio
        # ratio = np.clip(ratio, 0.2 , 1)

        # try:
        #     if (-180 < int(target_relative_theta) < -83):
        #         target_array[0:5] = np.ones((5,))
        #
        #     elif (81 < int(target_relative_theta) < 181):
        #         target_array[-5:] = np.ones((5,))
        #     else:
        #         target_array[int((target_relative_theta) // 3) + 28:int((target_relative_theta) // 3) + 33] = np.ones(
        #             (5,))
        #
        #     target_array += np.random.normal(0, 0.03, target_array.shape)
        #     target_array[np.where(target_array > 1)[0]] = 1
        #     cv_arr = target_array.reshape((60, 1))
        #     cv_arr = cv2.resize(cv_arr, (80, 360))
        #     cv2.imshow("Pos target array", cv_arr.T)
        # except:
        #     print("Exceptionnnnn")
        #
        # return target_array

    def take_action_and_collect_data(self):
        self.collect_peripheralData()
        self.request_current_lidar(True)
        return self.mcurrentSimOutput.astype(np.float32)

    def car_cosineReward(self, target_x, target_y, curr_x, curr_y, curr_heading):
        heading_vec = np.array([math.cos(curr_heading / 180 * math.pi), math.sin(curr_heading / 180 * math.pi)])
        target_vector = np.array([target_x - curr_x, target_y - curr_y])
        return (np.dot(heading_vec, target_vector)) / (
                np.linalg.norm(target_vector) * np.linalg.norm(heading_vec))  # reward_input

    def get_current_lidar_and_targetArray(self, TARGET_POS_X, TARGET_POS_Y, target_initial_distance):
        car_pos_lidar_data = self.take_action_and_collect_data()

        curr_x = car_pos_lidar_data[1]
        curr_y = car_pos_lidar_data[2]
        self.currpos = [curr_x, curr_y]

        curr_heading = car_pos_lidar_data[3]
        isCollidedFlag = car_pos_lidar_data[len(car_pos_lidar_data) - 1]

        lidar_data_sampled = car_pos_lidar_data[4:len(car_pos_lidar_data) - 1]

        lidar_right = lidar_data_sampled[3:]
        lidar_left = lidar_data_sampled[:self.lidar_dim-3]

        lidar_data_sampled[:self.lidar_dim-3] = np.maximum.reduce([lidar_data_sampled[:self.lidar_dim-3], lidar_right, lidar_left])

        distance_to_target = np.sqrt((TARGET_POS_X - curr_x) ** 2 + (TARGET_POS_Y - curr_y) ** 2)
        car_target_array_data = self.createTargetArray(TARGET_POS_X - curr_x, TARGET_POS_Y - curr_y, curr_heading,
                                                       distance_to_target / target_initial_distance)

        state_numpy = np.append(lidar_data_sampled, car_target_array_data)
        state_tensor = torch.from_numpy(state_numpy)

        middle_lidar_point = car_pos_lidar_data[54:134]
        is_clear = np.max(middle_lidar_point)

        return state_numpy, state_tensor, curr_x, curr_y, curr_heading, isCollidedFlag, is_clear