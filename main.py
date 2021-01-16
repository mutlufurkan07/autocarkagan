import numpy as np
import time
from airsim import Vector3r
import torch
from Agent import Agent
from mSimulationCar import mSimulationCar
import collections


if __name__ == "__main__":
    # Simulation params
    action_duration = 0.02  # secs
    TARGET_POS_X, TARGET_POS_Y = 0, -10
    target_initial_distance = np.sqrt((TARGET_POS_X ** 2) + (TARGET_POS_Y ** 2))

    experiment_id = 35
    SUCCESS_RADIUS = 5  # 15m close to reward
    MAX_EPOCH = 5e6
    MAX_TIME = 50  # secs
    gamma = 0.995
    tau = 0.01
    actorlr = 1e-5
    criticlr = 1e-4
    variance = 0.04
    action_dim = 1
    mem_size = 1e6  # 80000
    batch_size = 64
    reward_dim = 6  # step_reward = -0.1 , collusion_reward = -20 , angle_compensation_reward = 0.1 ,  target_accomplised_reward = 50  ###cumulative_Success_reward = 1 ?!?!?!
    state_dim = 245
    circle_num = 10
    eps_dec = 0.9  # kkullanımıoyrz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_choose_array = np.zeros((5, 5))
    map_choose_array[0, :] = range(1, 6)

    # inital unreal x = -18978  unreak y = -3646
    map_choose_array[1:, 0] = [0, 0, 67, -40]  # start_x,y  target_x,y of map 1  newmap->
    map_choose_array[1:, 1] = [0, 43, 67, 53]  # start_x,y  target_x,y of map 2
    map_choose_array[1:, 2] = [0, 111, 67, 148]  # start_x,y  target_x,y of map 3
    map_choose_array[1:, 3] = [0, 308, 104, 308]  # start_x,y  target_x,y of map 4
    map_choose_array[1:, 4] = [0, 399, 104, 399]  # start_x,y  target_x,y of map 5
    map_index_curr = 1
    random_array = np.arange(5)
    environment_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # %%

    mCar = mSimulationCar()
    training_flag = True

    if not training_flag:
        MAX_TIME = 500

    mCar.read_states()
    # mCar.createFolder() #call this to save the data to a text file ./data/....txt
    print("Starting")
    agent = Agent(gamma, tau, actorlr, criticlr, variance, action_dim, int(mem_size), batch_size, state_dim, reward_dim,
                  eps_dec)
    #agent.load_models(70200, 31)

    # agent.load_models(42000,253)

    f = open("data/log.txt", "w")
    f.close()

    car_x = 0
    car_y = 0

    training_starting_time = time.time()
    tot_step = 0
    average_reward_list = collections.deque(maxlen=50)

    for epoch in range(int(MAX_EPOCH)):

        car_steering_old = 0
        steering_que = collections.deque(maxlen=5)
        [steering_que.append(0.0) for i in range(5)]
        epoch_reward = 0
        circle_reward_arr = np.ones((circle_num, 1))
        step_len = 0
        curr_timee = time.time()
        success = 0

        circle_r_length = np.sqrt(((TARGET_POS_X - car_x) ** 2 + (TARGET_POS_Y - car_y) ** 2)) / (circle_num + 1)

        state_numpy, state_torch_tensor, curr_x, curr_y, curr_heading, isCollidedFlag, isClear_flag = mCar.take_action_and_get_current_lidar_and_targetArray(
            TARGET_POS_X, TARGET_POS_Y, target_initial_distance)

        state_torch_tensor = torch.cat((state_torch_tensor, torch.tensor(steering_que)))
        current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, curr_x, curr_y, curr_heading)
        state_torch_tensor.to(device)

        while 1:
            # check success condition first
            if np.sqrt((curr_x - TARGET_POS_X) ** 2 + (curr_y - TARGET_POS_Y) ** 2) < SUCCESS_RADIUS:
                print("\nSUCCCESSSSSSSSSSSSSSSSSSSSSS\n")
                success = 1
            # check time punishment flag
            time_pounishmet_flag = (MAX_TIME < time.time() - curr_timee)

            '''
            forward and get Q values for debug purposes,,, to be deleted
            '''
            # actor_QValues = agent.dqn_target.forward(state_torch_tensor.float())

            agent_action = agent.action_selection(state_torch_tensor.float())
            car_steering_old += agent_action / 10
            car_steering_old = min(car_steering_old, 0.75)
            car_steering_old = max(car_steering_old, -0.75)
            # print("steering  angle" , car_steering_old)
            mCar.car_api_control_steer(car_steering_old, action_duration)
            steering_que.append(car_steering_old)

            new_state_numpy, new_state_torch_tensor, new_curr_x, new_curr_y, new_curr_heading, isCollidedFlag, isClear_flag_new = mCar.take_action_and_get_current_lidar_and_targetArray(
                TARGET_POS_X, TARGET_POS_Y, target_initial_distance)
            new_state_torch_tensor = torch.cat((new_state_torch_tensor, torch.tensor(steering_que)))
            new_state_torch_tensor.to(device)
            new_current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, new_curr_x, new_curr_y,
                                                              new_curr_heading)
            # mCar.client.simPause(True)

            ####  TODO: create a function for this algorithm
            current_distance_to_target = np.sqrt((TARGET_POS_X - new_curr_x) ** 2 + (TARGET_POS_Y - new_curr_y) ** 2)
            curr_Region = int(current_distance_to_target / circle_r_length)
            temp_circle_reward = 0
            if not curr_Region > circle_num:
                temp_circle_reward = circle_reward_arr[curr_Region - 1, 0]
                # if (temp_circle_reward):
                #         # print("temp_circle_rewa" , temp_circle_reward)
                #         print( "current region" , curr_Region)
                circle_reward_arr[curr_Region - 1] = 0
                ####

            # update network
            reward_tensor = torch.tensor([-0.075, current_cosine_reward / 2000, (isCollidedFlag or time_pounishmet_flag) * (-200), success * 400,
                 temp_circle_reward * 25, isClear_flag * -1])
            # print(f"CCS: {current_cosine_reward}, TSR:  {temp_circle_reward*25},  ICR:{isClear_flag*-1}")

            if training_flag:
                agent.learn(state_torch_tensor, agent_action, reward_tensor, new_state_torch_tensor, isCollidedFlag)
            else:
                time.sleep(0.003)

            # DONMEK KOTUDUR ve düz gitmek iyidir
            if np.abs(agent_action) < 0.25:
                epoch_reward += reward_tensor.sum() + 2 * torch.tensor(1 - np.abs(agent_action))
            else:
                epoch_reward += reward_tensor.sum() - 1 * torch.tensor(np.abs(agent_action))

            step_len += 1
            # if time_current >= MAX_TIME or done or (success == 10):
            if (time_pounishmet_flag or isCollidedFlag or success):
                average_reward_list.append(epoch_reward)
                avg_reward = sum(average_reward_list) / len(average_reward_list)
                tot_step += step_len
                print(
                    f"Epoch: {epoch} , Reward: {epoch_reward.item():.3f} , Avg Reward: {avg_reward.item():.3f}  , Success: {success} , Step Len: {step_len}  , Total Steps: {tot_step}")
                # print("Epoch " ,epoch , "Reward:" , epoch_reward.item(), "Avg Reward: " , avg_reward.item()  , " Success: " , success , " Step len: ", step_len , "Total Steps: ", tot_step)

                # rand_rotation_deg = np.random.randint(-70,70)
                # rand_x = np.random.randint(-5,30)
                # rand_y = np.random.randint(-4,4)
                # print("Random Orientation" , rand_rotation_deg)
                # print("Random x,y" , rand_x,rand_y)
                if success:
                    environment_probs[map_index_curr] *= 0.95
                    environment_probs /= environment_probs.sum(axis=0)

                mCar.randomly_initiate_states()
                car_x = mCar.xx
                car_y = mCar.yy
                TARGET_POS_X = mCar.target_location[0]
                TARGET_POS_Y = mCar.target_location[1]

                # TARGET_POS_X = 0
                # TARGET_POS_Y = 30
                mCar.client.simFlushPersistentMarkers()
                mCar.client.simPlotPoints(points=[Vector3r(TARGET_POS_X, TARGET_POS_Y, -2)],
                                          color_rgba=[1, 0.0, 0.0, 1.0], size=150, duration=10, is_persistent=True)

                print(f"Target Position =  {TARGET_POS_X},{TARGET_POS_Y}")

                # mCar.setVehiclePose(car_x, car_y , 0)
                mCar.has_collided_flag = False
                time.sleep(0.1)
                print()
                break

            # update for next step in the episode
            state_torch_tensor = new_state_torch_tensor
            current_cosine_reward = new_current_cosine_reward
            curr_x = new_curr_x
            curr_y = new_curr_y
            curr_heading = new_curr_heading
            isClear_flag = isClear_flag_new

        if training_flag:
            if epoch % 200 == 0:
                agent.save_models(epoch, experiment_id)
            epoch_ending_time = time.time()
            elapsed_since_training_start = (training_starting_time - epoch_ending_time) / 60  # mins
            f = open("data/log_" + str(experiment_id) + ".txt", "a")
            f.write("Epoch num" + str(epoch) + "   Step_len: " + str(step_len) + "   Epoch Reward: " + str(
                epoch_reward.item()) + " Memory index: " + str(agent.memory.mem_index) + "Success: " + str(
                success) + " Time since beginning " + str(elapsed_since_training_start) + "\n")
            f.close()

    mCar.write_Pose_OrientTXT()
    input("Enter to exit...")
    print("Bitti")
