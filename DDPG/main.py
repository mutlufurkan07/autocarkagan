import numpy as np
import time
from airsim import Vector3r
import torch
from Agent import Agent
from mSimulationCar import mSimulationCar
import collections

# torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # Simulation params
    sim_clock_speed = 5
    action_duration = 1e-8/(sim_clock_speed*10)  # secs
    print(f"Action Duration: {action_duration*1000} msec")
    TARGET_POS_X, TARGET_POS_Y = 0, 50
    target_initial_distance = np.sqrt((TARGET_POS_X ** 2) + (TARGET_POS_Y ** 2))

    experiment_id = 41
    SUCCESS_RADIUS = 5  # 15m close to reward
    MAX_EPOCH = 5e6
    MAX_TIME = 50  # secs
    gamma = 0.995
    tau = 0.01
    actorlr = 2e-5
    criticlr = 2e-4
    variance = 0.2
    action_dim = 1
    mem_size = 4e5  # 80000
    batch_size = 64
    reward_dim = 6
    state_dim = 245
    circle_num = 10
    eps_dec = 0.9  # kkullanımıoyrz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mCar = mSimulationCar()
    training_flag = True

    if not training_flag:
        MAX_TIME = 500

    mCar.read_states()
    # mCar.createFolder() #call this to save the data to a text file ./data/....txt
    print("Starting")
    agent = Agent(gamma, tau, actorlr, criticlr, variance, action_dim, int(mem_size), batch_size, state_dim, reward_dim,
                  eps_dec, training_flag)
    # agent.load_models(70200, 31)

    # agent.load_models(10800, 35)
    car_x = 0
    car_y = 0

    training_starting_time = time.time()
    tot_step = 0
    average_reward_list = collections.deque(maxlen=50)
    average_success_list = collections.deque(maxlen=50)

    for epoch in range(int(MAX_EPOCH)):

        car_steering_old = 0
        steering_que = collections.deque(maxlen=5)
        [steering_que.append(0.0) for i in range(5)]
        epoch_reward = 0
        circle_reward_arr = np.ones((circle_num, 1))
        step_len = 0
        curr_time = time.time()
        success = 0

        circle_r_length = np.sqrt(((TARGET_POS_X - car_x) ** 2 + (TARGET_POS_Y - car_y) ** 2)) / (circle_num + 1)

        state_numpy, state_torch_tensor, curr_x, curr_y, curr_heading, isCollidedFlag, isClear_flag = mCar.take_action_and_get_current_lidar_and_targetArray(
            TARGET_POS_X, TARGET_POS_Y, target_initial_distance)

        state_torch_tensor = torch.cat((state_torch_tensor, torch.tensor(steering_que)))
        current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, curr_x, curr_y, curr_heading)
        state_torch_tensor = state_torch_tensor.to(device)
        st = time.time()
        while 1:
            # check success condition first
            if np.sqrt((curr_x - TARGET_POS_X) ** 2 + (curr_y - TARGET_POS_Y) ** 2) < SUCCESS_RADIUS:
                print("\nSUCCESS\n")
                success = 1
            # check time punishment flag
            time_punishment_flag = (MAX_TIME < time.time() - curr_time)

            agent_action = agent.action_selection(state_torch_tensor.float())

            car_steering_old += agent_action / 10
            car_steering_old = min(car_steering_old, 0.5)
            car_steering_old = max(car_steering_old, -0.5)
            # print("steering  angle" , car_steering_old)
            mCar.car_api_control_steer(car_steering_old, action_duration)
            steering_que.append(car_steering_old)

            new_state_numpy, new_state_torch_tensor, new_curr_x, new_curr_y, new_curr_heading, isCollidedFlag, isClear_flag_new = mCar.take_action_and_get_current_lidar_and_targetArray(
                TARGET_POS_X, TARGET_POS_Y, target_initial_distance)
            new_state_torch_tensor = torch.cat((new_state_torch_tensor, torch.tensor(steering_que)))
            new_state_torch_tensor = new_state_torch_tensor.to(device)
            new_current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, new_curr_x, new_curr_y,
                                                              new_curr_heading)
            # mCar.client.simPause(True)

            # ###  TODO: create a function for this algorithm
            current_distance_to_target = np.sqrt((TARGET_POS_X - new_curr_x) ** 2 + (TARGET_POS_Y - new_curr_y) ** 2)
            curr_Region = int(current_distance_to_target / circle_r_length)
            temp_circle_reward = 0
            if not curr_Region > circle_num:
                temp_circle_reward = circle_reward_arr[curr_Region - 1, 0]
                circle_reward_arr[curr_Region - 1] = 0
                ####

            # update network
            reward_tensor = torch.tensor([-0.075, current_cosine_reward / 2000, (isCollidedFlag or time_punishment_flag) * (-300), success * 400,
                                          temp_circle_reward * 35, isClear_flag * -2])
            # print(f"CCS: {current_cosine_reward}, TSR:  {temp_circle_reward*25},  ICR:{isClear_flag*-1}")

            if training_flag:
                agent.learn(state_torch_tensor, agent_action, reward_tensor, new_state_torch_tensor, isCollidedFlag)
            else:
                pass
                #time.sleep(action_duration)

            # DONMEK KOTUDUR ve düz gitmek iyidir
            if np.abs(agent_action) < 0.25:
                epoch_reward += reward_tensor.sum() + 2 * torch.tensor(1 - np.abs(agent_action))
            else:
                epoch_reward += reward_tensor.sum() - 1 * torch.tensor(np.abs(agent_action))

            step_len += 1
            # if time_current >= MAX_TIME or done or (success == 10):
            if time_punishment_flag or isCollidedFlag or success:
                q1 = agent.critic(state_torch_tensor.view(1, 245).float(),
                                   torch.tensor(agent_action).view(1, 1).to(device).float())

                average_reward_list.append(epoch_reward)
                avg_reward = sum(average_reward_list) / len(average_reward_list)
                tot_step += step_len
                average_success_list.append(success)
                avg_success = sum(average_success_list) / len(average_success_list)
                tt = time.time()
                print(
                    f"Episode: {epoch} , Reward: {epoch_reward.item():.3f} , Avg Reward: {avg_reward.item():.3f}  "
                    f"AvgSuccss: {avg_success*100:.2f}%, Success: {success} , Step Len: {step_len}  ,"
                    f" Total Steps: {tot_step} Time: {((tt - st) * 1000 / step_len):.3f}  msecs  "
                    f"Last Q-Value: {q1.item():8.2f}")

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

                # print(f"Target Position =  {TARGET_POS_X},{TARGET_POS_Y}")

                # mCar.setVehiclePose(car_x, car_y , 0)
                mCar.has_collided_flag = False
                # time.sleep(0.1)
                #print()
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
