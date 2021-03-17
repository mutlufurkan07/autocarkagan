import numpy as np
import time
from airsim import Vector3r
import torch
from Agent import Agent
from mSimulationCar import mSimulationCar
import collections


if __name__ == "__main__":
    # Simulation params
    simulation_clockspeed = 2
    desired_action_interval = 0.1  # secs
    action_duration = desired_action_interval / simulation_clockspeed  # secs

    TARGET_POS_X, TARGET_POS_Y = 0, 50
    target_initial_distance = np.sqrt((TARGET_POS_X ** 2) + (TARGET_POS_Y ** 2))

    experiment_id = 6
    SUCCESS_RADIUS = 5  # 5m close to reward
    MAX_EPOCH = 5e6
    MAX_TIME = 50  # secs
    gamma = 0.99
    criticlr = 1e-5
    std = 0.1
    noise_mag = 0.5
    max_action = 1  # max steering

    variance = 0.1
    action_dim = 1
    mem_size = 4e5
    batch_size = 64
    reward_dim = 6
    state_dim = 245
    circle_num = 10
    car_x = 0
    car_y = 0
    max_allowed_steering = 0.5

    # PPO Specific
    hidden_actor_dim1 = 256
    hidden_actor_dim2 = 256
    hidden_actor_dim3 = 128
    hidden_critic_dim1 = 256
    hidden_critic_dim2 = 128
    hidden_critic_dim3 = 128
    betas = (0.9, 0.999)
    K_epochs = 40
    eps_clip = 0.2
    update_horizon = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mCar = mSimulationCar()

    training_flag = True  # -----------------------------------------------------
    mCar.read_states()

    if not training_flag:
        MAX_TIME = 500  # secs

    agent = Agent(gamma=gamma, lr=criticlr, action_std=std, K_epochs=K_epochs, state_dim=state_dim, eps_clip=eps_clip,
                  betas=betas, action_dim=action_dim, hidden_actor_dim1=hidden_actor_dim1,
                  hidden_actor_dim2=hidden_actor_dim2,
                  hidden_actor_dim3=hidden_actor_dim3, hidden_critic_dim1=hidden_critic_dim1,
                  hidden_critic_dim2=hidden_critic_dim2, hidden_critic_dim3=hidden_critic_dim3, device=device,
                  update_horizon=update_horizon)

    training_starting_time = time.time()
    average_reward_list = collections.deque(maxlen=50)
    average_success_list = collections.deque(maxlen=50)

    all_steps = 0
    tot_step = 0

    for epoch in range(int(MAX_EPOCH)):

        car_steering_old = 0
        steering_que = collections.deque(maxlen=5)
        [steering_que.append(0.0) for i in range(5)]

        circle_reward_arr = np.ones((circle_num, 1))
        step_len = 0

        curr_time = time.time()
        success = 0

        circle_r_length = np.sqrt(((TARGET_POS_X - car_x) ** 2 + (TARGET_POS_Y - car_y) ** 2)) / (circle_num + 1)

        _, state_torch_tensor, _, _, _, _, _ = mCar.take_action_and_get_current_lidar_and_targetArray(TARGET_POS_X,
                                                                                                      TARGET_POS_Y,
                                                                                                      target_initial_distance)
        state_torch_tensor = torch.cat((state_torch_tensor, torch.tensor(steering_que)))
        state_torch_tensor = state_torch_tensor.to(device)

        agent_action, action_log_prob = agent.action_selection(state_torch_tensor.float())

        car_steering_old += agent_action / 10
        car_steering_old = min(car_steering_old, max_allowed_steering)
        car_steering_old = max(car_steering_old, -max_allowed_steering)
        steering_que.append(car_steering_old)

        mCar.car_api_control_steer(car_steering_old, action_duration)

        _, new_state_torch_tensor, new_curr_x, new_curr_y, new_curr_heading, isCollidedFlag, isClear_flag_new = \
            mCar.take_action_and_get_current_lidar_and_targetArray(TARGET_POS_X, TARGET_POS_Y, target_initial_distance)
        new_state_torch_tensor = torch.cat((new_state_torch_tensor, torch.tensor(steering_que)))
        new_state_torch_tensor = new_state_torch_tensor.to(device)

        new_current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, new_curr_x, new_curr_y,
                                                          new_curr_heading)

        reward_tensor = torch.tensor([-0.075, new_current_cosine_reward / 2000,
                                      (isCollidedFlag or False) * (-300),
                                      success * 600, 0, isClear_flag_new * -2])

        compound_reward = reward_tensor.sum()

        # DONMEK KOTUDUR ve düz gitmek iyidir
        if np.abs(agent_action) < 0.25:
            compound_reward += compound_reward + 2 * torch.tensor(1 - np.abs(agent_action))
        else:
            compound_reward += compound_reward - 1 * torch.tensor(np.abs(agent_action))

        if training_flag:
            if tot_step % update_horizon == 0:
                mCar.client.simPause(True)
                agent.update()
                agent.memory.clear_memory()
                mCar.client.simPause(False)
                tot_step = 0

            # print(f"fTotal_step = {tot_step}")
            agent.memory.store(s=state_torch_tensor.cpu(), a=agent_action, r=compound_reward,
                               d=isCollidedFlag,
                               logprob=action_log_prob.item())
            tot_step += 1

        episode_reward = 0
        while 1:
            action_start_time = time.time()

            # mCar.client.simPause(True)
            agent_action, action_log_prob = agent.action_selection(new_state_torch_tensor.float())

            car_steering_old += agent_action / 10
            car_steering_old = min(car_steering_old, max_allowed_steering)
            car_steering_old = max(car_steering_old, -max_allowed_steering)
            steering_que.append(car_steering_old)

            mCar.car_api_control_steer(car_steering_old, action_duration)

            state_torch_tensor = new_state_torch_tensor
            
            _, new_state_torch_tensor, new_curr_x, new_curr_y, new_curr_heading, isCollidedFlag, isClear_flag_new \
                = mCar.take_action_and_get_current_lidar_and_targetArray(TARGET_POS_X,
                                                                         TARGET_POS_Y,
                                                                         target_initial_distance)

            new_state_torch_tensor = torch.cat((new_state_torch_tensor, torch.tensor(steering_que)))
            new_state_torch_tensor = new_state_torch_tensor.to(device)

            new_current_cosine_reward = mCar.car_cosineReward(TARGET_POS_X, TARGET_POS_Y, new_curr_x, new_curr_y,
                                                              new_curr_heading)

            ####
            current_distance_to_target = np.sqrt((TARGET_POS_X - new_curr_x) ** 2 + (TARGET_POS_Y - new_curr_y) ** 2)
            curr_Region = int(current_distance_to_target / circle_r_length)
            temp_circle_reward = 0
            if not curr_Region > circle_num:
                temp_circle_reward = circle_reward_arr[curr_Region - 1, 0]
                circle_reward_arr[curr_Region - 1] = 0
            ####

            # check success condition first
            if np.sqrt((new_curr_x - TARGET_POS_X) ** 2 + (new_curr_x - TARGET_POS_Y) ** 2) < SUCCESS_RADIUS:
                print("\n--------------------------SUCCESS--------------------------\n")
                success = 1

            # check time punishment flag
            time_punishment_flag = (MAX_TIME < time.time() - curr_time)
                        
            # update network
            reward_tensor = torch.tensor(
                [-0.075, new_current_cosine_reward / 2000, (isCollidedFlag or time_punishment_flag) * (-300), success * 600,
                 temp_circle_reward * 30, isClear_flag_new * -2])

            compound_reward = reward_tensor.sum()

            # DONMEK KOTUDUR ve düz gitmek iyidir
            if np.abs(agent_action) < 0.25:
                compound_reward += 2 * torch.tensor(1 - np.abs(agent_action))
            else:
                compound_reward += -1 * torch.tensor(np.abs(agent_action))

            episode_reward += compound_reward
            step_len += 1

            if training_flag:
                if tot_step % update_horizon == 0:
                    mCar.client.simPause(True)
                    agent.update()
                    agent.memory.clear_memory()
                    mCar.client.simPause(False)
                    tot_step = 0


                # print(f"fTotal_step = {tot_step}")
                agent.memory.store(s=state_torch_tensor.cpu(), a=agent_action, r=compound_reward,
                                   d=isCollidedFlag,
                                   logprob=action_log_prob.item())
                tot_step += 1

            else:
                time.sleep(0.003)

            finish_time = time.time()
            elapsed_time = finish_time - action_start_time
            if elapsed_time < action_duration:
                time.sleep(action_duration - elapsed_time)
            else:
                print(f"Time error......")
                pass
            all_steps += 1
            if time_punishment_flag or isCollidedFlag or success:
                average_reward_list.append(episode_reward)
                avg_reward = sum(average_reward_list) / len(average_reward_list)

                average_success_list.append(success)
                avg_success = sum(average_success_list) * 100 / len(average_success_list)
                print(f"Episode: {epoch:5} , Reward: {episode_reward:8.3f} , Avg Reward: {avg_reward.item():8.3f} "
                      f"Avg Success: {avg_success:.1f}%, Success: {success:2} , Step Len: {step_len:4}  , Total Steps: "
                      f"{all_steps:7}", f"Last Q Value {agent.state_value(new_state_torch_tensor.float()).item():.6f}")

                mCar.randomly_initiate_states()
                car_x = mCar.initial_x
                car_y = mCar.initial_y
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
                # print()
                break

        if training_flag:
            if epoch % 200 == 0:
                agent.save_models(epoch, experiment_id)
            epoch_ending_time = time.time()
            elapsed_since_training_start = (training_starting_time - epoch_ending_time) / 60  # mins
            f = open("data/log_" + str(experiment_id) + ".txt", "a")
            f.write("Epoch num" + str(epoch) + "   Step_len: " + str(step_len) + "   Epoch Reward: " + str(
                episode_reward) + " Memory index: " + str(agent.memory.mem_index) + "Success: " + str(
                success) + " Time since beginning " + str(elapsed_since_training_start) + "\n")
            f.close()

    mCar.write_Pose_OrientTXT()
    input("Enter to exit...")
    print("Bitti")
