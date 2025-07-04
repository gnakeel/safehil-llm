#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:02:04 2023

@author: oscar
"""

import asyncio
from custom_ws import WebSocketClient
import logging

import os
import gym
import sys
import yaml
import time
import torch
import warnings
import statistics
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from collections import deque
import matplotlib.pyplot as plt

# sys.path.append('/home/oscar/Dropbox/SMARTS')
sys.path.append('/home/kang/code/rndix/safehil-llm/SMARTS')
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints

from drl_agent import DRL
from keyboard import HumanKeyboardAgent
from utils_ import soft_update, hard_update
from authority_allocation import Arbitrator

def plot_animation_figure(epoc):
    plt.figure()
    plt.clf()
    
    plt.subplot(1, 1, 1)
    plt.title(env_name + ' ' + name + ' Save Epoc:' + str(epoc) +
              ' Alpha:' + str(agent.alpha))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    plt.plot(reward_mean_list)

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.tight_layout()
    plt.show()

def vec_2d(v) -> np.ndarray:
    """Converts a higher order vector to a 2D vector."""

    assert len(v) >= 2
    return np.array(v[:2])

def signedDistToLine(point, line_point, line_dir_vec) -> float:
    p = vec_2d(point)
    p1 = line_point
    p2 = line_point + line_dir_vec

    u = abs(
        line_dir_vec[1] * p[0] - line_dir_vec[0] * p[1] + p2[0] * p1[1] - p2[1] * p1[0]
    )
    d = u / np.linalg.norm(line_dir_vec)

    line_normal = np.array([-line_dir_vec[1], line_dir_vec[0]])
    _sign = np.sign(np.dot(p - p1, line_normal))
    return d * _sign

def evaluate(env_eval, agent, eval_episodes=10, epoch=0):
    ep = 0
    success = int(0)
    avg_reward_list = []
    lane_center = [-3.2, 0, 3.2]
    
    v_list_avg = []
    offset_list_avg = []
    dist_list_avg = []
    
    while ep < eval_episodes:
        obs = env_eval.reset()
        obs = obs[AGENT_ID]
        s = observation_adapter(obs)
        done = False
        reward_total = 0.0 
        frame_skip = 5
        
        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))
        
        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list = deque(maxlen=5)
        pos_list.append(initial_pos)

        df = pd.DataFrame([])
        s_list = []
        l_list = []
        offset_list = []
        v_list = []
        steer_list = []

        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
       
            if t <= frame_skip:
                ##### Select and perform an action #####
                a = agent.choose_action(np.array(s), action_list[-1], evaluate=True)
                action = action_adapter(a)
                
                ##### Safety Mask #####
                ego_state = obs.ego_vehicle_state
                lane_id = ego_state.lane_index
                if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
                   action[0] > 0.0:
                        action = list(action)
                        action[0] = 0.0
                        action = tuple(action)
                
                action = {AGENT_ID:action}
                next_state, reward, done, info = env_eval.step(action)
                obs = next_state[AGENT_ID]
                s_ = observation_adapter(next_state[AGENT_ID])
                curr_pos = next_state[AGENT_ID].ego_vehicle_state.position[:2]
                engage = int(0)
                done = done[AGENT_ID]
                if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                    done = True
                    print('Done')
                elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                    done = True
                    print('Done')
                
                r = reward_adapter(next_state[AGENT_ID], pos_list, a, engage, done=done)
                pos_list.append(next_state[AGENT_ID].ego_vehicle_state.position[:2])
                action_list.append(a)       
                s = s_
                
                l = ego_state.lane_position.t + lane_center[obs.ego_vehicle_state.lane_index]
                s_list.append(ego_state.lane_position.s)
                l_list.append(l)
                v_list.append(ego_state.speed)
                steer_list.append(a[-1])
                
                if done:
                    s = env_eval.reset()                   
                    reward_total = 0
                    error = 0
                    ep -= 1
                    print("wtf?")
                    break
                continue
       
            ##### Select and perform an action ######
            a = agent.choose_action(np.array(s), action_list[-1], evaluate=True)
            entropy_rl = 0.0
            guidance = False
            
            action = action_adapter(a)
            
            ##### Safety Mask #####
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
               action[0] > 0.0:
                   action = list(action)
                   action[0] = 0.0
                   action = tuple(action)
            
            action = {AGENT_ID:action}
            
            ####### G29 Interface ######
            # pygame.event.get()
            # print('g29!')
            # steering = 0
            # if js.get_button(4):
            #     steering = 0.1
            #     if js.get_button(10):
            #         steering *= 1.5
            # elif js.get_button(5):
            #     steering = -0.1
            #     if js.get_button(11):
            #         steering *= 1.5
            # time.sleep(0.2)
            # action = {AGENT_ID:((-js.get_axis(2)+1)/2,(-js.get_axis(3)+1)/2, steering)} # G29 test
            # print(action)
            
            next_state, reward, done, info = env_eval.step(action)
            obs = next_state[AGENT_ID]
            s_ = observation_adapter(next_state[AGENT_ID])
            curr_pos = next_state[AGENT_ID].ego_vehicle_state.position[:2]
            print(next_state[AGENT_ID].ego_vehicle_state.speed)
            engage = int(0)
            done = done[AGENT_ID]
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(next_state[AGENT_ID], pos_list, a, engage, done=done)
            pos_list.append(next_state[AGENT_ID].ego_vehicle_state.position[:2])
       
            lane_name = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_id
            lane_id = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_index
       
            reward_total += r
            action_list.append(a)
            s = s_
            
            l = ego_state.lane_position.t + lane_center[ego_state.lane_index]
            s_list.append(ego_state.lane_position.s)
            l_list.append(l)
            offset_list.append(abs(ego_state.lane_position.t))
            v_list.append(ego_state.speed)
            steer_list.append(a[-1])
            
            if human.slow_down:
                time.sleep(1/40)
            
            if done:
                if not info[AGENT_ID]['env_obs'].events.off_road and \
                    not info[AGENT_ID]['env_obs'].events.collisions:
                    success += 1
                
                print('\n|Epoc:', ep,
                      '\n|Step:', t,
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
                
                df["s"] = s_list
                df["l"] = l_list
                df["v"] = v_list
                df["steer"] = steer_list
                
                df.to_csv('./store/' + env_name + '/data_%s_%s' % (name, ep) + '.csv', index=0)
                
                break
            
        ep += 1
        v_list_avg.append(np.mean(v_list))
        offset_list_avg.append(np.mean(offset_list))
        dist_list_avg.append(curr_pos[0] - initial_pos[0])
        avg_reward_list.append(reward_total)
        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward: %f, Success No. : %i " % (ep, t, reward_total, success))
        print("..............................................")

    reward = statistics.mean(avg_reward_list)
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward:%f, Success No.: %i" % (eval_episodes, ep, reward, success))
    print("..............................................")
        
    return reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list

# observation space
def observation_adapter(env_obs):
    global states

    new_obs = env_obs.top_down_rgb[1]# / 255.0
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs
    ogm = env_obs.occupancy_grid_map[1] 
    drivable_area = env_obs.drivable_area_grid_map[1]

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(screen_size, screen_size, 9))

    return np.array(states, dtype=np.uint8)

# reward function
def reward_adapter(env_obs, pos_list, action, engage=False, done=False):
    ego_obs = env_obs.ego_vehicle_state
    ego_pos = ego_obs.position[:2]
    lane_name = ego_obs.lane_id
    lane_id = ego_obs.lane_index
    ref = env_obs.waypoint_paths
    
    ##### For Scratch ######
    heuristic = ego_obs.speed * 0.01 
    
    ###### Ternimal Reward #######
    if done and not env_obs.events.reached_max_episode_steps and \
       not env_obs.events.off_road and not bool(len(env_obs.events.collisions)):
        print('Good Job!')
        goal = 3.0 
    else:
        goal = 0.0

    if env_obs.events.off_road:
        print('\n Off Road!')
        off_road = - 7.0
    else:
        off_road = 0.0

    if env_obs.events.collisions:
        print('\n crashed')
        crash = - 7.0
    else:
        crash = 0.0

    if engage and PENALTY_GUIDANCE:
        guidance = 0.0
    else:
        guidance = 0.0

    ###### Performance Penatly ######
    if len(ref[lane_id]) > 1:
        ref_pos_1st = ref[lane_id][0].pos
        ref_pos_2nd = ref[lane_id][1].pos
        ref_dir_vec = ref_pos_2nd - ref_pos_1st
        lat_error = signedDistToLine(ego_pos, ref_pos_1st, ref_dir_vec)
        
        ref_heading = ref[lane_id][0].heading
        ego_heading = ego_obs.heading
        heading_error = ref_heading - ego_heading
    else:
        lat_error= 0.0
        heading_error = 0.0

    performance = - 0.01 * lat_error**2 - 0.1 * heading_error**2

    ###### For Scratch ######
    if env_obs.events.on_shoulder:
        print('\n on_shoulder')
        performance -= 0.1

    return heuristic + off_road + crash + performance + guidance + goal

# action space
def action_adapter(action): 

    long_action = action[0]
    if long_action < 0:
        throttle = 0.0
        braking = abs(long_action)
    else:
        throttle = abs(long_action)
        braking = 0.0
 
    steering = action[1]

    return (throttle, braking, steering)

# information

def info_adapter(observation, reward, info):
    return info

async def train(env, agent, ws):
    save_threshold = 3.0
    trigger_reward = 3.0
    trigger_epoc = 400
    saved_epoc = 1
    epoc = 0
    pbar = tqdm(total=MAX_NUM_EPOC)
    
    while epoc <= MAX_NUM_EPOC:
        reward_total = 0.0 
        error = 0.0 
        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))
        guidance_count = int(0)
        guidance_rate = 0.0
        frame_skip = 5
        
        continuous_threshold = 100
        intermittent_threshold = 300
        
        pos_list = deque(maxlen=5)
        obs = env.reset()
        obs = obs[AGENT_ID]
        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list.append(initial_pos)
        s = observation_adapter(obs)
        
        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break

            ##### Select and perform an action ######
            # rl_a = agent.choose_action(np.array(s), action_list[-1])
            rl_a = np.array([0.0, 0.0]) #x
            guidance = False
            
            ##### Human-in-the-loop #######
            # if model != 'SAC' and human.intervention and epoc <= INTERMITTENT_THRESHOLD:
            #     human_action = human.act()
            #     guidance = True
            # else:
            #     human_a = np.array([0.0, 0.0])
            human_a = np.array([0.0, 0.0])
            
            ###### Assign final action ######
            # if guidance:
            #     if human_action[1] > human.MIN_BRAKE:
            #         human_a = np.array([-human_action[1], human_action[-1]])
            #     else:
            #         human_a = np.array([human_action[0], human_action[-1]])
                
            #     if arbitrator.shared_control and epoc > CONTINUAL_THRESHOLD:
            #         rl_authority, human_authority = arbitrator.authority(obs, rl_a, human_a)
            #         a = rl_authority * rl_a + human_authority * human_a
            #     else:
            #         a = human_a
            #         human_authority = 1.0 #np.array([1.0, 1.0])
            #     engage = int(1)
            #     authority = human_authority
            #     guidance_count += int(1)
            # else:
            if True: #x
                a = rl_a
                engage = int(0)
                authority = 0.0 
            
            ##### Interaction #####
            action = action_adapter(a)
            
            ##### Safety Mask #####
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
               action[0] > 0.0:
                   action = list(action)
                   action[0] = 0.0
                   action = tuple(action)

            if ws:
                message_to_send = {
                    "agent_id": AGENT_ID,
                    "agent_action": 'test',
                    "llm_action": 'test',
                    "timestamp": time.time()
                }
                try:
                    await asyncio.wait_for(ws.send_message(message_to_send), timeout=2.0)
                except asyncio.TimeoutError:
                    logging.error("[MyHiWayEnv] WebSocket send timeout. Skipping send.")
                       
            action = {AGENT_ID:action}
            next_state, reward, done, info = env.step(action)
            obs = next_state[AGENT_ID]
            s_ = observation_adapter(obs)
            curr_pos = next_state[AGENT_ID].ego_vehicle_state.position[:2]
            
            done = done[AGENT_ID]
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(next_state[AGENT_ID], pos_list, a, engage=engage, done=done)
            pos_list.append(curr_pos)

            ##### Store the transition in memory ######
            agent.store_transition(s, action_list[-1], a, human_a, r,
                                   s_, a, engage, authority, done)
            
            reward_total += r
            action_list.append(a)
            s = s_
                            
            if epoc >= THRESHOLD:   
                # Train the DRL model
                agent.learn_guidence(BATCH_SIZE)

            if human.slow_down:
                time.sleep(1/40)

            if done:
                epoc += 1
                # if epoc > THRESHOLD:
                #     reward_list.append(max(-15.0, reward_total))
                #     reward_mean_list.append(np.mean(reward_list[-10:]))
                
                #     ###### Evaluating the performance of current model ######
                #     if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                #         # trigger_reward = reward_mean_list[-1]
                #         print("Evaluating the Performance.")
                #         avg_reward, _, _, _, _ = evaluate(env, agent, EVALUATION_EPOC)
                #         trigger_reward = avg_reward
                #         if avg_reward > save_threshold:
                #             print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                #             saved_epoc = epoc
                            
                #             torch.save(agent.policy.state_dict(), os.path.join('trained_network/' + env_name,
                #                       name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                #                       str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                #                       + str(seed)+'_'+env_name+'_actornet.pkl'))
                #             torch.save(agent.critic.state_dict(), os.path.join('trained_network/' + env_name,
                #                       name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                #                       str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                #                       + str(seed)+'_'+env_name+'_criticnet.pkl'))
                #             save_threshold = avg_reward

                # print('\n|Epoc:', epoc,
                #       '\n|Step:', t,
                #       '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                #       '\n|Guidance Rate:', guidance_rate, '%',
                #       '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                #       '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                #       '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                #       '\n|R:', reward_total,
                #       '\n|Temperature:', agent.alpha,
                #       '\n|Reward Threshold:', save_threshold,
                #       '\n|Algo:', name,
                #       '\n|seed:', seed,
                #       '\n|Env:', env_name)
    
                s = env.reset()
                reward_total = 0
                error = 0
                pbar.update(1)
                break
        
        # if epoc % PLOT_INTERVAL == 0:
        #     plot_animation_figure(saved_epoc)
    
        # if (epoc % SAVE_INTERVAL == 0):
        #     np.save(os.path.join('store/' + env_name, 'reward_memo'+str(MEMORY_CAPACITY) +
        #                               '_epoc'+str(MAX_NUM_EPOC)+'_step' + str(MAX_NUM_STEPS) +
        #                               '_seed'+ str(seed) +'_'+env_name+'_' + name),
        #             [reward_mean_list], allow_pickle=True, fix_imports=True)
            
        #     pass
    pbar.close()
    print('Complete')
    return save_threshold

def set_agent(env, channel, config, mode_param):

    agent = DRL(
        seed=env.seed,
        action_dim=env.action_space.high.size,
        state_dim=channel,
        pstate_dim=config['condition_state_dim'],
        policy_type=config['ACTOR_TYPE'],
        critic_type=config['CRITIC_TYPE'],
        LR_A=config['LR_ACTOR'],
        LR_C=config['LR_CRITIC'],
        BUFFER_SIZE=config['MEMORY_CAPACITY'],
        BATCH_SIZE=config['BATCH_SIZE'],
        TAU=config['TAU'],
        GAMMA=config['GAMMA'],
        ALPHA=config['ALPHA'],
        POLICY_GUIDANCE=mode_param['POLICY_GUIDANCE'],
        ADAPTIVE_CONFIDENCE=mode_param['ADAPTIVE_CONFIDENCE'],
        automatic_entropy_tuning=config['ENTROPY']
    )

    return agent

def set_path(path, prefix, suffix, agent, num_epoc, num_steps):

    return os.path.join(path,
            f'{prefix}_memo{agent.buffer_size}_epoc{num_epoc}_step{num_steps}_seed{agent.seed}_{suffix}')

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    plt.ion()
    
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Individual parameters for each model ######
    model = 'SAC'
    mode_param = config[model]
    name = mode_param['name']
    
    if model != 'SAC':
        SHARED_CONTROL = mode_param['SHARED_CONTROL']
        CONTINUAL_THRESHOLD = mode_param['CONTINUAL_THRESHOLD']
        INTERMITTENT_THRESHOLD = mode_param['INTERMITTENT_THRESHOLD']
    else:
        SHARED_CONTROL = False
        
    ###### Default parameters for DRL ######
    mode = config['mode']
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    MAX_NUM_EPOC = config['MAX_NUM_EPOC']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PLOT_INTERVAL = config['PLOT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    EVALUATION_EPOC = config['EVALUATION_EPOC']
    VALUE_GUIDANCE = mode_param['VALUE_GUIDANCE'],

    ###### Env Settings #######
    env_name = config['env_name']
    scenario = config['scenario_path']
    screen_size = config['screen_size']
    view = config['view']
    AGENT_ID = config['AGENT_ID']
    
    # Create the network storage folders
    if not os.path.exists("./store/" + env_name):
        os.makedirs("./store/" + env_name)
        
    if not os.path.exists("./trained_network/" + env_name):
        os.makedirs("./trained_network/" + env_name)

    #### Environment specs ####
    ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))
    states = np.zeros(shape=(screen_size, screen_size, 9), dtype=np.float32)

    ##### Define agent interface #######
    agent_interface = AgentInterface(
        max_episode_steps=MAX_NUM_STEPS,
        waypoints=Waypoints(50),
        neighborhood_vehicles=NeighborhoodVehicles(radius=100),
        rgb=RGB(screen_size, screen_size, view/screen_size),
        ogm=OGM(screen_size, screen_size, view/screen_size),
        drivable_area_grid_map=DrivableAreaGridMap(screen_size, screen_size, view/screen_size),
        action=ActionSpaceType.Continuous,
    )

    ###### Define agent specs ######
    agent_spec = AgentSpec(
        interface=agent_interface,
        # observation_adapter=observation_adapter,
        # reward_adapter=reward_adapter,
        # action_adapter=action_adapter,
        # info_adapter=info_adapter,
    )
    
    ######## Human Intervention through g29 or keyboard ########
    human = HumanKeyboardAgent()

    ##### Set Env ######
    if model == 'SAC':
        envisionless = True
    else:
        envisionless = False

    img_h, img_w, channel = screen_size, screen_size, 9
    physical_state_dim = 2
    n_obs = img_h * img_w * channel

    ##### WebSocket URI #####
    WEBSOCKET_URI = 'ws://localhost:8082/actions'

    ##### Train #####
    _scenario_path = [scenario]
    _screen_size = screen_size
    _agent_specs = {AGENT_ID: agent_spec}
    _vehicle_count = 5
    _obs_space = OBSERVATION_SPACE
    _action_space = ACTION_SPACE
    _agent_id = AGENT_ID
    async def run_training(seed):
        async with WebSocketClient(uri=WEBSOCKET_URI) as ws:
            env = HiWayEnv(scenarios=_scenario_path, agent_specs=_agent_specs,
                       headless=False, visdom=False, sumo_headless=True, seed=seed)
            env.seed = seed
            env.action_space = _action_space
            agent = set_agent(env, channel, config, mode_param)

            print(f'''
                    The object is: {model}
                    |Seed: {agent.seed} 
                    |VALUE_GUIDANCE: {mode_param['VALUE_GUIDANCE']}
                    |PENALTY_GUIDANCE: {mode_param['PENALTY_GUIDANCE']}
                    ''')
            
            success_count = 0

            if mode == 'evaluation':
                name = 'sac'
                max_epoc = 820
                max_steps = 300
                # directory =  'best_candidate'
                
                filename = set_path('trained_network/'+env_name, name, env_name+'_actornet.pkl', agent, max_epoc, max_steps)
                        
                agent.policy.load_state_dict(torch.load(filename))
                agent.policy.eval()
                reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list = await evaluate(env, agent, eval_episodes=10)
                
                print(f'''
                        |Avg Speed: {np.mean(v_list_avg)}
                        |Std Speed: {np.std(v_list_avg)}
                        |Avg Dist: {np.mean(dist_list_avg)}
                        |Std Dist: {np.std(dist_list_avg)}
                        |Avg Offset: {np.mean(offset_list_avg)}
                        |Std Offset: {np.std(offset_list_avg)}
                        ''')

            else:
                save_threshold, reward_mean_list = await train(env, agent, ws)
            
                np.save(set_path(path='store/'+env_name, prefix='reward', suffix=env_name+'_'+name,
                                agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS),
                        [reward_mean_list], allow_pickle=True, fix_imports=True)
        
                torch.save(agent.policy.state_dict(), set_path(path='trained_network'+env_name, prefix=name, suffix=env_name+'_actornet_final.pkl',
                        agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS))
                
                
                torch.save(agent.critic.state_dict(), set_path(path='trained_network'+env_name, prefix=name, suffix=env_name+'_criticnet_final.pkl',
                        agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS))

    try:
        for i in range(1, 2):
            seed = i

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            asyncio.run(run_training(seed))

    except KeyboardInterrupt:
        logging.info("Training stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logging.critical(f"An unhandled error occurred during training: {e}", exc_info=True)