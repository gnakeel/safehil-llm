import os
import gym
import sys
import yaml
import time
import torch
import warnings
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from collections import deque
import matplotlib.pyplot as plt

sys.path.append('/home/kang/code/rndix/safehil-llm/SMARTS')
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints

from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv

sys.path.append('/home/kang/code/rndix/safehil-llm/Auto_Driving_Highway')
from scenario import Scenario
from customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe
)
from analysis_obs import available_action, get_available_lanes, get_involved_cars, extract_lanes_info, extract_lane_and_car_ids, assess_lane_change_safety, check_safety_in_current_lane, format_training_info
import ask_llm as ask_llm
from ask_llm import ACTIONS_ALL
from utils import extract_decision

from drl_agent import DRL
from keyboard import HumanKeyboardAgent
from utils_ import soft_update, hard_update
from authority_allocation import Arbitrator
from main import observation_adapter, action_adapter, reward_adapter

from custom_ws import WebSocketClient 
import logging
import asyncio


class MyHiWayEnv(gym.Env):
    def __init__(self, screen_size, scenario_path, agent_specs, seed, vehicleCount,
                 observation_space, action_space, agent_id,
                 headless=False, visdom=False, sumo_headless=True,
                 websocket_client: WebSocketClient = None):
        super(MyHiWayEnv, self).__init__()
        
        self.screen_size = screen_size
        self.states = np.zeros(shape=(screen_size, screen_size, 9), dtype=np.float32)
        self.meta_state = None

        self.scenario_path = scenario_path
        self.agent_specs = agent_specs
        self.seed = seed
        self.vehicleCount = vehicleCount

        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = agent_id
        self.headless = headless
        self.visdom = visdom
        self.sumo_headless = sumo_headless

        self.env = HiWayEnv(
            scenarios=self.scenario_path,
            agent_specs=self.agent_specs,
            headless=self.headless,
            visdom=self.visdom,
            sumo_headless=self.sumo_headless,
            seed=self.seed
        )

        self._ws_client = websocket_client 
        if self._ws_client is None:
            logging.warning("MyHiWayEnv initialized without a WebSocketClient. No data will be sent via WebSocket.")

    async def step(self, action):
        # 연결 준비 대기
        if self._ws_client:
            await self._ws_client.wait_until_connected()

        # 안전 마스크
        ego_state = self.meta_state.ego_vehicle_state
        lane_id = ego_state.lane_index
        if ego_state.speed >= self.meta_state.waypoint_paths[lane_id][0].speed_limit and action[0] > 0.0:
            action = list(action)
            action[0] = 0.0
            action = tuple(action)
        
        meta_state, reward, done, truncated = self.env.step({self.agent_id: action})
        done = done[self.agent_id]
        self.meta_state = meta_state[self.agent_id]
        obs = self.observation_adapter(self.meta_state)

        self.last_observation = obs

        action_name, _ = ask_llm.get_action_info(action)

        if self._ws_client:
            message_to_send = {
                "agent_id": self.agent_id,
                "agent_action": action_name,
                "llm_action": self.llm_action,
                "timestamp": time.time()
            }
            try:
                await asyncio.wait_for(self._ws_client.send_message(message_to_send), timeout=2.0)
            except asyncio.TimeoutError:
                logging.error("[MyHiWayEnv] WebSocket send timeout. Skipping send.")

        custom_reward = self.calculate_custom_reward(action_name)

        return obs, custom_reward, done, truncated

    def set_llm_suggested_action(self, action):
        self.llm_action = action
    
    def calculate_custom_reward(self, action_name):
        if action_name in self.llm_action:
            reward = 1.0
            print(f"✅ 액션 일치! 보상: {reward}")
            return reward
    
        speed_actions = ["FASTER", "SLOWER"]
        if action_name in speed_actions and self.llm_action in speed_actions:
            reward = 0.3
            print(f"⚠️ 액션 부분 일치! 보상: {reward}")
            return reward

        print(f"❌ 액션 불일치! 보상: 0")
        return 0.0
    
    def observation_adapter(self, meta_state):
        new_obs = meta_state.top_down_rgb[1] / 255.0
        self.states[:, :, 0:3] = self.states[:, :, 3:6]
        self.states[:, :, 3:6] = self.states[:, :, 6:9]
        self.states[:, :, 6:9] = new_obs

        if meta_state.events.collisions or meta_state.events.reached_goal:
            self.states = np.zeros(shape=(self.screen_size, self.screen_size, 9), dtype=np.float32)

        return self.states
    
    def reset(self, **kwargs):
        meta_state = self.env.reset(**kwargs)
        self.meta_state = meta_state[self.agent_id]
        state = self.observation_adapter(self.meta_state)

        self.last_observation = state
        return state
    
    def get_available_actions(self):
        self.sce = Scenario(vehicleCount=self.vehicleCount)
        self.sce.updateVehicles(self.meta_state, 0)

        self.toolModels = [
            getAvailableActions(self.env.unwrapped),
            getAvailableLanes(self.sce),
            getLaneInvolvedCar(self.sce),
            isChangeLaneConflictWithCar(self.sce),
            isAccelerationConflictWithCar(self.sce),
            isKeepSpeedConflictWithCar(self.sce),
            isDecelerationSafe(self.sce),
        ]

        available = available_action(self.toolModels)
        valid_action_ids = [i for i, act in ACTIONS_ALL.items() if available.get(act, False)]
        return valid_action_ids
    
    async def __aenter__(self):
        if self._ws_client:
            await self._ws_client.connect()
            await self._ws_client.wait_until_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._ws_client:
            await self._ws_client.disconnect()
        self.env.close()


async def evaluate(env, agent, eval_episodes=10, epoch=0):
    ep = 0
    success = 0
    avg_reward_list = []
    lane_center = [-3.2, 0, 3.2]
    
    v_list_avg = []
    offset_list_avg = []
    dist_list_avg = []

    engage = int(0)
    
    while ep < eval_episodes:
        done = False
        reward_total = 0.0 
        frame_skip = 5
        
        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))

        s = env.reset()
        obs = env.meta_state
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

            ##### Select and perform an action #####
            a = agent.choose_action(s[0], action_list[-1], evaluate=True)
            action = action_adapter(a)
            
            llm_suggested_action = 'RIGHT'
            print(f"llm action: {llm_suggested_action}")

            env.set_llm_suggested_action(llm_suggested_action)

            s_, custom_reward, done, info = await env.step(action)
            print(f"Reward: {custom_reward}\n")

            obs = env.meta_state
            ego_state = obs.ego_vehicle_state
            curr_pos = ego_state.position[:2]
            print(ego_state.speed)
            
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(obs, pos_list, a, engage=engage, done=done)
            pos_list.append(curr_pos)
            
            reward_total += r + custom_reward
            action_list.append(a)
            s = s_
            
            l = ego_state.lane_position.t + lane_center[ego_state.lane_index]
            s_list.append(ego_state.lane_position.s)
            l_list.append(l)
            v_list.append(ego_state.speed)
            steer_list.append(a[-1])
    
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
    
async def train(env, agent):
    save_threshold = 3.0
    trigger_reward = 3.0
    trigger_epoc = 400
    saved_epoc = 1
    epoc = 0
    pbar = tqdm(total=MAX_NUM_EPOC)
    frame = 1

    arbitrator = Arbitrator()
    arbitrator.shared_control = SHARED_CONTROL

    _ = env.reset()
    _ = env.get_available_actions()
    toolModels = env.toolModels
    
    while epoc <= MAX_NUM_EPOC:
        reward_total = 0.0 
        error = 0.0 
        guidance_count = 0
        guidance_rate = 0.0
        frame_skip = 5

        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))

        reward_list = []
        reward_mean_list = []
        
        continuous_threshold = 100
        intermittent_threshold = 300
        
        pos_list = deque(maxlen=5)
        s = env.reset()
        obs = env.meta_state
        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list.append(initial_pos)

        for t in count():
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
            
            env.sce.updateVehicles(obs, frame)
            # Observation translation
            msg0 = available_action(toolModels)
            msg1 = get_available_lanes(toolModels)
            msg2 = get_involved_cars((toolModels))
            msg1_info = next(iter(msg1.values()))
            lanes_info = extract_lanes_info(msg1_info)

            lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2)
            safety_assessment = assess_lane_change_safety(toolModels, lane_car_ids)
            lane_change_safety = assess_lane_change_safety(toolModels, lane_car_ids)
            safety_msg = check_safety_in_current_lane(toolModels, lane_car_ids)
            formatted_info = format_training_info(msg0, msg1, msg2, lanes_info, lane_car_ids, safety_assessment, lane_change_safety, safety_msg)

            ##### Select and perform an action ######
            rl_a = agent.choose_action(s, action_list[-1])
            guidance = False

            ##### Human-in-the-loop #######
            if model != 'SAC' and human.intervention and epoc <= INTERMITTENT_THRESHOLD:
                human_action = human.act()
                guidance = True
            else:
                human_a = np.array([0.0, 0.0])
            
            ###### Assign final action ######
            if guidance:
                if human_action[1] > human.MIN_BRAKE:
                    human_a = np.array([-human_action[1], human_action[-1]])
                else:
                    human_a = np.array([human_action[0], human_action[-1]])
                
                if arbitrator.shared_control and epoc > CONTINUAL_THRESHOLD:
                    rl_authority, human_authority = arbitrator.authority(obs, rl_a, human_a)
                    a = rl_authority * rl_a + human_authority * human_a
                else:
                    a = human_a
                    human_authority = 1.0 #np.array([1.0, 1.0])
                engage = int(1)
                authority = human_authority
                guidance_count += int(1)
            else:
                a = rl_a
                engage = int(0)
                authority = 0.0 

            ##### Interaction #####
            action = action_adapter(a)

            llm_response = ask_llm.send_to_chatgpt(action, formatted_info, env.sce)
            decision_content = llm_response.content
            llm_suggested_action = extract_decision(decision_content)
            # llm_suggested_action = 'RIGHT'
            print(f"llm action: {llm_suggested_action}")

            env.set_llm_suggested_action(llm_suggested_action)

            s_, custom_reward, done, info = await env.step(action)
            print(f"Reward: {custom_reward}\n")

            obs = env.meta_state
            curr_pos = obs.ego_vehicle_state.position[:2]
            
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(obs, pos_list, a, engage=engage, done=done)
            r += custom_reward
            
            pos_list.append(curr_pos)

            ##### Store the transition in memory ######
            agent.store_transition(s, action_list[-1], a, human_a, r,
                                   s_, a, engage, authority, done)
            
            reward_total += r
            action_list.append(a)
            s = s_
            frame += 1
                            
            if epoc >= THRESHOLD:   
                # Train the DRL model
                agent.learn_guidence()

            if human.slow_down:
                time.sleep(1/40)

            if done:
                epoc += 1
                if epoc > THRESHOLD:
                    reward_list.append(max(-15.0, reward_total))
                    reward_mean_list.append(np.mean(reward_list[-10:]))
                
                    ###### Evaluating the performance of current model ######
                    if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                        # trigger_reward = reward_mean_list[-1]
                        print("Evaluating the Performance.")
                        avg_reward, _, _, _, _ = evaluate(env, agent, EVALUATION_EPOC)
                        trigger_reward = avg_reward
                        if avg_reward > save_threshold:
                            print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                            saved_epoc = epoc
                            
                            torch.save(agent.policy.state_dict(), set_path(path='trained_network'+env_name, prefix=name, suffix=env_name+'_actornetwork.pkl',
                                                                           agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS))

                            torch.save(agent.critic.state_dict(),  set_path(path='trained_network'+env_name, prefix=name, suffix=env_name+'_criticnetwork.pkl',
                                                                           agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS))
                            save_threshold = avg_reward

                print('\n|Epoc:', epoc,
                      '\n|Step:', t,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Guidance Rate:', guidance_rate, '%',
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Temperature:', agent.alpha,
                      '\n|Reward Threshold:', save_threshold,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
    
                s = env.reset()
                reward_total = 0
                error = 0
                pbar.update(1)
                break
        
        # if epoc % PLOT_INTERVAL == 0:
        #     plot_animation_figure(saved_epoc)
    
        if (epoc % SAVE_INTERVAL == 0):
            np.save(set_path(path='store/'+env_name, prefix='reward', suffix=env_name+'_'+name,
                             agent=agent, num_epoc=MAX_NUM_EPOC, num_steps=MAX_NUM_STEPS),
                    [reward_mean_list], allow_pickle=True, fix_imports=True)

    pbar.close()
    print('Complete')
    return save_threshold, reward_mean_list

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
            async with MyHiWayEnv(
                screen_size=_screen_size,
                scenario_path=_scenario_path,
                agent_specs=_agent_specs,
                seed=seed if mode != 'evaluation' else -1,
                vehicleCount=_vehicle_count,
                observation_space=_obs_space,
                action_space=_action_space,
                agent_id=_agent_id,
                headless=False,
                visdom=False,
                sumo_headless=True,
                websocket_client=ws
            ) as env:

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
                    save_threshold, reward_mean_list = await train(env, agent)
                
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