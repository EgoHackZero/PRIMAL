import numpy as np
import torch

from mapf_gym import MAPFEnv


class PrimalEnvWrapper:
    # Wrapper over MAPFEnv:
    # creates env with needed obs size
    # provides obs for all agents formatted for ACNet (torch tensors)
    def __init__(
        self,
        num_agents=4,
        observation_size=10,
        size=(10, 40),
        prob=(0.0, 0.5),
        diagonal_movement=False,
        full_help=False,
        device=None,
    ):
        self.env = MAPFEnv(
            num_agents=num_agents,
            observation_size=observation_size,
            SIZE=size,
            PROB=prob,
            DIAGONAL_MOVEMENT=diagonal_movement,
            FULL_HELP=full_help,
        )
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # main function: retrieve obs/goal for all agents

    def get_all_agent_observations(self):
        """
        Возвращает:
          obs_tensor:  [N, 4, obs_size, obs_size]  (float32)
          goal_tensor: [N, 3]                       (float32)
        где N = num_agents.
        """
        obs_list = []
        goals_list = []

        for agent_id in range(1, self.num_agents + 1):
            (channels, goal_vec) = self.env._observe(agent_id)
            # channels = [poss_map, goal_map, goals_map, obs_map]

            processed = [np.asarray(ch, dtype=np.float32) for ch in channels]
            maps = np.stack(processed, axis=0)  # [4, H, W]
            goal = np.asarray(goal_vec, dtype=np.float32)  # [3]

            obs_list.append(maps)
            goals_list.append(goal)

        obs_array = np.stack(obs_list, axis=0).astype(np.float32)    
        goals_array = np.stack(goals_list, axis=0).astype(np.float32)  

        obs_tensor = torch.from_numpy(obs_array).to(self.device)
        goal_tensor = torch.from_numpy(goals_array).to(self.device)

        return obs_tensor, goal_tensor

    # --------- reset / step ---------

    def reset(self):
        # Recreates env with random map size while keeping agent count and obstacle density range.
        size_min, size_max = self.env.SIZE
        world_size = np.random.randint(size_min, size_max + 1)

        prob_range = self.env.PROB

        self.env = MAPFEnv(
            num_agents=self.num_agents,
            observation_size=self.observation_size,
            SIZE=(world_size, world_size),              
            PROB=prob_range,                            
            DIAGONAL_MOVEMENT=self.env.DIAGONAL_MOVEMENT,
            FULL_HELP=self.env.FULL_HELP,
        )

        return self.get_all_agent_observations()



    def step_single_agent(self, agent_id, action):
        (state, reward, done, _, on_goal, blocking, valid_action) = \
            self.env._step((agent_id, action))

        obs_tensor, goal_tensor = self.get_all_agent_observations()

        info = {
            "on_goal": on_goal,
            "blocking": blocking,
            "valid_action": valid_action,
        }

        return obs_tensor, goal_tensor, reward, done, info

    def step(self, actions):
        rewards = []
        done_flags = []

        a_size = self.env.action_space[1].n

        valids_masks = []
        for agent_id in range(1, self.num_agents + 1):
            avail = self.env._listNextValidActions(agent_id) 
            mask = np.zeros(a_size, dtype=np.float32)
            mask[avail] = 1.0
            valids_masks.append(mask)

        on_goal_list = []
        blocking_list = []
        valid_action_list = []

        for agent_id, action in enumerate(actions, start=1):
            state, reward, done, next_actions, on_goal, blocking, valid_action = \
                self.env._step((agent_id, int(action)))

            rewards.append(reward)
            done_flags.append(done)
            on_goal_list.append(float(on_goal))
            blocking_list.append(float(blocking))
            valid_action_list.append(float(valid_action))
            # self.prev_actions[agent_id-1] = int(action)

        obs_tensor, goal_tensor = self.get_all_agent_observations()

        done = any(done_flags)
        info = {
            "valids_mask": np.stack(valids_masks, axis=0),          
            "on_goal": np.array(on_goal_list, dtype=np.float32),    
            "blocking": np.array(blocking_list, dtype=np.float32), 
            "valid_action": np.array(valid_action_list, dtype=np.float32), 
        }

        return obs_tensor, goal_tensor, np.array(rewards, dtype=np.float32), done, info
