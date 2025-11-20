import numpy as np
import torch
import os

from mapf_gym import MAPFEnv


class PrimalEnvWrapper:
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
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.initial_size = size
        self.initial_prob = prob
        self.diagonal_movement = diagonal_movement
        self.full_help = full_help
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.env = MAPFEnv(
            num_agents=num_agents,
            observation_size=observation_size,
            SIZE=size,
            PROB=prob,
            DIAGONAL_MOVEMENT=diagonal_movement,
            FULL_HELP=full_help,
        )
        

        self.reset()

        # print("✓✓✓ AFTER RESET - env shape:", self.env.world.state)

    def get_all_agent_observations(self):
        obs_list = []
        goals_list = []

        for agent_id in range(1, self.num_agents + 1):
            (channels, goal_vec) = self.env._observe(agent_id)
            processed = [np.asarray(ch, dtype=np.float32) for ch in channels]
            maps = np.stack(processed, axis=0)
            goal = np.asarray(goal_vec, dtype=np.float32)

            obs_list.append(maps)
            goals_list.append(goal)

        obs_array = np.stack(obs_list, axis=0).astype(np.float32)    
        goals_array = np.stack(goals_list, axis=0).astype(np.float32)  

        obs_tensor = torch.from_numpy(obs_array).to(self.device)
        goal_tensor = torch.from_numpy(goals_array).to(self.device)

        return obs_tensor, goal_tensor

    def reset(self):
        use_maze = np.random.random() < 0.2
        maze_path = os.path.join("saved_environments", "maze.csv")
        
        if use_maze and os.path.exists(maze_path):
            try:
                maze_raw = np.loadtxt(maze_path, delimiter=';', dtype=np.int32, encoding='utf-8-sig')
                
                maze_obstacles = np.where(maze_raw == 0, -1, 0)
                if np.sum(maze_obstacles == 0) < self.num_agents * 2:
                    use_maze = False
                else:
                    self.env = MAPFEnv(
                        num_agents=self.num_agents,
                        observation_size=self.observation_size,
                        world0=maze_obstacles.copy(),
                        goals0=None,
                        SIZE=maze_obstacles.shape,
                        PROB=self.initial_prob,
                        DIAGONAL_MOVEMENT=self.diagonal_movement,
                        FULL_HELP=self.full_help,
                        blank_world=True
                    )
                    return self.get_all_agent_observations()
            except Exception as e:
                use_maze = False
        
        if not use_maze:
            world_size = np.random.choice([10, 40, 70], p=[0.5, 0.25, 0.25])

            self.env = MAPFEnv(
                num_agents=self.num_agents,
                observation_size=self.observation_size,
                SIZE=(world_size, world_size),
                PROB=self.initial_prob,
                DIAGONAL_MOVEMENT=self.diagonal_movement,
                FULL_HELP=self.full_help,
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

        obs_tensor, goal_tensor = self.get_all_agent_observations()

        done = any(done_flags)
        info = {
            "valids_mask": np.stack(valids_masks, axis=0),          
            "on_goal": np.array(on_goal_list, dtype=np.float32),    
            "blocking": np.array(blocking_list, dtype=np.float32), 
            "valid_action": np.array(valid_action_list, dtype=np.float32), 
        }

        return obs_tensor, goal_tensor, np.array(rewards, dtype=np.float32), done, info
