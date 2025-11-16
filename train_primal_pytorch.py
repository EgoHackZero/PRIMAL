import os
import random
from tqdm import tqdm
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mapf_gym
from ACNet_pytorch import ACNet
#from od_mstar3 import cpp_mstar
from od_mstar3 import od_mstar as py_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
from primal_env_wrapper import PrimalEnvWrapper

IL_TIME_LIMIT = 100 # 60 for c++, 300 for python
DEBUG_IL = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")


# Hyperparams
GRID_SIZE = 10
NUM_AGENTS = 8
A_SIZE = 5          

GAMMA = 0.95        
ENTROPY_BETA = 0.01 

BASE_LR = 2e-5     

MAX_EPISODES = 100000
MAX_STEPS_PER_EP = 256 

P_IL = 0.5 # IL episode probability


class PlannerFailed(Exception):
    pass



# Returns + Advantage

def compute_returns_and_advantages(rewards, values, dones, gamma=GAMMA):
    T = len(rewards)
    num_agents = rewards[0].shape[0]

    returns = [np.zeros(num_agents, dtype=np.float32) for _ in range(T)]
    advantages = [np.zeros(num_agents, dtype=np.float32) for _ in range(T)]

    R = values[-1].copy()

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].astype(np.float32)
        R = rewards[t] + gamma * R * mask
        returns[t] = R
        advantages[t] = returns[t] - values[t]

    return returns, advantages

def rl_loss_batch(policy_probs, values_pred,
                  actions, returns, advantages,
                  blocking_pred, blocking_tgt,
                  on_goal_pred, on_goal_tgt,
                  valids_pred, valids_tgt,
                  entropy_beta=ENTROPY_BETA):
    """
    policy_probs: [B, A_SIZE]
    values_pred:  [B, 1]
    actions:      [B]
    returns:      [B]
    advantages:   [B]
    blocking_pred / tgt: [B,1]
    on_goal_pred / tgt:  [B,1]
    valids_pred / tgt:   [B,A_SIZE]
    """
    log_probs = torch.log(policy_probs + 1e-8)
    act_log_probs = log_probs[torch.arange(actions.size(0)), actions]

    # policy loss (A2C)
    policy_loss = -(act_log_probs * advantages).mean()

    # value loss
    value_mse = (returns - values_pred.squeeze(-1)).pow(2).mean()
    value_loss = 0.5 * value_mse

    # entropy
    entropy = -(policy_probs * log_probs).sum(dim=1).mean()
    entropy_loss = -entropy_beta * entropy

    # valid actions
    valid_loss = F.binary_cross_entropy(valids_pred, valids_tgt)

    # blocking / on_goal 
    blocking_loss = F.binary_cross_entropy(blocking_pred, blocking_tgt)
    on_goal_loss  = F.binary_cross_entropy(on_goal_pred, on_goal_tgt)

    total_loss = (
        policy_loss
        + value_loss
        + 0.5 * valid_loss
        + 0.5 * blocking_loss
        + 0.5 * on_goal_loss
        + entropy_loss
    )

    return total_loss



def il_loss_batch(policy_probs, expert_actions):
    log_probs = torch.log(policy_probs + 1e-8)
    act_log_probs = log_probs[torch.arange(expert_actions.size(0)), expert_actions]
    ce_loss = -act_log_probs.mean()
    return ce_loss


# Expert ODrM* (cpp_mstar and pymstar)

def plan_expert_trajectory_cpp(env, max_steps=MAX_STEPS_PER_EP):
    base_env = env.env
    num_agents = base_env.num_agents

    raw_world = base_env.getObstacleMap()
    world = np.asarray(raw_world, dtype=np.int32)

    starts = base_env.getPositions()   
    goals  = base_env.getGoals()      
    try:
        joint_path = cpp_mstar.find_path(world, starts, goals, num_agents, 5)
    except (NoSolutionError, OutOfTimeError, MemoryError) as e:
        raise PlannerFailed(f"cpp_mstar failed: {e}")

    if not joint_path or len(joint_path) < 2:
        raise PlannerFailed("expert path too short")

    T = min(max_steps, len(joint_path) - 1)

    actions_seq = []

    for t in range(T):
        cur_state = joint_path[t]
        next_state = joint_path[t + 1]
        step_actions = np.zeros(num_agents, dtype=np.int64)

        for i in range(num_agents):
            cur = cur_state[i]
            nxt = next_state[i]
            dx = nxt[0] - cur[0]
            dy = nxt[1] - cur[1]

            if (dx, dy) not in mapf_gym.actionDict:
                raise PlannerFailed(f"non-cardinal expert move: {dx, dy}")

            step_actions[i] = mapf_gym.actionDict[(dx, dy)]

        actions_seq.append(step_actions)

    return np.stack(actions_seq, axis=0)  

def plan_expert_trajectory(env, max_steps=MAX_STEPS_PER_EP):
    base_env = env.env
    num_agents = base_env.num_agents

    world = np.asarray(base_env.getObstacleMap(), dtype=np.int32)
    starts = base_env.getPositions()
    goals  = base_env.getGoals()

    try:
        result = py_mstar.find_path(
            world, starts, goals, connect_8=False, time_limit=IL_TIME_LIMIT
        )
    except (NoSolutionError, OutOfTimeError, MemoryError) as e:
        if DEBUG_IL:
            print(f"[IL] py_mstar failed: {type(e).__name__} - {e}")
        raise PlannerFailed(f"py_mstar failed: {e}")

    if not result:
        raise PlannerFailed("empty expert result")

    joint_path = None

    if num_agents == 1 and isinstance(result, (list, tuple)) and result and isinstance(result[0], tuple):
        joint_path = [[pos] for pos in result]

    elif isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)) \
         and len(result[0]) == num_agents and isinstance(result[0][0], tuple):
        joint_path = result

    elif isinstance(result, (list, tuple)) and len(result) == num_agents \
         and all(isinstance(p, (list, tuple)) and (len(p) == 0 or isinstance(p[0], tuple)) for p in result):
        T_total = max((len(p) for p in result), default=0)
        if T_total < 2:
            raise PlannerFailed("expert path too short")
        joint_path = []
        for t in range(T_total):
            step = []
            for p in result:
                step.append(p[t] if t < len(p) else p[-1])
            joint_path.append(step)

    else:
        raise PlannerFailed("invalid expert paths shape")

    if len(joint_path) < 2:
        raise PlannerFailed("expert path too short")

    T = min(max_steps, len(joint_path) - 1)

    actions_seq = []
    for t in range(T):
        cur_state = joint_path[t]
        next_state = joint_path[t + 1]
        step_actions = np.zeros(num_agents, dtype=np.int64)

        for i in range(num_agents):
            (x0, y0) = cur_state[i]
            (x1, y1) = next_state[i]
            dx, dy = (x1 - x0), (y1 - y0)

            if (dx, dy) not in mapf_gym.actionDict:
                raise PlannerFailed(f"non-cardinal expert move: {(dx, dy)}")

            step_actions[i] = mapf_gym.actionDict[(dx, dy)]

        actions_seq.append(step_actions)

    return np.stack(actions_seq, axis=0)


def run_rl_episode(env, model, max_steps=MAX_STEPS_PER_EP):
    obs, goals = env.reset()
    obs = obs.to(DEVICE)
    goals = goals.to(DEVICE)

    hx, cx = model.init_hidden(env.num_agents)
    hx, cx = hx.to(DEVICE), cx.to(DEVICE)

    rewards = []
    dones_list = []

    policies = []
    actions_list = []
    values_torch = []
    values_np = []

    blocking_preds = []
    blocking_tgts = []
    on_goal_preds = []
    on_goal_tgts = []
    valids_preds = []
    valids_tgts = []

    for t in range(max_steps):
        policy, value, (hx, cx), blocking, on_goal, valids = model(obs, goals, (hx, cx))

        policy_np = policy.detach().cpu().numpy()
        acts = [np.random.choice(A_SIZE, p=policy_np[i])
                for i in range(env.num_agents)]
        acts = np.array(acts, dtype=np.int64)

        next_obs, next_goals, reward, done, info = env.step(acts)

        rewards.append(reward.astype(np.float32))
        dones = np.array([done] * env.num_agents, dtype=bool)
        dones_list.append(dones)

        values_torch.append(value)  
        values_np.append(value.detach().squeeze(-1).cpu().numpy()) 

        policies.append(policy)
        actions_list.append(torch.from_numpy(acts).to(DEVICE))

        blocking_preds.append(blocking)                 
        on_goal_preds.append(on_goal)                 
        valids_preds.append(valids)                       

        blocking_tgts.append(
            torch.from_numpy(info["blocking"]).view(-1, 1).to(DEVICE)
        )
        on_goal_tgts.append(
            torch.from_numpy(info["on_goal"]).view(-1, 1).to(DEVICE)
        )
        valids_tgts.append(
            torch.from_numpy(info["valids_mask"]).to(DEVICE)
        )

        obs, goals = next_obs.to(DEVICE), next_goals.to(DEVICE)

        if done:
            break

    returns, advantages = compute_returns_and_advantages(
        rewards, values_np, dones_list, gamma=GAMMA
    )

    policies_tensor = torch.cat(policies, dim=0).to(DEVICE)          
    actions_tensor  = torch.cat(actions_list, dim=0).long().to(DEVICE) 

    returns_tensor = torch.from_numpy(
        np.stack(returns).reshape(-1)
    ).to(DEVICE)                                                    
    advantages_tensor = torch.from_numpy(
        np.stack(advantages).reshape(-1)
    ).to(DEVICE)                                                     

    values_pred_tensor = torch.cat(values_torch, dim=0).view(-1, 1)  

    blocking_pred_tensor = torch.cat(blocking_preds, dim=0)          
    blocking_tgt_tensor  = torch.cat(blocking_tgts, dim=0)           

    on_goal_pred_tensor = torch.cat(on_goal_preds, dim=0)            
    on_goal_tgt_tensor  = torch.cat(on_goal_tgts, dim=0)              

    valids_pred_tensor = torch.cat(valids_preds, dim=0)               
    valids_tgt_tensor  = torch.cat(valids_tgts, dim=0)               

    ep_return_total = float(np.sum(np.stack(rewards))) 

    return (policies_tensor,
            actions_tensor,
            returns_tensor,
            advantages_tensor,
            values_pred_tensor,
            blocking_pred_tensor,
            blocking_tgt_tensor,
            on_goal_pred_tensor,
            on_goal_tgt_tensor,
            valids_pred_tensor,
            valids_tgt_tensor,
            ep_return_total)

def run_il_episode(env, model, max_steps=MAX_STEPS_PER_EP):
    obs, goals = env.reset()
    obs = obs.to(DEVICE)
    goals = goals.to(DEVICE)

    hx, cx = model.init_hidden(env.num_agents)
    hx, cx = hx.to(DEVICE), cx.to(DEVICE)

    actions_seq = plan_expert_trajectory(env, max_steps=max_steps)
    T = actions_seq.shape[0]

    policies = []
    expert_actions_list = []

    for t in range(T):
        expert_actions = actions_seq[t]             

        policy, value, (hx, cx), _, _, _ = model(obs, goals, (hx, cx))

        next_obs, next_goals, reward, done, info = env.step(expert_actions)

        policies.append(policy)
        expert_actions_list.append(
            torch.from_numpy(expert_actions.astype(np.int64)).to(DEVICE)
        )

        obs, goals = next_obs.to(DEVICE), next_goals.to(DEVICE)

        if done:
            break

    if not policies:
        raise PlannerFailed("empty IL episode")

    policies_tensor = torch.cat(policies, dim=0).to(DEVICE)             
    expert_actions_tensor = torch.cat(expert_actions_list, dim=0).long().to(DEVICE) 

    return policies_tensor, expert_actions_tensor

def train():
    env = PrimalEnvWrapper(size=(GRID_SIZE, GRID_SIZE * 4),
                           num_agents=NUM_AGENTS)

    model = ACNet("global", a_size=A_SIZE,
                  grid_size=GRID_SIZE, training=True).to(DEVICE)

    optimizer = torch.optim.NAdam(model.parameters(), lr=BASE_LR)

    episode_rewards = deque(maxlen=100)

    il_ok = il_fail = 0

    for ep in tqdm(range(1, MAX_EPISODES + 1)):
        for g in optimizer.param_groups:
            g["lr"] = BASE_LR / np.sqrt(ep)

        use_il = (random.random() < P_IL)

        if use_il:
            try:
                policies, expert_actions = run_il_episode(env, model)
                il_ok += 1
            except PlannerFailed as e:
                il_fail += 1
                if DEBUG_IL:
                    print(f"Episode {ep} | [IL] skipped (fail={il_fail}, ok={il_ok}): {e}")
                continue

            model.train()
            optimizer.zero_grad()
            loss = il_loss_batch(policies, expert_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
            optimizer.step()

            mode = "IL"
            ep_return = None
        else:
            (policies, actions, returns, advantages, values_pred,
             blocking_pred, blocking_tgt,
             on_goal_pred, on_goal_tgt,
             valids_pred, valids_tgt, ep_return) = run_rl_episode(env, model)

            model.train()
            optimizer.zero_grad()
            loss = rl_loss_batch(
                policy_probs=policies,
                values_pred=values_pred,
                actions=actions,
                returns=returns,
                advantages=advantages,
                blocking_pred=blocking_pred,
                blocking_tgt=blocking_tgt,
                on_goal_pred=on_goal_pred,
                on_goal_tgt=on_goal_tgt,
                valids_pred=valids_pred,
                valids_tgt=valids_tgt,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
            optimizer.step()

            mode = "RL"
            # ep_return = returns.view(-1, env.num_agents)[0].sum().item()


        if ep_return is not None:
            episode_rewards.append(ep_return)

        if ep % 10 == 0:
            avg_rew = np.mean(episode_rewards) if episode_rewards else 0.0
            print(f"Episode {ep} | mode={mode} | loss={loss.item():.3f} | avg_R={avg_rew:.2f} | IL ok/fail={il_ok}/{il_fail}")

        if ep % 1000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"primal_ep{ep}.pt")
            model.save(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
