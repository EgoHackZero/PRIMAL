import os
import time
import random
from collections import deque, defaultdict
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mapf_gym
from ACNet_pytorch import ACNet
from od_mstar3 import od_mstar as py_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
from primal_env_wrapper import PrimalEnvWrapper

DEVICE_GLOBAL = torch.device("cpu")
GRID_SIZE   = int(os.getenv("PRIMAL_GRID", "10"))
NUM_AGENTS  = int(os.getenv("PRIMAL_N", "8"))
A_SIZE      = 5

GAMMA        = 0.95
ENTROPY_BETA0= float(os.getenv("PRIMAL_ENT_BETA", "0.01"))
BASE_LR      = float(os.getenv("PRIMAL_LR", "2e-5"))

MAX_EPISODES       = int(os.getenv("PRIMAL_MAX_EP", "100000"))
MAX_STEPS_PER_EP   = int(os.getenv("PRIMAL_MAX_STEPS", "1000")) 
P_IL               = float(os.getenv("PRIMAL_P_IL", "0.5"))    
IL_TIME_LIMIT      = int(os.getenv("PRIMAL_IL_TL", "60"))       
NUM_WORKERS        = int(os.getenv("PRIMAL_WORKERS", "4"))
GRAD_CLIP          = float(os.getenv("PRIMAL_GRAD_CLIP", "40.0"))
SAVE_EVERY         = int(os.getenv("PRIMAL_SAVE_EVERY", "1000"))
CKPT_DIR           = os.getenv("PRIMAL_CKPT_DIR", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

METRICS_WINDOW     = int(os.getenv("PRIMAL_METR_WIN", "100"))
PRINT_EVERY_PACKETS= int(os.getenv("PRIMAL_METR_PRINT_EVERY", "20"))


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()


class PlannerFailed(Exception):
    pass

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
                  entropy_beta):
    log_probs = torch.log(policy_probs + 1e-8)
    act_log_probs = log_probs[torch.arange(actions.size(0)), actions]
    policy_loss = -(act_log_probs * advantages).mean()
    value_mse = (returns - values_pred.squeeze(-1)).pow(2).mean()
    value_loss = 0.5 * value_mse
    entropy = -(policy_probs * log_probs).sum(dim=1).mean()
    entropy_loss = -entropy_beta * entropy
    valid_loss = F.binary_cross_entropy(valids_pred, valids_tgt)
    blocking_loss = F.binary_cross_entropy(blocking_pred, blocking_tgt)
    on_goal_loss  = F.binary_cross_entropy(on_goal_pred, on_goal_tgt)
    total_loss = policy_loss + value_loss + 0.5*valid_loss + 0.5*blocking_loss + 0.5*on_goal_loss + entropy_loss
    return total_loss

def il_loss_batch(policy_probs, expert_actions):
    log_probs = torch.log(policy_probs + 1e-8)
    act_log_probs = log_probs[torch.arange(expert_actions.size(0)), expert_actions]
    return -act_log_probs.mean()


def plan_expert_trajectory(env, max_steps=MAX_STEPS_PER_EP):
    base_env = env.env
    num_agents = base_env.num_agents
    world  = np.asarray(base_env.getObstacleMap(), dtype=np.int32)
    starts = base_env.getPositions()
    goals  = base_env.getGoals()
    try:
        result = py_mstar.find_path(world, starts, goals, connect_8=False, time_limit=IL_TIME_LIMIT)
    except (NoSolutionError, OutOfTimeError, MemoryError) as e:
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

    T = min(max_steps, len(joint_path)-1)
    actions_seq = []
    for t in range(T):
        cur_state  = joint_path[t]
        next_state = joint_path[t+1]
        step_actions = np.zeros(num_agents, dtype=np.int64)
        for i in range(num_agents):
            (x0, y0) = cur_state[i]
            (x1, y1) = next_state[i]
            dx, dy = (x1-x0), (y1-y0)
            if (dx, dy) not in mapf_gym.actionDict:
                raise PlannerFailed(f"non-cardinal expert move: {(dx,dy)}")
            step_actions[i] = mapf_gym.actionDict[(dx, dy)]
        actions_seq.append(step_actions)
    return np.stack(actions_seq, axis=0)


def run_rl_episode(env, model, max_steps=MAX_STEPS_PER_EP, device=torch.device("cpu"),
                   entropy_beta=ENTROPY_BETA0):
    obs, goals = env.reset()
    obs, goals = obs.to(device), goals.to(device)
    hx, cx = model.init_hidden(env.num_agents)
    hx, cx = hx.to(device), cx.to(device)

    rewards, dones_list = [], []
    policies, actions_list, values_torch, values_np = [], [], [], []
    blocking_preds, blocking_tgts = [], []
    on_goal_preds,  on_goal_tgts  = [], []
    valids_preds,   valids_tgts   = [], []

    block_count = 0
    invalid_choices = 0
    steps_taken = 0
    last_on_goal = None

    for _ in range(max_steps):
        policy, value, (hx, cx), blocking, on_goal, valids = model(obs, goals, (hx, cx))
        policy_np = policy.detach().cpu().numpy()
        acts = np.array([np.random.choice(A_SIZE, p=policy_np[i]) for i in range(env.num_agents)], dtype=np.int64)
        next_obs, next_goals, reward, done, info = env.step(acts)

        steps_taken += 1
        block_count += int(np.asarray(info["blocking"]).sum())
        vm = np.asarray(info["valids_mask"])
        for i, a in enumerate(acts):
            if vm[i, a] == 0:
                invalid_choices += 1
        last_on_goal = np.asarray(info["on_goal"])

        rewards.append(reward.astype(np.float32))
        dones = np.array([done] * env.num_agents, dtype=bool)
        dones_list.append(dones)
        values_torch.append(value)
        values_np.append(value.detach().squeeze(-1).cpu().numpy())
        policies.append(policy)
        actions_list.append(torch.from_numpy(acts).to(device))

        blocking_preds.append(blocking)
        on_goal_preds.append(on_goal)
        valids_preds.append(valids)
        blocking_tgts.append(torch.from_numpy(info["blocking"]).view(-1,1).to(device))
        on_goal_tgts.append(torch.from_numpy(info["on_goal"]).view(-1,1).to(device))
        valids_tgts.append(torch.from_numpy(info["valids_mask"]).to(device))

        obs, goals = next_obs.to(device), next_goals.to(device)
        if done:
            break

    returns, advantages = compute_returns_and_advantages(rewards, values_np, dones_list, gamma=GAMMA)

    policies_tensor  = torch.cat(policies, dim=0).to(device)
    actions_tensor   = torch.cat(actions_list, dim=0).long().to(device)
    returns_tensor   = torch.from_numpy(np.stack(returns).reshape(-1)).to(device)
    advantages_tensor= torch.from_numpy(np.stack(advantages).reshape(-1)).to(device)
    values_pred_tensor = torch.cat(values_torch, dim=0).view(-1,1)

    blocking_pred_tensor = torch.cat(blocking_preds, dim=0)
    blocking_tgt_tensor  = torch.cat(blocking_tgts, dim=0)
    on_goal_pred_tensor  = torch.cat(on_goal_preds, dim=0)
    on_goal_tgt_tensor   = torch.cat(on_goal_tgts, dim=0)
    valids_pred_tensor   = torch.cat(valids_preds, dim=0)
    valids_tgt_tensor    = torch.cat(valids_tgts, dim=0)

    ep_return_total = float(np.sum(np.stack(rewards)))

    loss_val = rl_loss_batch(policies_tensor, values_pred_tensor,
                             actions_tensor, returns_tensor, advantages_tensor,
                             blocking_pred_tensor, blocking_tgt_tensor,
                             on_goal_pred_tensor, on_goal_tgt_tensor,
                             valids_pred_tensor, valids_tgt_tensor,
                             entropy_beta=entropy_beta)

    N = env.num_agents
    ep_len = steps_taken
    ep_success = 1 if (last_on_goal is not None and last_on_goal.all()) else 0
    block_rate = (block_count / float(max(1, ep_len * N)))
    invalid_rate = (invalid_choices / float(max(1, ep_len * N)))

    return loss_val, ep_return_total, ep_len, ep_success, block_rate, invalid_rate


def run_il_episode(env, model, max_steps=MAX_STEPS_PER_EP, device=torch.device("cpu")):
    obs, goals = env.reset()
    obs, goals = obs.to(device), goals.to(device)
    hx, cx = model.init_hidden(env.num_agents)
    hx, cx = hx.to(device), cx.to(device)

    actions_seq = plan_expert_trajectory(env, max_steps=max_steps)
    T = actions_seq.shape[0]
    policies, expert_actions_list = [], []

    il_rewards = []
    il_match = 0
    il_total = 0

    block_count = 0
    invalid_choices = 0
    steps_taken = 0
    last_on_goal = None

    for t in range(T):
        expert_actions = actions_seq[t]
        policy, value, (hx, cx), _, _, valids = model(obs, goals, (hx, cx))

        with torch.no_grad():
            il_match += (policy.argmax(dim=1).cpu().numpy() == expert_actions).sum()
            il_total += env.num_agents

        next_obs, next_goals, reward, done, info = env.step(expert_actions)

        steps_taken += 1
        block_count += int(np.asarray(info["blocking"]).sum())
        vm = np.asarray(info["valids_mask"])
        for i, a in enumerate(expert_actions):
            if vm[i, a] == 0:
                invalid_choices += 1
        last_on_goal = np.asarray(info["on_goal"])

        il_rewards.append(reward.astype(np.float32))
        policies.append(policy)
        expert_actions_list.append(torch.from_numpy(expert_actions.astype(np.int64)).to(device))
        obs, goals = next_obs.to(device), next_goals.to(device)
        if done:
            break

    if not policies:
        raise PlannerFailed("empty IL episode")

    policies_tensor = torch.cat(policies, dim=0).to(device)
    expert_actions_tensor = torch.cat(expert_actions_list, dim=0).long().to(device)
    il_return_total = float(np.sum(np.stack(il_rewards))) if il_rewards else 0.0
    il_acc = (il_match / max(il_total, 1)) if il_total else 0.0

    N = env.num_agents
    ep_len = steps_taken
    ep_success = 1 if (last_on_goal is not None and last_on_goal.all()) else 0
    block_rate = (block_count / float(max(1, ep_len * N)))
    invalid_rate = (invalid_choices / float(max(1, ep_len * N)))

    loss_val = il_loss_batch(policies_tensor, expert_actions_tensor)

    return loss_val, il_return_total, il_acc, ep_len, ep_success, block_rate, invalid_rate


def worker_proc(wid, global_model, optimizer, global_ep, done_flag, metrics_q):
    torch.manual_seed(1234 + wid)
    random.seed(1234 + wid)
    np.random.seed(1234 + wid)

    local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PrimalEnvWrapper(size=(GRID_SIZE, GRID_SIZE*4), num_agents=NUM_AGENTS)
    local_model = ACNet(f"worker{wid}", a_size=A_SIZE, grid_size=GRID_SIZE, training=True).to(local_device)
    local_model.load_state_dict(global_model.state_dict())

    while True:
        with global_ep.get_lock():
            if global_ep.value >= MAX_EPISODES:
                done_flag.value = 1
                break
            ep = global_ep.value + 1
            global_ep.value += 1

        lr_now = BASE_LR / np.sqrt(max(ep, 1))
        entb_now = ENTROPY_BETA0 / np.sqrt(max(ep, 1))
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        use_il = (random.random() < P_IL)

        try:
            if use_il:
                loss_val, il_R, il_acc, ep_len, ep_succ, block_r, invalid_r = run_il_episode(
                    env, local_model, device=local_device)
                mode = "IL"
                R_str = f"{il_R:.2f} (acc={il_acc:.2f})"
            else:
                loss_val, ep_R, ep_len, ep_succ, block_r, invalid_r = run_rl_episode(
                    env, local_model, device=local_device, entropy_beta=entb_now)
                mode = "RL"
                R_str = f"{ep_R:.2f}"
        except PlannerFailed as e:
            print(f"[W{wid}] Episode {ep} | [IL] skipped: {e}", flush=True)
            continue

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=GRAD_CLIP)

        for lp, gp in zip(local_model.parameters(), global_model.parameters()):
            if lp.grad is None:
                continue
            g = lp.grad.detach().cpu()
            if gp.grad is None:
                gp.grad = g.clone()
            else:
                gp.grad.copy_(g)

        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

        print(f"[W{wid}] Episode {ep} | mode={mode} | loss={float(loss_val.item()):.3f} | R={R_str}", flush=True)

        pkt = {
            "mode": mode,
            "len": ep_len,
            "succ": ep_succ,
            "block_rate": block_r,
            "invalid_rate": invalid_r,
            "loss": float(loss_val.item()),
            "return": (ep_R if mode == "RL" else None),
        }
        metrics_q.put_nowait(pkt)

        if ep % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"primal_ep{ep}.pt")
            global_model.save(path)
            print(f"[W{wid}] Saved checkpoint to {path}", flush=True)


def consume_metrics(q: mp.Queue, stop_flag: mp.Event):
    hist = defaultdict(deque)
    counted = 0
    while not stop_flag.is_set():
        try:
            m = q.get(timeout=0.2)
        except:
            continue
        counted += 1

        def push(name, val):
            dq = hist[name]
            dq.append(float(val))
            if len(dq) > METRICS_WINDOW:
                dq.popleft()

        if "len" in m and m["len"] is not None:
            push("len", m["len"])
        if "succ" in m and m["succ"] is not None:
            push("succ", m["succ"])
        if "block_rate" in m and m["block_rate"] is not None:
            push("block_rate", m["block_rate"])
        if "invalid_rate" in m and m["invalid_rate"] is not None:
            push("invalid_rate", m["invalid_rate"])
        if m.get("mode") == "RL" and m.get("return") is not None:
            push("rl_return", m["return"])
            push("rl_loss", m["loss"])
        if m.get("mode") == "IL":
            push("il_loss", m["loss"])

        if counted % PRINT_EVERY_PACKETS == 0:
            mean_len   = (sum(hist["len"]) / max(1, len(hist["len"]))) if "len" in hist else float("nan")
            succ_rate  = (sum(hist["succ"]) / max(1, len(hist["succ"]))) if "succ" in hist else float("nan")
            block_r    = (sum(hist["block_rate"]) / max(1, len(hist["block_rate"]))) if "block_rate" in hist else float("nan")
            inv_r      = (sum(hist["invalid_rate"]) / max(1, len(hist["invalid_rate"]))) if "invalid_rate" in hist else float("nan")
            rl_ret     = (sum(hist["rl_return"]) / max(1, len(hist["rl_return"]))) if "rl_return" in hist else float("nan")
            rl_loss    = (sum(hist["rl_loss"]) / max(1, len(hist["rl_loss"]))) if "rl_loss" in hist else float("nan")
            il_loss    = (sum(hist["il_loss"]) / max(1, len(hist["il_loss"]))) if "il_loss" in hist else float("nan")

            print(f"[GLOBAL] mean_len={mean_len:.1f} | succ={succ_rate*100:.1f}% | "
                  f"block_rate={block_r:.3f} | invalid={inv_r:.3f} | "
                  f"RL: return={rl_ret:.1f}, loss={rl_loss:.3f} | IL: loss={il_loss:.3f}",
                  flush=True)


def main():
    mp.set_start_method('spawn', force=True)

    global_model = ACNet("global", a_size=A_SIZE, grid_size=GRID_SIZE, training=True).to(DEVICE_GLOBAL)
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=BASE_LR)

    global_ep = mp.Value('i', 0)
    done_flag = mp.Value('i', 0)

    metrics_q = mp.Queue(maxsize=2000)
    stop_ev = mp.Event()
    consumer = mp.Process(target=consume_metrics, args=(metrics_q, stop_ev), daemon=True)
    consumer.start()

    procs = []
    for wid in range(NUM_WORKERS):
        p = mp.Process(target=worker_proc, args=(wid, global_model, optimizer, global_ep, done_flag, metrics_q), daemon=True)
        p.start()
        procs.append(p)

    try:
        while done_flag.value == 0:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted, terminating workers...", flush=True)
    finally:
        for p in procs:
            p.join()
        stop_ev.set()
        consumer.join(timeout=2.0)

    final_path = os.path.join(CKPT_DIR, "primal_final.pt")
    global_model.save(final_path)
    print(f"Final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
