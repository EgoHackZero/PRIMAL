#!/usr/bin/env python
"""
Simple demonstration of PRIMAL environment
This script creates a basic multi-agent pathfinding scenario and runs it
"""

import mapf_gym as MAPF_Env
import numpy as np
try:
    from od_mstar3 import cpp_mstar
except ImportError:
    cpp_mstar = None
from od_mstar3 import od_mstar as py_mstar

def create_simple_scenario():
    """Create a simple 2-agent pathfinding scenario"""
    SIZE = 10
    num_agents = 2
    
    # Create world with some obstacles
    world = np.zeros((SIZE, SIZE), dtype=int)
    world[4, 3:7] = -1  # Horizontal obstacle
    world[2:5, 5] = -1  # Vertical obstacle
    
    # Place agents
    world[0, 0] = 1  # Agent 1 at top-left
    world[SIZE-1, SIZE-1] = 2  # Agent 2 at bottom-right
    
    # Set goals (agents want to swap positions)
    goals = np.zeros((SIZE, SIZE), dtype=int)
    goals[SIZE-1, SIZE-1] = 1  # Agent 1 wants to go to bottom-right
    goals[0, 0] = 2  # Agent 2 wants to go to top-left
    
    return world, goals, num_agents

def run_manual_simulation():
    """Run a simple manual simulation of agents moving"""
    print("=" * 60)
    print("PRIMAL Environment Demo")
    print("=" * 60)
    
    # Create environment
    world, goals, num_agents = create_simple_scenario()
    
    print(f"\nCreated environment with {num_agents} agents")
    print(f"World size: {world.shape[0]}x{world.shape[1]}")
    
    env = MAPF_Env.MAPFEnv(num_agents, world0=world, goals0=goals, DIAGONAL_MOVEMENT=False)
    
    # Run simulation
    max_steps = 50
    step = 0
    
    print("\nStarting simulation...")
    print("-" * 60)
    
    while not env.finished and step < max_steps:
        step += 1
        actions = []
        
        # Get valid actions for each agent
        for agent_id in range(1, num_agents + 1):
            valid_actions = env._listNextValidActions(agent_id)
            action = valid_actions[0]  # Take first valid action (simple policy)
            actions.append((agent_id, action))
        
        # Execute all actions
        for agent_id, action in actions:
            state, reward, done, truncated, info = env.step((agent_id, action))
            on_goal = info.get("on_goal", False)
        
        if step % 5 == 0:
            print(f"Step {step}: Agents are navigating...")
        
        if env.finished:
            print(f"\n🎉 All agents reached their goals in {step} steps!")
            break
    
    if not env.finished:
        print(f"\n⚠️  Simulation ended after {max_steps} steps (agents may need more time)")
    
    print("\nSimulation complete!")
    print("=" * 60)

def test_cpp_pathfinder():
    """Test the C++ pathfinder directly"""
    print("\n" + "=" * 60)
    print("Testing C++ Pathfinder")
    print("=" * 60)
    
    # Create a simple scenario
    world = np.zeros((5, 5), dtype=int)
    world[2, :] = 1  # Middle row is obstacle
    world[2, 2] = 0  # Leave gap in middle
    
    init_pos = [(0, 0), (4, 4)]  # Two agents
    goals = [(4, 4), (0, 0)]  # Swap positions
    
    print("\nWorld map:")
    print(world)
    print(f"\nAgents start: {init_pos}")
    print(f"Agents goals: {goals}")
    
    try:
        # Call C++ pathfinder if available, otherwise fall back to Python implementation
        if cpp_mstar is not None:
            paths = cpp_mstar.find_path(world, init_pos, goals, inflation=1.0, time_limit=10.0)
        else:
            paths = py_mstar.find_path(world, init_pos, goals, connect_8=True, time_limit=10.0)
        
        print(f"\n✅ Found path with {len(paths[0])} steps for agent 1")
        print(f"✅ Found path with {len(paths[1])} steps for agent 2")
        print("\nPath for Agent 1:")
        for i, pos in enumerate(paths[0]):
            print(f"  Step {i}: {pos}")
            
    except Exception as e:
        print(f"\n❌ Pathfinding error: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PRIMAL Simple Demo")
    print("=" * 60)
    print("\nThis script demonstrates:")
    print("1. Creating a PRIMAL environment")
    print("2. Running a simple simulation")
    print("3. Testing the C++ pathfinder")
    
    # Run manual simulation
    run_manual_simulation()
    
    # Test C++ pathfinder
    test_cpp_pathfinder()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo explore more:")
    print("- Read README.md for detailed documentation")
    print("- Check mapf_gym.py for environment implementation")
    print("- Run mapf_gym_unittests.py for more test scenarios")
    print("=" * 60 + "\n")
