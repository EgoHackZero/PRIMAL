"""
Visualization for Trained PRIMAL Models
Shows agents navigating in real-time using matplotlib animation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb
import os

from ACNet_pytorch import ACNet
from primal_env_wrapper import PrimalEnvWrapper
from mapf_gym import MAPFEnv


def load_map_from_csv(csv_path):
    """
    Load obstacle map from CSV file.

    Args:
        csv_path: Path to CSV file (e.g., 'saved_environments/maze.csv')

    Returns:
        world: numpy array where -1 = obstacle, 0 = free space
    """
    import csv

    world_list = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            world_list.append([int(cell) for cell in row])

    world = np.array(world_list, dtype=int)
    # Convert: 1 -> free space (0), 0 -> obstacle (-1)
    world = np.where(world == 1, 0, -1)

    return world


def load_environment_from_npy(npy_path):
    """
    Load environment from .npy file (world and goals).

    Args:
        npy_path: Path to .npy file containing [world, goals]

    Returns:
        world: numpy array with obstacles (-1), free space (0), and agents (positive integers)
        goals: numpy array with agent goals (positive integers)
    """
    data = np.load(npy_path)
    world = data[0]
    goals = data[1]
    return world, goals


def create_custom_environment(world, start_pos, end_pos, num_agents=1):
    """
    Create environment with custom start and end positions.

    Args:
        world: Obstacle map (numpy array)
        start_pos: Tuple (x, y) for start position
        end_pos: Tuple (x, y) for end position
        num_agents: Number of agents

    Returns:
        world: numpy array with obstacles and agents placed
        goals: numpy array with goal positions
    """
    # Note: In the grid, positions are stored as (row, col) but user provides (x, y)
    # So we need to convert: (x, y) -> (y, x) for array indexing
    start_row, start_col = start_pos[1], start_pos[0]
    end_row, end_col = end_pos[1], end_pos[0]

    # Create copies
    world_with_agents = world.copy()
    goals = np.zeros_like(world)

    # Place agent at start position (agent ID = 1)
    world_with_agents[start_row, start_col] = 1

    # Place goal at end position
    goals[end_row, end_col] = 1

    return world_with_agents, goals


class PRIMALVisualizer:
    """Visualize trained PRIMAL agents navigating environment"""

    def __init__(self, model_path, num_agents=8, world_size=20, obstacle_density=0.2,
                 custom_world=None, custom_goals=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_agents: Number of agents in environment
            world_size: Size of grid world
            obstacle_density: Probability of obstacles (0-0.5)
            custom_world: Optional pre-loaded world map (numpy array)
            custom_goals: Optional pre-loaded goal positions (numpy array)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ACNet("global", a_size=5, grid_size=10, training=False)
        self.model.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.custom_world = custom_world
        self.custom_goals = custom_goals

        # Create environment
        if custom_world is not None:
            # Use custom world with MAPFEnv directly through wrapper
            # The wrapper will handle blank_world mode to randomize agent positions
            self.env_direct = MAPFEnv(
                num_agents=num_agents,
                observation_size=10,
                world0=custom_world.copy(),
                goals0=custom_goals,
                blank_world=(custom_goals is None)
            )
            self.env = None  # We'll use env_direct instead
            self.num_agents = num_agents
            self.world_size = custom_world.shape[0]
        else:
            # Use a range for obstacle density: (0, obstacle_density)
            # This allows the triangular distribution to work properly
            self.env = PrimalEnvWrapper(
                num_agents=num_agents,
                observation_size=10,
                size=(world_size, world_size),
                prob=(0.0, obstacle_density),
                diagonal_movement=False
            )
            self.env_direct = None
            self.num_agents = num_agents
            self.world_size = world_size

        # Episode data
        self.episode_step = 0
        self.max_steps = 1000
        self.done = False

        # Colors for agents (HSV to RGB)
        self.colors = self._init_colors()

        # Path tracking for static visualization
        self.agent_paths = [[] for _ in range(num_agents)]

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.reset_episode()

    def _init_colors(self):
        """Generate distinct colors for each agent"""
        colors = []
        for i in range(self.num_agents):
            hue = i / self.num_agents
            rgb = hsv_to_rgb([hue, 1.0, 1.0])
            colors.append(rgb)
        return colors

    def reset_episode(self):
        """Reset environment and start new episode"""
        if self.env_direct is not None:
            # Using custom world - reset directly
            # Use blank_world=True if goals are None (will randomize agents and goals)
            self.env_direct._setWorld(
                self.custom_world.copy(),
                self.custom_goals,
                blank_world=(self.custom_goals is None)
            )

            # Get observations manually
            obs_list = []
            goals_list = []
            for agent_id in range(1, self.num_agents + 1):
                channels, goal_vec = self.env_direct._observe(agent_id)
                processed = [np.asarray(ch, dtype=np.float32) for ch in channels]
                maps = np.stack(processed, axis=0)
                goal = np.asarray(goal_vec, dtype=np.float32)
                obs_list.append(maps)
                goals_list.append(goal)

            obs_array = np.stack(obs_list, axis=0).astype(np.float32)
            goals_array = np.stack(goals_list, axis=0).astype(np.float32)

            self.obs = torch.from_numpy(obs_array).to(self.device)
            self.goals = torch.from_numpy(goals_array).to(self.device)

            # Get world state
            self.world = self.env_direct.world.state.copy()
            self.goal_positions = self.env_direct.getGoals()
            self.agent_positions = self.env_direct.getPositions()
        else:
            # Using wrapper environment
            obs, goals = self.env.reset()
            self.obs = obs.to(self.device)
            self.goals = goals.to(self.device)

            # Get world state
            self.world = self.env.env.world.state.copy()
            self.goal_positions = self.env.env.getGoals()
            self.agent_positions = self.env.env.getPositions()

        self.hx, self.cx = self.model.init_hidden(self.num_agents)
        self.hx = self.hx.to(self.device)
        self.cx = self.cx.to(self.device)

        self.episode_step = 0
        self.done = False

        # Update actual world size from environment
        self.world_size = self.world.shape[0]

        # Reset path tracking
        self.agent_paths = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.agent_paths[i].append(self.agent_positions[i])

    def step(self):
        """Execute one step of the policy"""
        if self.done:
            return False

        # Get actions from model
        with torch.no_grad():
            policy, value, (self.hx, self.cx), blocking, on_goal, valids = \
                self.model(self.obs, self.goals, (self.hx, self.cx))

        # Sample actions from policy
        policy_np = policy.detach().cpu().numpy()
        actions = np.array([np.random.choice(5, p=policy_np[i])
                           for i in range(self.num_agents)])

        # Execute actions
        if self.env_direct is not None:
            # Using direct environment
            done_flags = []
            for agent_id, action in enumerate(actions, start=1):
                _, _, done, _, _, _, _ = self.env_direct._step((agent_id, int(action)))
                done_flags.append(done)

            self.done = any(done_flags)

            # Get observations manually
            obs_list = []
            goals_list = []
            for agent_id in range(1, self.num_agents + 1):
                channels, goal_vec = self.env_direct._observe(agent_id)
                processed = [np.asarray(ch, dtype=np.float32) for ch in channels]
                maps = np.stack(processed, axis=0)
                goal = np.asarray(goal_vec, dtype=np.float32)
                obs_list.append(maps)
                goals_list.append(goal)

            obs_array = np.stack(obs_list, axis=0).astype(np.float32)
            goals_array = np.stack(goals_list, axis=0).astype(np.float32)

            self.obs = torch.from_numpy(obs_array).to(self.device)
            self.goals = torch.from_numpy(goals_array).to(self.device)

            # Update positions
            self.agent_positions = self.env_direct.getPositions()
        else:
            # Using wrapper environment
            self.obs, self.goals, rewards, self.done, info = self.env.step(actions)
            self.obs = self.obs.to(self.device)
            self.goals = self.goals.to(self.device)

            # Update positions
            self.agent_positions = self.env.env.getPositions()

        # Track paths
        for i in range(self.num_agents):
            self.agent_paths[i].append(self.agent_positions[i])

        self.episode_step += 1

        if self.episode_step >= self.max_steps:
            self.done = True

        return True

    def draw_grid(self):
        """Draw the environment grid"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.world_size - 0.5)
        self.ax.set_ylim(-0.5, self.world_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.world_size))
        self.ax.set_yticks(range(self.world_size))
        self.ax.grid(True, alpha=0.3, linewidth=0.5)

        # Draw obstacles
        for i in range(self.world_size):
            for j in range(self.world_size):
                if self.world[i, j] == -1:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            linewidth=0, edgecolor='none',
                                            facecolor='gray', alpha=0.8)
                    self.ax.add_patch(rect)

        # Draw goals (stars)
        for agent_id in range(self.num_agents):
            goal = self.goal_positions[agent_id]
            self.ax.plot(goal[1], goal[0], marker='*', markersize=20,
                        color=self.colors[agent_id], markeredgecolor='black',
                        markeredgewidth=1.5, zorder=2)

        # Draw agents (circles)
        for agent_id in range(self.num_agents):
            pos = self.agent_positions[agent_id]
            circle = patches.Circle((pos[1], pos[0]), 0.35,
                                   facecolor=self.colors[agent_id],
                                   edgecolor='black', linewidth=2, zorder=3)
            self.ax.add_patch(circle)

            # Check if on goal
            if pos == self.goal_positions[agent_id]:
                # Draw checkmark or highlight
                self.ax.plot(pos[1], pos[0], marker='o', markersize=25,
                           color='none', markeredgecolor='lime',
                           markeredgewidth=3, zorder=4)

        # Title with episode info
        on_goal_count = sum(1 for i in range(self.num_agents)
                           if self.agent_positions[i] == self.goal_positions[i])
        title = f'PRIMAL - Step {self.episode_step}/{self.max_steps} | '
        title += f'Agents on Goal: {on_goal_count}/{self.num_agents}'
        if self.done:
            title += ' | DONE'
        self.ax.set_title(title, fontsize=14, fontweight='bold')

        # Invert y-axis to match typical grid representation
        self.ax.invert_yaxis()

    def draw_paths(self):
        """Draw the environment with agent paths as a static image"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.world_size - 0.5)
        self.ax.set_ylim(-0.5, self.world_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.world_size))
        self.ax.set_yticks(range(self.world_size))
        self.ax.grid(True, alpha=0.3, linewidth=0.5)

        # Draw obstacles
        for i in range(self.world_size):
            for j in range(self.world_size):
                if self.world[i, j] == -1:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            linewidth=0, edgecolor='none',
                                            facecolor='gray', alpha=0.8)
                    self.ax.add_patch(rect)

        # Draw agent paths (lines connecting positions)
        for agent_id in range(self.num_agents):
            path = self.agent_paths[agent_id]
            if len(path) > 1:
                # Extract y and x coordinates (note: positions are (row, col) = (y, x))
                path_y = [pos[0] for pos in path]
                path_x = [pos[1] for pos in path]
                self.ax.plot(path_x, path_y, color=self.colors[agent_id],
                           linewidth=2, alpha=0.5, linestyle='-', marker='o',
                           markersize=2, markerfacecolor=self.colors[agent_id])

        # Draw start positions (filled squares)
        for agent_id in range(self.num_agents):
            if len(self.agent_paths[agent_id]) > 0:
                start = self.agent_paths[agent_id][0]
                # Draw filled square for start position
                square = patches.Rectangle((start[1] - 0.4, start[0] - 0.4), 0.8, 0.8,
                                          facecolor=self.colors[agent_id],
                                          edgecolor='black', linewidth=2, zorder=5)
                self.ax.add_patch(square)

        # Draw goal positions (filled squares with different shade or pattern)
        for agent_id in range(self.num_agents):
            goal = self.goal_positions[agent_id]
            # Draw filled square for goal position with slightly lighter color
            square = patches.Rectangle((goal[1] - 0.4, goal[0] - 0.4), 0.8, 0.8,
                                      facecolor=self.colors[agent_id],
                                      edgecolor='black', linewidth=2,
                                      alpha=0.7, zorder=4)
            self.ax.add_patch(square)
            # Add a small dot in center to distinguish from start
            self.ax.plot(goal[1], goal[0], marker='o', markersize=8,
                        color='white', markeredgecolor='black',
                        markeredgewidth=1, zorder=6)

        # Title with episode info
        on_goal_count = sum(1 for i in range(self.num_agents)
                           if len(self.agent_paths[i]) > 0 and
                           self.agent_paths[i][-1] == self.goal_positions[i])
        title = f'PRIMAL Paths - Total Steps: {self.episode_step} | '
        title += f'Agents Reached Goal: {on_goal_count}/{self.num_agents}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')

        # Invert y-axis to match typical grid representation
        self.ax.invert_yaxis()

    def animate_step(self, frame):
        """Animation callback for FuncAnimation"""
        if not self.done:
            self.step()
        self.draw_grid()
        return []

    def run_interactive(self):
        """Run visualization with step-by-step control"""
        self.draw_grid()
        plt.show(block=False)

        print("\n=== PRIMAL Visualization ===")
        print("Commands:")
        print("  [Enter] - Execute one step")
        print("  'r' - Reset episode")
        print("  'q' - Quit")
        print("  'a' - Auto-run episode")
        print("============================\n")

        while True:
            try:
                cmd = input(f"Step {self.episode_step}: ").strip().lower()

                if cmd == 'q':
                    break
                elif cmd == 'r':
                    self.reset_episode()
                    self.draw_grid()
                    plt.pause(0.01)
                    print("Episode reset!")
                elif cmd == 'a':
                    # Auto-run
                    while not self.done:
                        self.step()
                        self.draw_grid()
                        plt.pause(0.1)
                    print(f"Episode finished in {self.episode_step} steps")
                else:
                    # Execute single step
                    if not self.done:
                        self.step()
                        self.draw_grid()
                        plt.pause(0.01)
                    else:
                        print("Episode done! Press 'r' to reset.")

            except KeyboardInterrupt:
                break

        plt.close()

    def run_animated(self, interval=200, save_gif=None, show_plot=True):
        """
        Run visualization as animation

        Args:
            interval: Milliseconds between frames
            save_gif: If provided, save animation to this filename
            show_plot: Whether to display the plot window
        """
        anim = FuncAnimation(self.fig, self.animate_step,
                           frames=self.max_steps,
                           interval=interval, repeat=False)

        if save_gif:
            print(f"Saving animation to {save_gif}...")
            print("This may take a while...")
            anim.save(save_gif, writer='pillow', fps=5, dpi=100)
            print(f"Animation saved to {save_gif}!")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def run_and_save_paths(self, output_path='primal_paths.png'):
        """
        Run episode and save a static image showing all agent paths

        Args:
            output_path: Filename to save the static image
        """
        print(f"Running episode to generate paths...")

        # Run the entire episode
        while not self.done:
            self.step()
            if self.episode_step % 50 == 0:
                print(f"  Step {self.episode_step}/{self.max_steps}...")

        print(f"Episode completed in {self.episode_step} steps")

        # Draw the paths
        self.draw_paths()

        # Save the figure
        print(f"Saving paths visualization to {output_path}...")
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Paths image saved to: {output_path}")

        plt.close()


def main():
    """Example usage"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize trained PRIMAL model navigating in an environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random environment - save paths as PNG
  python visualize_trained_model.py checkpoints/primal_ep1000.pt

  # Use saved map (CSV) - randomize agent positions
  python visualize_trained_model.py checkpoints/primal_ep1000.pt --map saved_environments/maze.csv --agents 4 paths maze_test.png

  # Use saved environment (NPY with world and goals)
  python visualize_trained_model.py checkpoints/primal_ep1000.pt --map saved_environments/env.npy paths result.png

  # Save as GIF with custom name
  python visualize_trained_model.py checkpoints/primal_ep1000.pt gif my_demo.gif

  # Interactive step-by-step
  python visualize_trained_model.py checkpoints/primal_ep1000.pt interactive

  # Auto-play with display (no save)
  python visualize_trained_model.py checkpoints/primal_ep1000.pt show
        """)

    parser.add_argument('model_path', type=str,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('mode', nargs='?', default='paths',
                        choices=['paths', 'gif', 'interactive', 'show'],
                        help='Visualization mode (default: paths)')
    parser.add_argument('output_file', nargs='?', default=None,
                        help='Output filename (PNG for paths mode, GIF for gif mode)')
    parser.add_argument('--map', type=str, default=None,
                        help='Path to saved map (CSV for obstacles only, NPY for full environment)')
    parser.add_argument('--agents', type=int, default=1,
                        help='Number of agents (default: 1)')
    parser.add_argument('--world-size', type=int, default=20,
                        help='World size for random environments (default: 20, ignored if --map is provided)')
    parser.add_argument('--obstacle-density', type=float, default=0.2,
                        help='Obstacle density for random environments (default: 0.2, ignored if --map is provided)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start position as "x,y" (e.g., "1,5"). Only works with --map CSV files.')
    parser.add_argument('--end', type=str, default=None,
                        help='End/goal position as "x,y" (e.g., "7,20"). Only works with --map CSV files.')

    args = parser.parse_args()

    # Set default output filename based on mode
    if args.output_file is None:
        if args.mode == 'gif':
            args.output_file = 'primal_demo.gif'
        elif args.mode == 'paths':
            args.output_file = 'primal_paths.png'

    # Load custom map if provided
    custom_world = None
    custom_goals = None
    if args.map:
        if args.map.endswith('.csv'):
            print(f"Loading map from CSV: {args.map}")
            custom_world = load_map_from_csv(args.map)
            print(f"Loaded map with shape: {custom_world.shape}")

            # Check if custom start/end positions are provided
            if args.start and args.end:
                # Parse start and end positions
                try:
                    start_x, start_y = map(int, args.start.split(','))
                    end_x, end_y = map(int, args.end.split(','))
                    print(f"Using custom start position: ({start_x}, {start_y})")
                    print(f"Using custom end position: ({end_x}, {end_y})")

                    # Create environment with custom positions
                    custom_world, custom_goals = create_custom_environment(
                        custom_world, (start_x, start_y), (end_x, end_y), args.agents
                    )
                except ValueError:
                    print("Error: Invalid start/end format. Use 'x,y' format (e.g., '1,5')")
                    sys.exit(1)
            else:
                custom_goals = None  # Will randomize agent positions and goals
        elif args.map.endswith('.npy'):
            print(f"Loading environment from NPY: {args.map}")
            custom_world, custom_goals = load_environment_from_npy(args.map)
            print(f"Loaded world with shape: {custom_world.shape}")
            # Count agents from the environment
            args.agents = np.max(custom_world[custom_world > 0])
            print(f"Environment has {args.agents} agents")

            if args.start or args.end:
                print("Warning: --start and --end are ignored for .npy files (positions already defined)")
        else:
            print(f"Error: Unsupported file format. Use .csv or .npy")
            sys.exit(1)

    print(f"Loading model from: {args.model_path}")
    viz = PRIMALVisualizer(
        model_path=args.model_path,
        num_agents=args.agents,
        world_size=args.world_size,
        obstacle_density=args.obstacle_density,
        custom_world=custom_world,
        custom_goals=custom_goals
    )

    if args.mode == 'interactive':
        viz.run_interactive()
    elif args.mode == 'show':
        viz.run_animated(interval=200, save_gif=None, show_plot=True)
    elif args.mode == 'gif':
        viz.run_animated(interval=200, save_gif=args.output_file, show_plot=False)
        print(f"\n[DONE] GIF saved to: {args.output_file}")
    else:  # 'paths' or default
        viz.run_and_save_paths(output_path=args.output_file)
        print(f"\n[DONE] Paths image saved to: {args.output_file}")


if __name__ == "__main__":
    main()