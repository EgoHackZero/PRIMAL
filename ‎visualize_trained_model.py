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

from ACNet_pytorch import ACNet
from primal_env_wrapper import PrimalEnvWrapper


class PRIMALVisualizer:
    """Visualize trained PRIMAL agents navigating environment"""

    def __init__(self, model_path, num_agents=8, world_size=20, obstacle_density=0.2):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_agents: Number of agents in environment
            world_size: Size of grid world
            obstacle_density: Probability of obstacles (0-0.5)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ACNet("global", a_size=5, grid_size=10, training=False)
        self.model.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create environment
        # Use a range for obstacle density: (0, obstacle_density)
        # This allows the triangular distribution to work properly
        self.env = PrimalEnvWrapper(
            num_agents=num_agents,
            observation_size=10,
            size=(world_size, world_size),
            prob=(0.0, obstacle_density),
            diagonal_movement=False
        )

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
        obs, goals = self.env.reset()
        self.obs = obs.to(self.device)
        self.goals = goals.to(self.device)

        self.hx, self.cx = self.model.init_hidden(self.num_agents)
        self.hx = self.hx.to(self.device)
        self.cx = self.cx.to(self.device)

        self.episode_step = 0
        self.done = False

        # Get world state
        self.world = self.env.env.world.state.copy()
        self.goal_positions = self.env.env.getGoals()
        self.agent_positions = self.env.env.getPositions()

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
        self.obs, self.goals, rewards, self.done, info = self.env.step(actions)
        self.obs = self.obs.to(self.device)
        self.goals = self.goals.to(self.device)

        # Update positions and track paths
        self.agent_positions = self.env.env.getPositions()
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
                           linewidth=2, alpha=0.7, linestyle='-', marker='o',
                           markersize=3, markerfacecolor=self.colors[agent_id])

        # Draw start positions (circles)
        for agent_id in range(self.num_agents):
            if len(self.agent_paths[agent_id]) > 0:
                start = self.agent_paths[agent_id][0]
                circle = patches.Circle((start[1], start[0]), 0.35,
                                       facecolor=self.colors[agent_id],
                                       edgecolor='black', linewidth=2, zorder=3)
                self.ax.add_patch(circle)

        # Draw goal positions (stars)
        for agent_id in range(self.num_agents):
            goal = self.goal_positions[agent_id]
            self.ax.plot(goal[1], goal[0], marker='*', markersize=20,
                        color=self.colors[agent_id], markeredgecolor='black',
                        markeredgewidth=1.5, zorder=4)

        # Draw final positions (outlined if on goal)
        for agent_id in range(self.num_agents):
            if len(self.agent_paths[agent_id]) > 0:
                final = self.agent_paths[agent_id][-1]
                if final == self.goal_positions[agent_id]:
                    # Draw checkmark for agents that reached goal
                    self.ax.plot(final[1], final[0], marker='o', markersize=25,
                               color='none', markeredgecolor='lime',
                               markeredgewidth=3, zorder=5)

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

    if len(sys.argv) < 2:
        print("Usage: python visualize_trained_model.py <model_checkpoint.pt> [mode] [output_file]")
        print("\nModes:")
        print("  paths        - Save agent paths as static PNG image (default)")
        print("  gif          - Save as GIF animation")
        print("  interactive  - Step-by-step control")
        print("  show         - Auto-play with display window")
        print("\nExamples:")
        print("  # Save paths as PNG (default name: primal_paths.png)")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt")
        print("")
        print("  # Save paths with custom name")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt paths my_paths.png")
        print("")
        print("  # Save as GIF with custom name")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt gif my_demo.gif")
        print("")
        print("  # Interactive step-by-step")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt interactive")
        print("")
        print("  # Auto-play with display (no save)")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt show")
        sys.exit(1)

    model_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'paths'

    # Set default output filename based on mode
    if mode == 'gif':
        default_output = 'primal_demo.gif'
    elif mode == 'paths':
        default_output = 'primal_paths.png'
    else:
        default_output = None

    output_file = sys.argv[3] if len(sys.argv) > 3 else default_output

    print(f"Loading model from: {model_path}")
    viz = PRIMALVisualizer(
        model_path=model_path,
        num_agents=1,
        world_size=20,
        obstacle_density=0.2
    )

    if mode == 'interactive':
        viz.run_interactive()
    elif mode == 'show':
        viz.run_animated(interval=200, save_gif=None, show_plot=True)
    elif mode == 'gif':
        viz.run_animated(interval=200, save_gif=output_file, show_plot=False)
        print(f"\n[DONE] GIF saved to: {output_file}")
    else:  # 'paths' or default
        viz.run_and_save_paths(output_path=output_file)
        print(f"\n[DONE] Paths image saved to: {output_file}")


if __name__ == "__main__":
    main()