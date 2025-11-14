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
        self.env = PrimalEnvWrapper(
            num_agents=num_agents,
            observation_size=10,
            size=(world_size, world_size),
            prob=(obstacle_density, obstacle_density),
            diagonal_movement=False
        )

        self.num_agents = num_agents
        self.world_size = world_size

        # Episode data
        self.episode_step = 0
        self.max_steps = 256
        self.done = False

        # Colors for agents (HSV to RGB)
        self.colors = self._init_colors()

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

        # Update positions
        self.agent_positions = self.env.env.getPositions()

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


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_trained_model.py <model_checkpoint.pt> [mode] [output.gif]")
        print("\nModes:")
        print("  gif          - Save as GIF animation (default)")
        print("  interactive  - Step-by-step control")
        print("  show         - Auto-play with display window")
        print("\nExamples:")
        print("  # Save as GIF (default name: primal_demo.gif)")
        print("  python visualize_trained_model.py checkpoints/primal_ep1000.pt")
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
    mode = sys.argv[2] if len(sys.argv) > 2 else 'gif'
    gif_output = sys.argv[3] if len(sys.argv) > 3 else 'primal_demo.gif'

    print(f"Loading model from: {model_path}")
    viz = PRIMALVisualizer(
        model_path=model_path,
        num_agents=8,
        world_size=20,
        obstacle_density=0.2
    )

    if mode == 'interactive':
        viz.run_interactive()
    elif mode == 'show':
        viz.run_animated(interval=200, save_gif=None, show_plot=True)
    else:  # 'gif' or default
        viz.run_animated(interval=200, save_gif=gif_output, show_plot=False)
        print(f"\n✓ Done! GIF saved to: {gif_output}")


if __name__ == "__main__":
    main()
