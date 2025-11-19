"""
Simple script to run visualization with maze.csv
Start position: (1, 5)
End position: (7, 20)
"""

from visualize_trained_model import (
    PRIMALVisualizer,
    load_map_from_csv,
    create_custom_environment
)

# Load the obstacle map from CSV
custom_world = load_map_from_csv('saved_environments/maze.csv')
print(f"Loaded map with shape: {custom_world.shape}")

# Set start and end positions
start_pos = (1, 5)   # x=1, y=5
end_pos = (7, 20)    # x=7, y=20
print(f"Start position: {start_pos}")
print(f"End position: {end_pos}")

# Create environment with custom positions
custom_world, custom_goals = create_custom_environment(
    world=custom_world,
    start_pos=start_pos,
    end_pos=end_pos,
    num_agents=1
)

# Create visualizer
viz = PRIMALVisualizer(
    model_path='../checkpoints/a3c/primal_ep45000.pt',
    num_agents=1,
    custom_world=custom_world,
    custom_goals=custom_goals
)

# Run and save paths as PNG
viz.run_and_save_paths(output_path='maze_result.png')
print("\nDone! Saved to: maze_result.png")
