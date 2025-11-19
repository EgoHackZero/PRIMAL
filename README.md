# PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

Reinforcement learning code to train multiple agents to
collaboratively plan their paths in a 2D grid world, as
well as to test/visualize the learned policy on handcrafted
scenarios.

**NEW**: Please try the [brand new online interactive
demo](https://primalgrid.netlify.app/primal) of our trained
PRIMAL model! You can customize the grid size, add/remove
obstacle, add agents and assign them goals, and finally
run the model online and see the results.

## Quick Start

Get started quickly with the PyTorch implementation:

### 1. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install gymnasium numpy matplotlib imageio pillow networkx
```

### 2. Run Visualization on Custom Map
```bash
# Option 1: Simple script with predefined settings
python run_visualization.py

# Option 2: Full control with command-line arguments
python visualize_trained_model.py --map saved_environments/maze.csv \
  --start 1,5 --end 7,20 --agents 1 \
  ../checkpoints/a3c/primal_ep45000.pt paths result.png
```

### 3. Train Your Own Model
```bash
python train_primal_pytorch.py
```

Checkpoints will be saved to `../checkpoints/a2c/` every 1000 episodes.

### File list

#### PyTorch Implementation (Recommended)
- **train_primal_pytorch.py**: PyTorch training script using A2C algorithm with imitation learning
- **ACNet_pytorch.py**: PyTorch neural network architecture (Actor-Critic with LSTM)
- **visualize_trained_model.py**: **NEW** Visualization tool for trained models with support for:
  - Loading custom maps from CSV files
  - Setting custom start/end positions
  - Creating static path visualizations (PNG)
  - Creating animated GIFs
  - Interactive step-by-step mode
- **run_visualization.py**: Simple script to run visualization with predefined parameters
- **example_visualize.py**: Examples showing how to use visualization functions programmatically
- **primal_env_wrapper.py**: PyTorch-compatible environment wrapper
- **mapf_gym.py**: Multi-agent path planning gym environment

#### Legacy TensorFlow Implementation
- DRLMAPF_A3C_RNN.ipynb: Multi-agent training code (TensorFlow 1.x). Training
runs on GPU by default, change line "with tf.device("/gpu:0"):"
to "with tf.device("/cpu:0"):" to train on CPU (much slower).
- primal_testing.py: Code to run systematic validation tests
of PRIMAL, pulled from the saved_environments folder as .npy
files (examples available [here](https://drive.google.com/file/d/193mv6mhlcu9Bqxs6hSMTfSk_1GrPAiNO/view?usp=sharing)) and output results in a given
folder (by default: primal_results).
- mapf_gym_cap.py: Multi-agent path planning gym environment,
with capped goal distance state value for validation in
larger environments.
- mapgenerator.py: Script for creating custom environments and
testing a trained model on them. As an example, the trained
model used in our paper can be found [here](https://drive.google.com/file/d/1AtAeUwLF1Rn_X3b2FHkHi4fI5vveUHF6/view?usp=sharing).

## Visualization

The new PyTorch implementation includes a powerful visualization tool that allows you to test trained models on custom maps and visualize agent paths.

### Quick Start

#### Option 1: Using the Simple Script
```bash
python run_visualization.py
```

This will run the visualization with predefined settings:
- Map: `saved_environments/maze.csv`
- Start position: (1, 5)
- End position: (7, 20)
- Output: `maze_result.png`

#### Option 2: Command-Line Tool

**Random environment:**
```bash
python visualize_trained_model.py ../checkpoints/a3c/primal_ep45000.pt paths output.png
```

**Custom map with random positions:**
```bash
python visualize_trained_model.py --map saved_environments/maze.csv --agents 1 \
  ../checkpoints/a3c/primal_ep45000.pt paths result.png
```

**Custom map with specific start/end positions:**
```bash
python visualize_trained_model.py --map saved_environments/maze.csv \
  --start 1,5 --end 7,20 --agents 1 \
  ../checkpoints/a3c/primal_ep45000.pt paths maze_custom.png
```

**Create animated GIF:**
```bash
python visualize_trained_model.py --map saved_environments/maze.csv \
  --start 1,5 --end 7,20 --agents 1 \
  ../checkpoints/a3c/primal_ep45000.pt gif animation.gif
```

**Interactive mode (step-by-step):**
```bash
python visualize_trained_model.py --map saved_environments/maze.csv \
  --start 1,5 --end 7,20 --agents 1 \
  ../checkpoints/a3c/primal_ep45000.pt interactive
```

### Command-Line Arguments

- `model_path`: Path to trained model checkpoint (.pt file)
- `mode`: Visualization mode - `paths` (PNG), `gif`, `interactive`, or `show`
- `output_file`: Output filename (optional)
- `--map`: Path to saved map (CSV for obstacles only, NPY for full environment)
- `--start`: Start position as "x,y" (e.g., "1,5")
- `--end`: End/goal position as "x,y" (e.g., "7,20")
- `--agents`: Number of agents (default: 1)
- `--world-size`: World size for random environments (default: 20)
- `--obstacle-density`: Obstacle density for random environments (default: 0.2)

### Python API

You can also use the visualization functions programmatically:

```python
from visualize_trained_model import (
    PRIMALVisualizer,
    load_map_from_csv,
    create_custom_environment
)

# Load obstacle map
custom_world = load_map_from_csv('saved_environments/maze.csv')

# Create environment with custom start/end positions
custom_world, custom_goals = create_custom_environment(
    world=custom_world,
    start_pos=(1, 5),   # x=1, y=5
    end_pos=(7, 20),    # x=7, y=20
    num_agents=1
)

# Create visualizer
viz = PRIMALVisualizer(
    model_path='../checkpoints/a3c/primal_ep45000.pt',
    num_agents=1,
    custom_world=custom_world,
    custom_goals=custom_goals
)

# Save paths as PNG
viz.run_and_save_paths(output_path='result.png')
```

See `example_visualize.py` for more examples.

### Creating Custom Maps

Maps are CSV files with semicolon delimiters:
- `1` = free space
- `0` = obstacle

Example (`maze.csv`):
```csv
0;0;0;0;0;0
0;1;1;1;1;0
0;1;0;0;1;0
0;1;1;1;1;0
0;0;0;0;0;0
```

## Before compilation: compile cpp_mstar code

- cd into the od_mstar3 folder.
- python3 setup.py build_ext (may need --inplace as extra argument).
- copy so object from build/lib.*/ at the root of the od_mstar3 folder.
- Check by going back to the root of the git folder,
running python3 and "import cpp_mstar"

### Custom testing

Edit mapgenerator.py to the correct path for the model.
By default, the model is loaded from the model_primal folder.

Hotkeys:
- o: obstacle mode
- a: agent mod
- g: goal mode, click an agent then click a free tile to place its goal
- c: clear agents
- r: reset
- up/down arrows: change size
- p: pause inference

### Requirements

#### PyTorch Implementation (Recommended)
- Python 3.10+
- PyTorch 2.x (with CUDA support for GPU training)
- gymnasium (modern replacement for OpenAI Gym)
- numpy
- matplotlib
- imageio (for GIF creation)
- pillow (for image processing)
- networkx (for M* planner)

Install PyTorch dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium numpy matplotlib imageio pillow networkx
```

#### Legacy TensorFlow Implementation
- Python 3.4
- Cython 0.28.4
- OpenAI Gym 0.9.4
- Tensorflow 1.3.1
- Numpy 1.13.3
- matplotlib
- imageio (for GIFs creation)
- tk
- networkx (if using od_mstar.py and not the C++ version)

### Authors

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)

[Justin Kerr](jgkerr@andrew.cmu.edu)
