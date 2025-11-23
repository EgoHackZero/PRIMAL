# PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

Reinforcement learning code to train multiple agents to collaboratively plan their paths in a 2D grid world, as well as to test/visualize the learned policy on handcrafted scenarios.

## Table of Contents

- [Quick Start](#quick-start)
- [Visualization](#visualization)
- [Training](#training)
- [Resume Training](#resume-training-from-checkpoint)
- [File Structure](#file-structure)
- [Requirements](#requirements)

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
python train_primal_pytorch_a3c.py
```

Checkpoints will be saved to `../checkpoints/a2c/` every 1000 episodes.

## Visualization

The PyTorch implementation includes a powerful visualization tool that allows you to test trained models on custom maps and visualize agent paths.

### Visualization Modes

1. **Static Path Image (PNG)** - Shows complete agent trajectories
2. **Animated GIF** - Step-by-step animation
3. **Interactive Mode** - Manual control with keyboard
4. **Live Display** - Real-time auto-play

### Quick Visualization Examples

All commands should be run from the PRIMAL directory using the `primal` conda environment.

```bash
cd D:\astudy\NMvR\PRIMAL
conda activate primal
```

#### 1. Generate Static Path Image (PNG)

```bash
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths my_paths.png
```

**Output:** `my_paths.png` - Shows agent trajectories with colored lines

#### 2. Create Animated GIF

```bash
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt gif demo.gif
```

**Output:** `demo.gif` - Animated visualization (may take a few minutes to generate)

#### 3. Interactive Step-by-Step Mode

```bash
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt interactive
```

**Controls:**
- `[Enter]` - Execute one step
- `r` - Reset episode
- `q` - Quit
- `a` - Auto-run entire episode

#### 4. Auto-Play with Live Display

```bash
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt show
```

**Output:** Live matplotlib window (no file saved)

### Custom Map Visualization

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

### Windows-Specific Examples

#### Using Full Paths (Recommended for Windows)

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate primal

# Change to PRIMAL directory
cd "D:\astudy\NMvR\PRIMAL"

# Generate paths visualization
python visualize_trained_model.py "D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt" paths output.png
```

#### One-Liner Command

```bash
cd "D:\astudy\NMvR\PRIMAL" && source ~/miniconda3/etc/profile.d/conda.sh && conda activate primal && python visualize_trained_model.py "D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt" paths visualization_output.png
```

### Testing Different Checkpoints

```bash
# Test early training checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep1000.pt paths ep1000_paths.png

# Test mid training checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep5000.pt paths ep5000_paths.png

# Test final checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths ep7000_paths.png
```

### Troubleshooting

#### Error: "No such file or directory"
Make sure you're in the correct directory and using quotes around paths with spaces.

#### Error: "No module named..."
Activate the primal conda environment first:
```bash
conda activate primal
```

#### Visualization window doesn't appear
For WSL/remote systems, make sure X11 forwarding is set up, or use `paths` or `gif` mode instead.

### Output Files Location

All output files are saved in the current working directory (PRIMAL folder) unless you specify a full path:

```bash
# Save to specific location
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths "D:\outputs\my_visualization.png"
```

## Training

### Start Training from Scratch

```bash
python train_primal_pytorch_a3c.py
```

Training configuration (hardcoded in script):
- Grid size: 10x40 (dynamic: 10, 40, or 70)
- Agents: 8
- Max episodes: 100,000
- Checkpoints: saved every 1000 episodes to `../checkpoints/`
- IL probability: 50%
- Base learning rate: 2e-5 (decays as lr = BASE_LR / sqrt(episode))

## Resume Training from Checkpoint

The training script supports resuming from a checkpoint using the `PRIMAL_RESUME_FROM` environment variable.

### Basic Resume

To resume training from a specific checkpoint (e.g., episode 45000):

```bash
PRIMAL_RESUME_FROM=../checkpoints/a3c/primal_ep45000.pt python train_primal_pytorch_a3c.py
```

### Windows (PowerShell)

```powershell
$env:PRIMAL_RESUME_FROM="../checkpoints/a3c/primal_ep45000.pt"
python train_primal_pytorch_a3c.py
```

### Windows (CMD)

```cmd
set PRIMAL_RESUME_FROM=../checkpoints/a3c/primal_ep45000.pt
python train_primal_pytorch_a3c.py
```

### What Gets Loaded

When resuming from a checkpoint, the script will:

1. **Load model weights** - Restores the neural network parameters
2. **Load optimizer state** (if available) - Restores Adam optimizer momentum
3. **Extract episode number** - Either from checkpoint metadata or filename (e.g., "primal_ep45000.pt" → episode 45000)
4. **Continue training** - Training will resume from episode N+1 and continue to MAX_EPISODES

### Episode Number Detection

The script tries to determine the starting episode in this order:

1. **From checkpoint data**: If the checkpoint contains an `'episode'` key
2. **From filename**: Extracts number from patterns like `primal_ep45000.pt`
3. **Default to 0**: If neither works, starts from episode 0

### Checkpoint Compatibility

Works with checkpoints saved by:
- Current checkpoints (using `ACNet.save()` method)
- Contains `'model_state_dict'` key
- Optionally contains `'optimizer_state_dict'` and `'episode'` keys

### Example Workflow

```bash
# Start training from scratch
python train_primal_pytorch_a3c.py

# Training runs and saves checkpoints every 1000 episodes:
# checkpoints/primal_ep1000.pt
# checkpoints/primal_ep2000.pt
# ...
# checkpoints/primal_ep45000.pt

# If interrupted, resume from latest checkpoint
PRIMAL_RESUME_FROM=checkpoints/primal_ep45000.pt python train_primal_pytorch_a3c.py

# Training continues from episode 45001 → 100000
```

### Combined with Other Settings

You can combine `PRIMAL_RESUME_FROM` with other environment variables:

```bash
# Resume from checkpoint and train for 50000 more episodes
PRIMAL_RESUME_FROM=checkpoints/primal_ep45000.pt \
PRIMAL_MAX_EP=95000 \
python train_primal_pytorch_a3c.py

# Resume with different checkpoint directory
PRIMAL_RESUME_FROM=checkpoints/primal_ep45000.pt \
PRIMAL_CKPT_DIR=checkpoints_continued \
python train_primal_pytorch_a3c.py
```

### Notes on Resume Training

- The script automatically detects and prints the starting episode number
- Training will save new checkpoints to the directory specified by `PRIMAL_CKPT_DIR`
- Make sure the checkpoint path is correct and the file exists
- If the checkpoint format is incompatible, you'll see warning messages

## File Structure

### PyTorch Implementation (Recommended)

- **train_primal_pytorch_a3c.py**: PyTorch training script using A2C algorithm with imitation learning
- **ACNet_pytorch.py**: PyTorch neural network architecture (Actor-Critic with LSTM)
- **visualize_trained_model.py**: Visualization tool for trained models with support for:
  - Loading custom maps from CSV files
  - Setting custom start/end positions
  - Creating static path visualizations (PNG)
  - Creating animated GIFs
  - Interactive step-by-step mode
- **run_visualization.py**: Simple script to run visualization with predefined parameters
- **example_visualize.py**: Examples showing how to use visualization functions programmatically
- **primal_env_wrapper.py**: PyTorch-compatible environment wrapper
- **mapf_gym.py**: Multi-agent path planning gym environment
- **run_tests.py**: Model tests (gradients, GPU)
- **mapf_gym_unittests.py**: Environment unit tests
- **od_mstar3/**: Expert planner (M*) for imitation learning

### Legacy TensorFlow Implementation (Deprecated)

- **ACNet.py**: TensorFlow 1.x Actor-Critic network
- **primal_testing.py**: Code to run systematic validation tests of PRIMAL
- **mapf_gym_cap.py**: Multi-agent path planning gym environment with capped goal distance
- **mapgenerator.py**: Script for creating custom environments using Tkinter GUI
- **GroupLock.py**: Thread synchronization utility (unused)

### Before compilation: compile cpp_mstar code

- cd into the od_mstar3 folder.
- python3 setup.py build_ext (may need --inplace as extra argument).
- copy so object from build/lib.*/ at the root of the od_mstar3 folder.
- Check by going back to the root of the git folder, running python3 and "import cpp_mstar"

### Custom testing (Legacy)

Edit mapgenerator.py to the correct path for the model. By default, the model is loaded from the model_primal folder.

Hotkeys:
- o: obstacle mode
- a: agent mode
- g: goal mode, click an agent then click a free tile to place its goal
- c: clear agents
- r: reset
- up/down arrows: change size
- p: pause inference

## Requirements

### PyTorch Implementation (Recommended)

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

### Legacy TensorFlow Implementation

- Python 3.4
- Cython 0.28.4
- OpenAI Gym 0.9.4
- Tensorflow 1.3.1
- Numpy 1.13.3
- matplotlib
- imageio (for GIFs creation)
- tk
- networkx (if using od_mstar.py and not the C++ version)
