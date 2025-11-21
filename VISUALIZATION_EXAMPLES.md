# PRIMAL Visualization Examples

## Quick Start

All commands should be run from the PRIMAL directory using the `primal` conda environment.

```bash
cd D:\astudy\NMvR\PRIMAL
```

## Basic Examples

### 1. Generate Static Path Image (PNG)

This creates a PNG image showing the complete paths of all agents from start to goal.

```bash
# Using primal conda environment
conda activate primal
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths my_paths.png
```

**Output:** `my_paths.png` - Shows agent trajectories with colored lines

### 2. Create Animated GIF

This generates an animated GIF showing agents moving step-by-step.

```bash
conda activate primal
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt gif demo.gif
```

**Output:** `demo.gif` - Animated visualization (may take a few minutes to generate)

### 3. Interactive Step-by-Step Mode

Control the visualization manually with keyboard commands.

```bash
conda activate primal
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt interactive
```

**Controls:**
- `[Enter]` - Execute one step
- `r` - Reset episode
- `q` - Quit
- `a` - Auto-run entire episode

### 4. Auto-Play with Live Display

Watch agents navigate in real-time without saving.

```bash
conda activate primal
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt show
```

**Output:** Live matplotlib window (no file saved)

## Windows-Specific Examples

### Using Full Paths (Recommended for Windows)

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate primal

# Change to PRIMAL directory
cd "D:\astudy\NMvR\PRIMAL"

# Generate paths visualization
python visualize_trained_model.py "D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt" paths output.png
```

### One-Liner Command

```bash
cd "D:\astudy\NMvR\PRIMAL" && source ~/miniconda3/etc/profile.d/conda.sh && conda activate primal && python visualize_trained_model.py "D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt" paths visualization_output.png
```

## Testing Different Checkpoints

```bash
# Test early training checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep1000.pt paths ep1000_paths.png

# Test mid training checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep5000.pt paths ep5000_paths.png

# Test final checkpoint
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths ep7000_paths.png
```

## Customization

The visualization script has hardcoded parameters in `main()` function:
- `num_agents=1` - Number of agents in the environment
- `world_size=20` - Size of the grid world
- `obstacle_density=0.2` - Probability of obstacles (0-0.5)

To change these, edit lines 401-405 in `visualize_trained_model.py`.

## Troubleshooting

### Error: "No such file or directory"
Make sure you're in the correct directory and using quotes around paths with spaces.

### Error: "No module named..."
Activate the primal conda environment first:
```bash
conda activate primal
```

### Visualization window doesn't appear
For WSL/remote systems, make sure X11 forwarding is set up, or use `paths` or `gif` mode instead.

## Output Files Location

All output files are saved in the current working directory (PRIMAL folder) unless you specify a full path:

```bash
# Save to specific location
python visualize_trained_model.py D:\astudy\NMvR\checkpoints\a2c\primal_ep7000.pt paths "D:\outputs\my_visualization.png"
```
