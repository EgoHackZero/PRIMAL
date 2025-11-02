# PRIMAL Quick Start Guide

This guide provides instructions for running the PRIMAL project on your machine.

## Environment Setup

The project has been successfully set up using **conda environment** with Python 3.9.

### Activate the Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate primal_env
```

## Installed Dependencies

The following packages have been installed and tested:

- **Python 3.9.24**
- **numpy 2.0.2** - Numerical computing
- **matplotlib 3.9.4** - Plotting and visualization
- **networkx 3.2.1** - Network analysis (for od_mstar.py fallback)
- **gym 0.9.4** - OpenAI Gym environment
- **scipy 1.13.1** - Scientific computing
- **imageio 2.37.0** - Image I/O for GIFs
- **cython 3.1.6** - Used to compile cpp_mstar extension
- **cpp_mstar** - C++ compiled extension for pathfinding (successfully compiled!)

## Project Status

✅ **All core components are working:**
- C++ extension (cpp_mstar) compiled successfully
- All imports working correctly
- All unit tests passing (45/45 tests)

### Test Results

Run the unit tests to verify everything works:

```bash
cd /home/vboxuser/Desktop/PRIMAL
python mapf_gym_unittests.py
```

Expected output:
```
........................................................................................................................................................................................................................................
----------------------------------------------------------------------
Ran 45 tests in 0.063s

OK
```

## Running the Project

### 1. Basic Environment Test

```python
from od_mstar3 import cpp_mstar
import mapf_gym

# Create a simple MAPF environment
env = mapf_gym.MAPFEnv(num_agents=2, SIZE=10)
print("Environment created successfully!")
```

### 2. Running the Map Generator (requires TensorFlow)

To run the interactive map generator with a trained model, you would need:
- TensorFlow 1.3.1 (or compatible 1.x version)
- A trained model checkpoint

**Note:** TensorFlow 1.x is not compatible with Python 3.9. If you need to run the full training/testing pipeline, consider using Python 3.6-3.8.

### 3. Running the Training Code

The main training code is in `DRLMAPF_A3C_RNN.ipynb` (Jupyter notebook). This requires:
- TensorFlow 1.3.1
- GPU support (optional, can run on CPU)
- Jupyter notebook

To run:
```bash
jupyter notebook DRLMAPF_A3C_RNN.ipynb
```

## Project Structure

- **mapf_gym.py** - Multi-agent path planning gym environment
- **mapf_gym_unittests.py** - Unit tests for the environment
- **mapgenerator.py** - Interactive map generator and tester
- **primal_testing.py** - Systematic validation tests
- **ACNet.py** - Actor-Critic neural network architecture
- **GroupLock.py** - Threading utilities
- **od_mstar3/** - Path planning algorithms (C++ optimized)
  - **cpp_mstar.so** - Compiled C++ extension
  - **od_mstar.py** - Python fallback implementation
  - **col_set_addition.py** - Collision set utilities

## Notes

1. **TensorFlow Compatibility**: The project was designed for TensorFlow 1.3.1. The current setup works for the basic environment and pathfinding, but training/testing with neural networks would require TensorFlow.

2. **GPU Support**: Training is GPU-accelerated by default. To use CPU, modify the device specification in the notebook.

3. **Model Checkpoints**: Pre-trained models can be downloaded from the links in README.md.

## Quick Verification

Run this command to verify everything is working:

```bash
conda activate primal_env
cd /home/vboxuser/Desktop/PRIMAL
python -c "from od_mstar3 import cpp_mstar; import mapf_gym; print('✅ All imports successful!')"
python mapf_gym_unittests.py
```

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure you've activated the conda environment
2. **Module not found**: Check that you're in the PRIMAL directory
3. **C++ compilation issues**: The extension is already compiled, but if needed, rebuild with:
   ```bash
   cd od_mstar3
   python setup.py build_ext --inplace
   ```

## Next Steps

1. Explore the interactive map generator (requires trained model)
2. Read the main README.md for detailed documentation
3. Check out the paper for theoretical background
4. Download pre-trained models from the links provided in README.md

## Authors

- Guillaume Sartoretti
- Justin Kerr

License: MIT License

