# PRIMAL System Setup & Compatibility Notes

This project now targets Python 3.11 and TensorFlow 2.15 (running in `tf.compat.v1`
mode). Follow the steps below to reproduce the environment we used while
updating and validating the code base.

## 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate primal
```

The spec installs the core dependencies, including:

- Python 3.11
- TensorFlow 2.15 (via pip) configured for compatibility mode
- Gym 0.26, NumPy 1.26, SciPy, matplotlib, imageio, networkx, tk
- Cython (needed for the native M* planner)

## 2. Toolchain for the C++ M* planner

If you want the high‑performance `cpp_mstar` extension, install a compiler and
Boost headers inside the environment before building:

```bash
conda install -n primal gxx_linux-64 boost-cpp
```

You can use system packages instead (`build-essential` + `libboost-all-dev`) if
conda is not an option, but keeping everything inside the environment avoids
global configuration changes.

Then build the module in place:

```bash
cd /home/vboxuser/Desktop/PRIMAL/od_mstar3
conda run -n primal python setup.py build_ext --inplace
```

This produces `cpp_mstar.cpython-*.so`. If it’s missing, the project falls back
to the pure-Python `od_mstar` implementation (functional but slower).

## 3. Additional Python packages

The training notebook (`DRLMAPF_A3C_RNN.ipynb`) imports `scipy.signal`; this is
installed automatically via `environment.yml`. If you built the environment
manually, add SciPy:

```bash
conda install -n primal scipy
```

## 4. TensorFlow compatibility changes

The code disables eager execution and uses `tf.compat.v1.*` APIs. Key points:

- GPU is optional. The notebook now logs a warning instead of asserting that a
  GPU is present.
- Legacy calls such as `tf.placeholder`, `tf.get_collection`, etc., must be
  accessed through `tf.compat.v1`.
- The `ACNet` model has been rewritten to use modern `tf.keras.layers.*`, so old
  checkpoints are incompatible. Retrain if you need updated weights; loader
  paths now raise a clear error when a checkpoint doesn’t match the graph.

## 5. Validation

After setting up the environment, run the unit tests:

```bash
conda run -n primal python -m unittest mapf_gym_unittests.py
```

Expect to see Gym’s deprecation notice; all tests should pass.

---

These instructions capture all troubleshooting steps from our recent setup:
missing SciPy, absent GPU, Boost headers, and the compiler requirements for the
native planner. Follow them sequentially to reach a fully working installation.***
