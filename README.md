# MEAM5170_Final_Project

## pympc

The pip package does not list any dependency, so please install the dependencies manually.

To install the pip package locally:

```bash
pip install --editable /path/to/this/repository
```

To run the demo script:

```bash
pympc-mujoco-aliengo --vel-base-des 2.0 --yaw-turn-rate-des 0.0 --gait TROT
```

- Passing `--save-npy` causes saving NPY files to the workdir.
- Passing `--no-viewer` turns off the viewer UI.
- Passing a negative value to `--vel-base-des` causes a random trajectory of desired x speeds.
- Passing a negative value to `--yaw-turn-rate-des` causes a random trajectory of desired yaw turn rates.
- Valid gait values are `TROT`, `PACE`, and `WALK`.
