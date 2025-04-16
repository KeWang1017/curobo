# Intro:
This package provide a simple Franka arm and Robotiq Gripper simulator written in Mujoco.
It includes a state-based and a vision-based Franka lift cube task environment.

# Installation:
- From `hil-serl` folder, cd into `franka_sim`.
- In your `hil-serl` conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Explore the Environments
- Run `python franka_sim/test/test_gym_env_human.py` to launch a display window and visualize the task.
- Run `python franka_sim/test/test_gym_env_joystick.py --controller_type=<controller_type>` to launch a display window and use joystick to control the arm. `controller_type` can be `ps5` or `xbox`. You can use the keyboard to switch the camera view of each window by pressing `[` or  `]`.

# Troubleshooting
Sometimes the window will frozen and simulation will stop. Even if you killed the process, the program will still be running in the background. You can kill the process by running
  - `kill -9 $(pidof simulate)`.
  - ` lsof /dev/nvidia*` to find the process id of the program using the GPU and kill the relevant process.
  
# Credits:
- This simulation is initially built by [Kevin Zakka](https://kzakka.com/).
- Under Kevin's permission, we adopted a Gymnasium environment based on it.

# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
