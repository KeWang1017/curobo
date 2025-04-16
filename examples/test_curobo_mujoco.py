import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs

from franka_sim.utils.viewer_utils import MujocoViewer
# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
DT = 0.02
env = envs.PandaPickCubeGymEnv(render_mode="human", action_scale=(0.1, 1))
action_spec = env.action_space
# Standard Library
tensor_args = TensorDeviceType()
world_file = "collision_table.yml"
robot_file = "franka.yml"
config_file = load_yaml(join_path(get_robot_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
kin_model = CudaRobotModel(robot_cfg.kinematics)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    None,
    tensor_args,
    interpolation_dt=DT,
    # trajopt_dt=0.15,
    # velocity_scale=0.1,
    use_cuda_graph=True,
    # finetune_dt_scale=2.5,
    interpolation_steps=10000,
)

motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

def demo_motion_gen(js=False, start_joint_state=None, goal_pose=None):
    goal_pose = Pose.from_list(goal_pose)  # x, y, z, qw, qx, qy, qz
    start_state = JointState.from_position(
        torch.tensor(start_joint_state).view(1, -1)
    )
    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js, parallel_finetune=True)
    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    # robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    # retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1))
    goal_state = goal_pose

    # start_state.position[0, 0] += 0.25
    goal_state.position[0,2] += 0.1
    if js:
        result = motion_gen.plan_single_js(
            start_state,
            goal_pose,
            MotionGenPlanConfig(max_attempts=1, time_dilation_factor=1.0),
        )
    else:
        result = motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                max_attempts=1,
                timeout=5,
                time_dilation_factor=1.0,
            ),
        )
    # new_result = result.clone()
    # new_result.retime_trajectory(0.5, create_interpolation_buffer=True)
    # print(new_result.optimized_dt, new_result.motion_time, result.motion_time)
    # print(
    #     "Trajectory Generated: ",
    #     result.success,
    #     result.solve_time,
    #     result.status,
    #     result.optimized_dt,
    # )
    traj = result.get_interpolated_plan()
    q = torch.tensor(traj.position, **(tensor_args.as_torch_dict()))
    out = kin_model.get_state(q)

    return out

# def sample():
#     a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
#     return a.astype(action_spec.dtype)


m = env.model
d = env.data

key_reset = False
KEY_SPACE = 32


def key_callback(keycode):
    if keycode == KEY_SPACE:
        global key_reset
        key_reset = True


obs, _ = env.reset()
block_pos = obs["state"]["block_pos"]
block_quat = obs["state"]["block_quat"]
goal_pose = np.concatenate([block_pos, block_quat])
init_joint_state = obs["state"]["joint_pos"]

traj = demo_motion_gen(js=False, start_joint_state=init_joint_state, goal_pose=goal_pose)
pos = traj.ee_position.squeeze().cpu().numpy()
quat = traj.ee_quaternion.squeeze().cpu().numpy()
gripper = np.zeros((len(quat), 1))
actions = np.concatenate([pos, quat, gripper], axis=1)
# breakpoint()
# Create the viewer
viewer = MujocoViewer(env.unwrapped.model, env.unwrapped.data)
with viewer as viewer:
    start = time.time()
    while viewer.is_running():
        for i, cmd in enumerate(actions):
            step_start = time.time()
            obs, rew, terminated, truncated, info = env.step(cmd)
            # print(obs)
            viewer.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            if i >= len(actions) - 1:
                obs, _ = env.reset()
                block_pos = obs["state"]["block_pos"]
                block_quat = obs["state"]["block_quat"]
                goal_pose = np.concatenate([block_pos, block_quat])
                init_joint_state = obs["state"]["joint_pos"]
                start_time = time.perf_counter()
                traj = demo_motion_gen(js=False, start_joint_state=init_joint_state, goal_pose=goal_pose)
                print("Time to generate trajectory: ", time.perf_counter() - start_time)
                pos = traj.ee_position.squeeze().cpu().numpy()
                quat = traj.ee_quaternion.squeeze().cpu().numpy()
                gripper = np.zeros((len(quat), 1))
                actions = np.concatenate([pos, quat, gripper], axis=1)


