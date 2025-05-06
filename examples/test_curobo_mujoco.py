import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs

from franka_sim.utils.viewer_utils import MujocoViewer
# Third Party
import torch
import os
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def plot_traj(trajectory, dt, file_name_1="test.png", file_name_2="test.png"):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(3, 1)
    q = trajectory.position.cpu().numpy()
    qd = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = trajectory.jerk.cpu().numpy()
    timesteps = [i * dt for i in range(q.shape[0])]
    joint_limits = robot_cfg.kinematics.get_joint_limits()
    position_limits = joint_limits.position.cpu().numpy()
    velocity_limits = joint_limits.velocity.cpu().numpy()
    acceleration_limits = joint_limits.acceleration.cpu().numpy()
    jerk_limits = joint_limits.jerk.cpu().numpy()
    for i in range(q.shape[-1]):
        # axs[0].plot(timesteps, q[:, i], label=str(i))
        # # add two lines to plot the joint limits with different color
        # axs[0].plot([0, q.shape[0] * dt], [position_limits[0, i], position_limits[0, i]], "k--", color="red")
        # axs[0].plot([0, q.shape[0] * dt], [position_limits[1, i], position_limits[1, i]], "k--", color="blue")
        axs[0].plot(timesteps, qd[:, i], label=str(i))
        axs[0].plot([0, q.shape[0] * dt], [velocity_limits[0, i], velocity_limits[0, i]], "k--")
        axs[0].plot([0, q.shape[0] * dt], [velocity_limits[1, i], velocity_limits[1, i]], "k--")
        axs[1].plot(timesteps, qdd[:, i], label=str(i))
        axs[1].plot([0, q.shape[0] * dt], [acceleration_limits[0, i], acceleration_limits[0, i]], "k--")
        axs[1].plot([0, q.shape[0] * dt], [acceleration_limits[1, i], acceleration_limits[1, i]], "k--")
        axs[2].plot(timesteps, qddd[:, i], label=str(i))
        axs[2].plot([0, q.shape[0] * dt], [jerk_limits[0, i], jerk_limits[0, i]], "k--")
        axs[2].plot([0, q.shape[0] * dt], [jerk_limits[1, i], jerk_limits[1, i]], "k--")

    plt.legend()

    plt.savefig(os.path.join("plots", file_name_2))
    plt.close()

    _, axs = plt.subplots(q.shape[-1], 1)
    for i in range(q.shape[-1]):
        axs[i].plot(timesteps, q[:, i], label=str(i))
        axs[i].plot([0, q.shape[0] * dt], [position_limits[0, i], position_limits[0, i]], "k--", color="red")
        axs[i].plot([0, q.shape[0] * dt], [position_limits[1, i], position_limits[1, i]], "k--", color="blue")
    plt.legend()
    plt.savefig(os.path.join("plots", file_name_1))
    plt.close()

DT = 0.02
env = envs.PandaPickCubeGymEnv(render_mode="human", action_scale=(0.1, 1))
action_spec = env.action_space
# Standard Library
tensor_args = TensorDeviceType()
world_file = "collision_table.yml"
robot_file = "franka.yml"
config_file = load_yaml(join_path(get_robot_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
# creat a folder to save the plots
os.makedirs("plots", exist_ok=True)
kin_model = CudaRobotModel(robot_cfg.kinematics)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    None,
    tensor_args,
    interpolation_dt=DT,
    velocity_scale=1.0, # used when generating slow trajectories
    # used for non-zero start velocity and acceleration
    trajopt_tsteps = 36,
    trajopt_dt = 0.05,
    optimize_dt = False,
    # max_attemtps = 1,
    trim_steps = [1, None]
    # trajopt_dt=0.15,
    # velocity_scale=0.1,
    # collision_checker_type=CollisionCheckerType.PRIMITIVE,
    # use_cuda_graph=True,
    # num_trajopt_seeds=12,
    # num_graph_seeds=1,
    # num_ik_seeds=30,
)

motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup(parallel_finetune=True)

def demo_motion_gen_multi_segment(start_joint_state=None, goal_pose=None):
    start_state = JointState.from_position(
        torch.tensor(start_joint_state).view(1, -1).cuda()
    )
    # make the goal state a bit higher than cube
    goal_pose[2] += 0.1
    start_pose = kin_model.get_state(torch.tensor(start_joint_state).view(1, -1).cuda())
    # breakpoint()
    intermediate_pose_1 =  np.concatenate([(start_pose.ee_position.squeeze().cpu().numpy() + goal_pose[:3]) / 2, goal_pose[3:]])# x, y, z, qw, qx, qy, qz
    from copy import deepcopy
    intermediate_pose_2 = deepcopy(goal_pose)
    intermediate_pose_2[2] += 0.2

    pose_list = [intermediate_pose_1, intermediate_pose_2,  goal_pose] # 
    trajectory = start_state
    for i, pose in enumerate(pose_list):
        goal_pose = Pose.from_list(pose)  # x, y, z, qw, qx, qy, qz
        start_state = trajectory[-1].unsqueeze(0).clone()
        # start_state.velocity[:] = 0.0
        # start_state.acceleration[:] = 0.0
        result = motion_gen.plan_single(
            start_state.clone(),
            goal_pose,
            plan_config=MotionGenPlanConfig(parallel_finetune=True, max_attempts=1, time_dilation_factor=1.0),
        )
        if result.success.item():
            plan = result.get_interpolated_plan()
            trajectory = trajectory.stack(plan.clone())
            # motion_time += result.motion_time
        else:
            print(i, "fail", result.status)
    
    # breakpoint()
    # goal_pose = Pose.from_list(goal_pose)  # x, y, z, qw, qx, qy, qz
    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js, parallel_finetune=True)
    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    # robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    # breakpoint()
    # goal_state = goal_pose
    # result = motion_gen.plan_single(
    #     start_state,
    #     goal_state,
    #     MotionGenPlanConfig(
    #             max_attempts=1,
    #             timeout=5,
    #             time_dilation_factor=1.0,
    #         ),
    #     )
    # retime the trajectory
    # new_result = result.clone()
    # new_result.retime_trajectory(0.8, create_interpolation_buffer=True)
    # print(new_result.optimized_dt, new_result.motion_time, result.motion_time)
    # print(
    #     "Trajectory Generated: ",
    #     result.success,
    #     result.solve_time,
    #     result.status,
    #     result.optimized_dt,
    # )
    # traj = result.get_interpolated_plan()
    # breakpoint()
    print(trajectory.position.shape[0])
    q = torch.tensor(trajectory.position, **(tensor_args.as_torch_dict()))
    task_space_traj = kin_model.get_state(q)

    return task_space_traj, trajectory 

def demo_motion_gen_single_segment(start_joint_state=None, goal_pose=None):
    start_state = JointState.from_position(
        torch.tensor(start_joint_state).view(1, -1).cuda()
    )
    # make the goal state a bit higher than cube
    goal_pose[2] += 0.1
    goal_pose = Pose.from_list(goal_pose)
    start_pose = kin_model.get_state(torch.tensor(start_joint_state).view(1, -1).cuda())
    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js, parallel_finetune=True)
    # goal_state = goal_pose

    # start_state.position[0, 0] += 0.25
    result = motion_gen.plan_single(
        start_state,
        goal_pose,
        MotionGenPlanConfig(
            max_attempts=1,
            timeout=5,
            time_dilation_factor=1.0,
        ),
    )
    traj = result.get_interpolated_plan()
    q = torch.tensor(traj.position, **(tensor_args.as_torch_dict()))
    task_space_traj = kin_model.get_state(q)

    return task_space_traj, traj

def motion_gen_block(x, y):
    obs, _ = env.reset(x=x, y=y)
    block_pos = obs["state"]["block_pos"]
    block_quat = obs["state"]["block_quat"]
    goal_pose = np.concatenate([block_pos, block_quat])
    init_joint_state = obs["state"]["joint_pos"]
    start_time = time.perf_counter()
    task_space_traj, traj = demo_motion_gen_multi_segment(start_joint_state=init_joint_state, goal_pose=goal_pose)
    print("Time to generate trajectory: ", time.perf_counter() - start_time)
    file_name_1 = f"traj_{x}_{y}_pos.png"
    file_name_2 = f"traj_{x}_{y}_vel.png"
    plot_traj(traj, DT, file_name_1, file_name_2)
    time_spend.append(time.perf_counter() - start_time)
    pos = task_space_traj.ee_position.squeeze().cpu().numpy()
    quat = task_space_traj.ee_quaternion.squeeze().cpu().numpy()
    gripper = np.zeros((len(quat), 1))
    actions = np.concatenate([pos, quat, gripper], axis=1)
    return actions

time_spend = []
error_pos = []
actions = motion_gen_block(0, 0)
# Create the viewer
viewer = MujocoViewer(env.unwrapped.model, env.unwrapped.data)
with viewer as viewer:
    start = time.time()
    while viewer.is_running():
        for x in [0.25 * i for i in range(0, 5)]:
            for y in [0.25 * i for i in range(0, 5)]:
                for i, cmd in enumerate(actions):
                    step_start = time.time()
                    obs, rew, terminated, truncated, info = env.step(cmd)
                    # print(obs)
                    viewer.sync()
                    time_until_next_step = env.control_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                if i >= len(actions) - 1:
                    block_pos = obs["state"]["block_pos"]
                    block_pos[2] += 0.1
                    ee_pos = obs["state"]["tcp_pose"][:3]
                    error = np.linalg.norm(block_pos - ee_pos)
                    print("error_pos: ", error)
                    error_pos.append(error)
                    actions = motion_gen_block(x, y)
                    if x == 1.0 and y == 1.0:
                        print("time_spend: ", np.mean(time_spend), np.std(time_spend))
                        print("error_pos: ", np.mean(error_pos), np.std(error_pos))
                        exit()

