from pathlib import Path
from typing import Any, Literal, Tuple, Dict

# import gym
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None
# from mujoco.glfw import glfw

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

# From CuRobo
# Standard Library
import time

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.2, -0.3], [0.6, 0.3]])


class PandaPickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
    ):
        self._action_scale = action_scale
        self.reward_type = reward_type

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)
        self.image_obs = image_obs
        
        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32
                        ),
                        "tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(6,), dtype=np.float32
                        ),
                        "gripper_pose": spaces.Box(
                            -1, 1, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {
                            "tcp_pose": spaces.Box(
                                -np.inf, np.inf, shape=(7,), dtype=np.float32
                            ),
                            "tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "gripper_pose": spaces.Box(
                                -1, 1, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=np.asarray([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            high=np.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            dtype=np.float32,
        )

        self._viewer = mujoco.Renderer(
            self.model,
            height=render_spec.height,
            width=render_spec.width
        )
        self._viewer.render()

    def reset(self, seed=None, x=0, y=0, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        # block_xyz = np.random.uniform(*_SAMPLING_BOUNDS)
        # get xy from x, y as porportional to sampling bounds
        sample = np.array([x, y])
        block_xyz = sample * (_SAMPLING_BOUNDS[1] - _SAMPLING_BOUNDS[0]) + _SAMPLING_BOUNDS[0]
        # self._block_z = np.random.uniform(0.1, 0.3)
        self._data.jnt("block").qpos[:3] = (*block_xyz, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        # change to absolute pose
        x, y, z, quat_w, quat_x, quat_y, quat_z, grasp = action

        # Set the mocap position.
        # pos = self._data.mocap_pos[0].copy()
        # dpos = np.asarray([x, y, z]) * self._action_scale[0]
        # npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        # self._data.mocap_pos[0] = npos
        self._data.mocap_pos[0] = np.clip(np.array([x, y, z]), *_CARTESIAN_BOUNDS)

        # quat = np.array([quat_x, quat_y, quat_z, quat_w])
        # self._data.mocap_quat[0] = quat

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()
        block_pos = self._data.sensor("block_pos").data
        outside_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05))
        terminated = self.time_limit_exceeded() or success or outside_bounds

        return obs, rew, terminated, False, {"succeed": success}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(
                self._viewer.render()
            )
        return rendered_frames


    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = np.asarray(self._data.sensor("2f85/pinch_pos").data).reshape(-1)
        tcp_quat = np.asarray(self._data.sensor("2f85/pinch_quat").data).reshape(-1)

        obs["state"]["tcp_pose"] = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)

        tcp_vel = np.asarray(self._data.sensor("2f85/pinch_vel").data).reshape(-1)
        tcp_angvel = np.asarray(self._data.sensor("2f85/pinch_angvel").data).reshape(-1)
        obs["state"]["tcp_vel"] = np.concatenate([tcp_vel, tcp_angvel]).astype(np.float32)
        obs["state"]["joint_pos"] = self._data.qpos[self._panda_dof_ids].astype(np.float32)

        gripper_pose = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["gripper_pose"] = gripper_pose

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = np.asarray(self._data.sensor("block_pos").data).reshape(-1).astype(np.float32)
            obs["state"]["block_pos"] = block_pos
            obs["state"]["block_quat"] = np.asarray(self._data.sensor("block_quat").data).reshape(-1).astype(np.float32)

        return obs

    def _compute_reward(self) -> float:
        if self.reward_type == "dense":
            block_pos = self._data.sensor("block_pos").data
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            rew = 0.3 * r_close + 0.7 * r_lift
            return rew
        else:
            block_pos = self._data.sensor("block_pos").data
            lift = block_pos[2] - self._z_init
            return float(lift > 0.2)

    def _is_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.2
    
    # def compute_ik(self, pos, quat):
    #     tensor_args = TensorDeviceType()

    #     config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    #     print(join_path(get_robot_configs_path(), "franka.yml"))
    #     urdf_file = config_file["robot_cfg"]["kinematics"][
    #         "urdf_path"
    #     ]  # Send global path starting with "/"
    #     base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    #     ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    #     robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    #     ik_config = IKSolverConfig.load_from_robot_config(
    #         robot_cfg,
    #         None,
    #         rotation_threshold=0.05,
    #         position_threshold=0.005,
    #         num_seeds=20,
    #         self_collision_check=False,
    #         self_collision_opt=False,
    #         tensor_args=tensor_args,
    #         use_cuda_graph=True,
    #     )
    #     ik_solver = IKSolver(ik_config)

    #     for _ in range(10):
    #         q_sample = ik_solver.sample_configs(100)
    #         kin_state = ik_solver.fk(q_sample)
    #         # print(kin_state)
    #         goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    #         st_time = time.time()
    #         result = ik_solver.solve_batch(goal)
    #         torch.cuda.synchronize()
    #         print(
    #             "Success, Solve Time(s), hz ",
    #             torch.count_nonzero(result.success).item() / len(q_sample),
    #             result.solve_time,
    #             q_sample.shape[0] / (time.time() - st_time),
    #             torch.mean(result.position_error),
    #             torch.mean(result.rotation_error),
    #         )

if __name__ == "__main__":
    env = PandaPickCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
