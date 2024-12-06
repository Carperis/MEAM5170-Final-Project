import os
from argparse import ArgumentParser

import mujoco_py
import numpy as np
import scipy.ndimage
from tqdm import trange

from ..config.linear_mpc_configs import LinearMpcConfig
from ..config.robot_configs import AliengoConfig
from ..linear_mpc.gait import Gait
from ..linear_mpc.leg_controller import LegController
from ..linear_mpc.mpc import ModelPredictiveController
from ..linear_mpc.swing_foot_trajectory_generator import SwingFootTrajectoryGenerator
from ..utils.robot_data import RobotData

STATE_ESTIMATION = False


def reset(sim, robot_config):
    sim.reset()
    # q_pos_init = np.array([
    #     0, 0, 0.116536,
    #     1, 0, 0, 0,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77,
    #     0, 1.16, -2.77
    # ])
    q_pos_init = np.array(
        [
            0,
            0,
            robot_config.base_height_des,
            1,
            0,
            0,
            0,
            0,
            0.8,
            -1.6,
            0,
            0.8,
            -1.6,
            0,
            0.8,
            -1.6,
            0,
            0.8,
            -1.6,
        ]
    )

    q_vel_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    init_state = mujoco_py.cymj.MjSimState(
        time=0.0, qpos=q_pos_init, qvel=q_vel_init, act=None, udd_state={}
    )
    sim.set_state(init_state)


def get_true_simulation_data(sim):
    pos_base = sim.data.body_xpos[1]
    vel_base = sim.data.body_xvelp[1]
    quat_base = sim.data.sensordata[0:4]
    omega_base = sim.data.sensordata[4:7]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]
    pos_foothold = [
        sim.data.get_geom_xpos("fl_foot"),
        sim.data.get_geom_xpos("fr_foot"),
        sim.data.get_geom_xpos("rl_foot"),
        sim.data.get_geom_xpos("rr_foot"),
    ]
    vel_foothold = [
        sim.data.get_geom_xvelp("fl_foot"),
        sim.data.get_geom_xvelp("fr_foot"),
        sim.data.get_geom_xvelp("rl_foot"),
        sim.data.get_geom_xvelp("rr_foot"),
    ]
    pos_thigh = [
        sim.data.get_body_xpos("FL_thigh"),
        sim.data.get_body_xpos("FR_thigh"),
        sim.data.get_body_xpos("RL_thigh"),
        sim.data.get_body_xpos("RR_thigh"),
    ]

    true_simulation_data = [
        pos_base,
        vel_base,
        quat_base,
        omega_base,
        pos_joint,
        vel_joint,
        touch_state,
        pos_foothold,
        vel_foothold,
        pos_thigh,
    ]
    # print(true_simulation_data)
    return true_simulation_data


def get_simulated_sensor_data(sim):
    imu_quat = sim.data.sensordata[0:4]
    imu_gyro = sim.data.sensordata[4:7]
    imu_accelerometer = sim.data.sensordata[7:10]
    pos_joint = sim.data.sensordata[10:22]
    vel_joint = sim.data.sensordata[22:34]
    touch_state = sim.data.sensordata[34:38]

    simulated_sensor_data = [
        imu_quat,
        imu_gyro,
        imu_accelerometer,
        pos_joint,
        vel_joint,
        touch_state,
    ]
    # print(simulated_sensor_data)
    return simulated_sensor_data


def initialize_robot(sim, viewer, robot_config, robot_data):
    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)
    init_gait = Gait.STANDING
    vel_base_des = [0.0, 0.0, 0.0]

    for iter_counter in range(800):

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5],
        )

        init_gait.set_iteration(
            predictive_controller.iterations_between_mpc, iter_counter
        )
        swing_states = init_gait.get_swing_state()
        gait_table = init_gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)
        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter,
            vel_base_des,
            0.0,
            gait_table,
            solver="drake",
            debug=False,
            iter_debug=0,
        )

        torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states)
        sim.data.ctrl[:] = torque_cmds

        sim.step()
        if viewer is not None:
            viewer.render()


def main():
    parser = ArgumentParser()
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--vel-base-des", type=float, default=0.0
    )  # Negative for random
    parser.add_argument(
        "--yaw-turn-rate-des", type=float, default=0.0
    )  # Negative for random
    parser.add_argument("--gait", type=str, default="TROTTING10")
    args = parser.parse_args()

    cur_path = os.path.dirname(__file__)
    mujoco_xml_path = os.path.join(cur_path, "../robot/aliengo/aliengo.xml")
    model = mujoco_py.load_model_from_path(mujoco_xml_path)
    sim = mujoco_py.MjSim(model)
    if args.no_viewer:
        viewer = None
    else:
        viewer = mujoco_py.MjViewer(sim)

    robot_config = AliengoConfig

    reset(sim, robot_config)
    sim.step()

    urdf_path = os.path.join(cur_path, "../robot/aliengo/urdf/aliengo.urdf")
    robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
    # initialize_robot(sim, viewer, robot_config, robot_data)

    predictive_controller = ModelPredictiveController(LinearMpcConfig, AliengoConfig)
    leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

    gait = getattr(Gait, args.gait)
    swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)]

    dt_range = np.arange(0, 10, step=predictive_controller.dt_control)

    vels_base_des = np.zeros((dt_range.shape[0], 3))
    if args.vel_base_des >= 0:
        vels_base_des[:, 0] = args.vel_base_des
    else:
        np.random.seed(0)
        noise = np.random.randn(dt_range.shape[0])
        noise = scipy.ndimage.gaussian_filter1d(noise, 500)
        noise -= noise.min()
        noise /= noise.max()
        noise *= 2.5  # max speed 2.5 m/s
        vels_base_des[:, 0] = noise
    lin_vel_base_actual = np.empty((dt_range.shape[0], 3))

    yaw_turn_rates_des = np.zeros(dt_range.shape[0])
    if args.vel_base_des >= 0:
        yaw_turn_rates_des[:] = args.yaw_turn_rate_des
    else:
        np.random.seed(42)
        noise = np.random.randn(dt_range.shape[0])
        noise -= noise.min()
        noise /= noise.max()
        noise -= 0.5
        noise *= 1.6  # -0.8 to 0.8 rad/s
        yaw_turn_rates_des[:] = noise
    ang_vel_base_actual = np.empty((dt_range.shape[0], 3))

    joints_qdot = np.empty((dt_range.shape[0], 12))
    joints_torque = np.empty((dt_range.shape[0], 12))

    touch_states = np.empty((dt_range.shape[0], 4), dtype=bool)

    iter_counter = 0

    for i_step in trange(dt_range.shape[0]):

        if not STATE_ESTIMATION:
            data = get_true_simulation_data(sim)
        else:
            data = get_simulated_sensor_data(sim)

        robot_data.update(
            pos_base=data[0],
            lin_vel_base=data[1],
            quat_base=data[2],
            ang_vel_base=data[3],
            q=data[4],
            qdot=data[5],
        )

        lin_vel_base_actual[i_step] = robot_data.lin_vel_base
        ang_vel_base_actual[i_step] = robot_data.ang_vel_base
        joints_qdot[i_step] = robot_data.qdot
        touch_states[i_step] = np.array(data[5 if STATE_ESTIMATION else 6]) > 0

        gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
        swing_states = gait.get_swing_state()
        gait_table = gait.get_gait_table()

        predictive_controller.update_robot_state(robot_data)

        contact_forces = predictive_controller.update_mpc_if_needed(
            iter_counter,
            vels_base_des[i_step],
            yaw_turn_rates_des[i_step],
            gait_table,
            solver="drake",
            debug=False,
            iter_debug=0,
        )
        # contact_forces_int = np.array(contact_forces, dtype=np.int8)
        # print(f"LF: {np.sum(contact_forces_int[0:3])}, RF: {np.sum(contact_forces_int[3:6])}, LH: {np.sum(contact_forces_int[6:9])}, RH: {np.sum(contact_forces_int[9:12])}")
        pos_targets_swingfeet = np.zeros((4, 3))
        vel_targets_swingfeet = np.zeros((4, 3))

        for leg_idx in range(4):
            if swing_states[leg_idx] > 0:  # leg is in swing state
                swing_foot_trajs[leg_idx].set_foot_placement(
                    robot_data, gait, vels_base_des[i_step], yaw_turn_rates_des[i_step]
                )
                base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = (
                    swing_foot_trajs[leg_idx].compute_traj_swingfoot(robot_data, gait)
                )
                pos_targets_swingfeet[leg_idx, :] = base_pos_base_swingfoot_des
                vel_targets_swingfeet[leg_idx, :] = base_vel_base_swingfoot_des

        torque_cmds = leg_controller.update(
            robot_data,
            contact_forces,
            swing_states,
            pos_targets_swingfeet,
            vel_targets_swingfeet,
        )
        # print(torque_cmds)
        sim.data.ctrl[:] = torque_cmds
        joints_torque[i_step] = torque_cmds

        sim.step()
        if viewer is not None:
            viewer.render()
        iter_counter += 1

    if args.save_npy:
        np.save("vels_base_des.npy", vels_base_des)
        np.save("yaw_turn_rates_des.npy", yaw_turn_rates_des)
        np.save("lin_vel_base_actual.npy", lin_vel_base_actual)
        np.save("ang_vel_base_actual.npy", ang_vel_base_actual)
        np.save("joints_qdot.npy", joints_qdot)
        np.save("joints_torque.npy", joints_torque)
        np.save("touch_states.npy", touch_states)


if __name__ == "__main__":
    main()
