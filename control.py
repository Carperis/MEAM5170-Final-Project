import numpy as np
import pybullet as p
from contact import contact
from velocity import getVelocity
from leg_displacement import getTargetLegDisplacement, getLegLength
from transform import transform_H_to_B

def control_hopper(hopperID, state, stance_made, stance_duration, targetVelocity, prev_orientation, k_flight, k_stance, hip_joint_kp, hip_joint_kd):
    tipLinkIndex = 6
    outer_hip_joint_index = 2
    inner_hip_joint_index = 5
    pneumatic_joint_index = 6
    dt = 1. / 240.

    position = p.getJointState(hopperID, pneumatic_joint_index)[0]

    if contact(hopperID, tipLinkIndex):
        state = 0
    else:
        state = 1

    if state == 1:  # Flight phase
        stance_made = False
        legForce = -(k_flight) * position
        targetLegDisplacement_H = getTargetLegDisplacement(hopperID, targetVelocity, stance_duration)
        targetLegDisplacement_H = np.append(targetLegDisplacement_H, 1)
        targetLegDisplacement_H = np.matrix(targetLegDisplacement_H).T
        targetLegDisplacement_B = transform_H_to_B(hopperID, targetLegDisplacement_H)
        x_disp_B = targetLegDisplacement_B[0, 0]
        y_disp_B = targetLegDisplacement_B[1, 0]
        d = getLegLength(hopperID)
        theta_inner = np.arcsin(x_disp_B / d)
        theta_outer = np.arcsin(y_disp_B / (-d * np.cos(theta_inner)))
        p.setJointMotorControl2(hopperID, outer_hip_joint_index, p.POSITION_CONTROL, targetPosition=theta_outer)
        p.setJointMotorControl2(hopperID, inner_hip_joint_index, p.POSITION_CONTROL, targetPosition=theta_inner)
    else:  # Stance phase
        if not stance_made:
            stance_made = True
            stance_duration = 0
        stance_duration += dt
        base_orientation = p.getLinkState(hopperID, 1)[1]
        base_orientation_euler = np.array(p.getEulerFromQuaternion(base_orientation))
        orientation_change = base_orientation_euler - prev_orientation
        orientation_velocity = orientation_change / dt
        outer_hip_joint_target_vel = -hip_joint_kp * base_orientation_euler[0] - hip_joint_kd * orientation_velocity[0]
        inner_hip_joint_target_vel = -hip_joint_kp * base_orientation_euler[1] - hip_joint_kd * orientation_velocity[1]
        p.setJointMotorControl2(hopperID, outer_hip_joint_index, p.VELOCITY_CONTROL, targetVelocity=outer_hip_joint_target_vel)
        p.setJointMotorControl2(hopperID, inner_hip_joint_index, p.VELOCITY_CONTROL, targetVelocity=inner_hip_joint_target_vel)
        prev_orientation = base_orientation_euler
        legForce = (-(k_stance) * position) - 20

    p.setJointMotorControl2(hopperID, pneumatic_joint_index, p.TORQUE_CONTROL, force=legForce)
    return state, stance_made, stance_duration, prev_orientation
