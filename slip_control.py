import time
import numpy as np
import pybullet as p
import pybullet_data as pd


def contact():
    tip_z_position = p.getLinkState(hopperID, tipLinkIndex)[0][2]
    if tip_z_position < (0.05 + 0.002):
        return True
    else:
        return False


def getVelocity():
    hopper_vel = np.array(p.getBaseVelocity(hopperID)[0])
    return hopper_vel


def getTargetLegDisplacement():
    vel = getVelocity()[0:2]
    vel_error = vel - targetVelocity
    vel_gain = 0.027  # µ
    neutral_point = (vel * stance_duration) / 2.0  # s0_n = [s0_nx, s0_ny]
    diff_disp = vel_gain * vel_error  # ds0 = [ds_nx, ds_ny]
    x_y_disp = neutral_point + diff_disp  # s0 + ds0
    z_disp = -np.sqrt(
        getLegLength() ** 2 - x_y_disp[0] ** 2 - x_y_disp[1] ** 2
    )  # z = -√(L^2 - x^2 - y^2)
    disp = np.append(
        x_y_disp, z_disp
    )  # disp contains the x, y, z displacement of the tip of the leg in the base frame
    # if np.isnan(disp[2]):
    #     print('legs too short')
    return disp


def getLegLength():
    length = 0.5 - p.getJointState(hopperID, pneumatic_joint_index)[0]
    return length


def transform_H_to_B(vec):
    """
    Transform a vector from the hopper frame to the base frame
    """
    HB_matrix_row_form = p.getMatrixFromQuaternion(
        p.getBasePositionAndOrientation(hopperID)[1]
    )
    HB_matrix = np.zeros((4, 4))
    HB_matrix[3, 3] = 1
    HB_matrix[0, 0:3] = HB_matrix_row_form[0:3]
    HB_matrix[1, 0:3] = HB_matrix_row_form[3:6]
    HB_matrix[2, 0:3] = HB_matrix_row_form[6:9]
    HB_matrix = np.matrix(HB_matrix)
    BH_matrix = np.linalg.inv(HB_matrix)
    return BH_matrix * vec


def getSystemEnergy(k) -> float:
    m = 12.761 + 0.46057 + 1.7676 + 0.51686
    # k = 6000
    tipLinkIndex = 2

    velocity = getVelocity()
    kinetic_energy = 0.5 * m * np.linalg.norm(velocity) ** 2

    height_base = p.getBasePositionAndOrientation(hopperID)[0][2]
    height_toe = p.getLinkState(hopperID, tipLinkIndex)[0][2]
    potential_energy = m * 9.81 * (height_base - height_toe - 0.2)

    spring_length = getLegLength()
    spring_potential_energy = 0.5 * k * (spring_length - 0.5) ** 2

    return kinetic_energy + potential_energy + spring_potential_energy


def calculate_energy_loss(E_TD, E_LO, delta_E_plus):
    delta_E_minus = E_TD + delta_E_plus - E_LO
    return delta_E_minus


def calculate_energy_compensation(delta_E_minus, E_LO, E_des):
    delta_E_plus_next = delta_E_minus + E_des - E_LO
    return delta_E_plus_next


# -----------Start Setup---------------
k_flight = 6000
k_stance = 6000 * 2.25
state = 0
legForce = 0
tipLinkIndex = 2

outer_hip_joint_index = 0
inner_hip_joint_index = 1
pneumatic_joint_index = 2

hip_joint_kp = 100
hip_joint_kd = 0.5

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

planeID = p.loadURDF("plane.urdf")
p.changeDynamics(
    planeID, -1, lateralFriction=60
)  # frction coefficient is set to 60 (unitless)
# p.resetDebugVisualizerCamera(cameraDistance=1.62, cameraYaw=47.6, cameraPitch=-30.8,
#                              cameraTargetPosition=[0.43, 1.49, -0.25])

# hopperID = p.loadURDF("./slip/urdf/slip.urdf", [0, 0, 1], [0.00, 0.0, 0, 1])
hopperID = p.loadURDF("./slip_flat/urdf/slip_flat.urdf", [0, 0, 1], [0.00, 0.001, 0, 1])
p.setJointMotorControl2(
    hopperID, pneumatic_joint_index, p.VELOCITY_CONTROL, force=0
)  # set the pneumatic joint to be in position control mode
p.setGravity(0, 0, -9.81)

curtime = 0
dt = 1.0 / 240.0

# num_joint = p.getNumJoints(hopperID)
# num_link = p.getNumJoints(hopperID)
# for i in range(num_joint):
#     print(p.getJointInfo(hopperID, i))
# print(p.getLinkState(hopperID, 0))

prev_orientation = np.array([0, 0, 0])
count = 0

stance_made = False
stance_duration = 0.17  # this value is determined by trial and error

targetVelocity = np.array([0.0, 0.0])
speed = 0.5
delta_E_plus = 0  # Initial compensatory energy
E_des = 100  # Example desired energy level (tune as necessary)
start_time = time.perf_counter()
while 1:
    keys = p.getKeyboardEvents()
    key_pressed = False
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        targetVelocity = np.array([0.0, 1.0]) * speed
        key_pressed = True
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        targetVelocity = np.array([0.0, -1.0]) * speed
        key_pressed = True
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        targetVelocity = np.array([-1.0, 0.0]) * speed
        key_pressed = True
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        targetVelocity = np.array([1.0, 0.0]) * speed
        key_pressed = True

    if not key_pressed:
        targetVelocity = np.array([0.0, 0.0])

    count = count + 1
    curtime = curtime + dt
    position = p.getJointState(hopperID, pneumatic_joint_index)[
        0
    ]  # get the position on axis z of the pneumatic joint

    if contact():
        state = 0
    else:
        state = 1

    if state == 1: # flight, state = 1
        stance_made = False
        legForce = -(k_flight) * position
        targetLegDisplacement_H = getTargetLegDisplacement()
        targetLegDisplacement_H = np.append(targetLegDisplacement_H, 1)
        targetLegDisplacement_H = np.matrix(targetLegDisplacement_H)
        targetLegDisplacement_H = targetLegDisplacement_H.T
        targetLegDisplacement_B = transform_H_to_B(
            targetLegDisplacement_H
        )  # get the target leg displacement in the base frame
        x_disp_B = targetLegDisplacement_B[0, 0]
        y_disp_B = targetLegDisplacement_B[1, 0]
        # d = getLegLength()
        d = 0.5  # getLegLength() ~ 0.5
        theta_inner = np.arcsin(x_disp_B / d)  # θtd_n
        theta_outer = -np.arcsin(y_disp_B / (d * np.cos(theta_inner)))  # ϕtd_n
        current_inner_hip_angle = p.getJointState(hopperID, inner_hip_joint_index)[0]
        current_outer_hip_angle = p.getJointState(hopperID, outer_hip_joint_index)[0]
        p.setJointMotorControl2(
            hopperID,
            outer_hip_joint_index,
            p.POSITION_CONTROL,
            targetPosition=theta_outer,
        )
        p.setJointMotorControl2(
            hopperID,
            inner_hip_joint_index,
            p.POSITION_CONTROL,
            targetPosition=theta_inner,
        )
        E_LO = getSystemEnergy(k_stance)
    else: # stance, state = 0
        if not stance_made:
            stance_made = True
            stance_duration = 0
        stance_duration = stance_duration + dt
        base_orientation = p.getBasePositionAndOrientation(hopperID)[1]
        base_orientation_euler = np.array(p.getEulerFromQuaternion(base_orientation))
        orientation_change = base_orientation_euler - prev_orientation
        orientation_velocity = orientation_change / dt
        orientation_acceleration = orientation_velocity / dt
        outer_hip_joint_target_torque = (
            -hip_joint_kp * (base_orientation_euler[0] - 0)
            - hip_joint_kd * orientation_velocity[0]
        )  # −k_γp (γ_n − γ_des) − k_γv * γ_dot_n, where γ_des = 0
        inner_hip_joint_target_torque = (
            -hip_joint_kp * (base_orientation_euler[1] - 0)
            - hip_joint_kd * orientation_velocity[1]
        )  # −k_αp (α_n − α_des) − k_αv * α_dot_n, where α_des = 0
        p.setJointMotorControl2(
            hopperID,
            outer_hip_joint_index,
            p.VELOCITY_CONTROL,
            targetVelocity=outer_hip_joint_target_torque,
        )
        p.setJointMotorControl2(
            hopperID,
            inner_hip_joint_index,
            p.VELOCITY_CONTROL,
            targetVelocity=inner_hip_joint_target_torque,
        )
        prev_orientation = base_orientation_euler
        E_TD = getSystemEnergy(k_stance)
        delta_E_minus = calculate_energy_loss(E_TD, E_LO, delta_E_plus)  # Calculate energy loss
        delta_E_plus = calculate_energy_compensation(delta_E_minus, E_LO, E_des)  # Calculate energy compensation for next step

        legForce = (-(k_stance) * position) - 500
        legForce2 = (2 * delta_E_plus * (0.5 - getLegLength())) / (0.5 - 0.1)**2
        print(delta_E_minus, delta_E_plus, legForce, legForce2)
        force_limit_pos = 5000
        # legForce = -max(min(legForce2, force_limit_pos), -force_limit_pos)
        # legForce = legForce2

    p.setJointMotorControl2(
        hopperID, pneumatic_joint_index, p.TORQUE_CONTROL, force=legForce
    )

    if count % 100 == 0:
        # print(np.round(getVelocity(), 3), np.round(getTargetLegDisplacement(), 3))
        print(p.getBasePositionAndOrientation(hopperID)[0])
        print(state, legForce)
        # print(
        #     np.round(current_inner_hip_angle - theta_inner, 3),
        #     np.round(current_outer_hip_angle - theta_outer, 3),
        # )

    p.stepSimulation()
    expected_time = start_time + count * dt
    actual_time = time.perf_counter()
    if expected_time > actual_time:
        time.sleep(expected_time - actual_time)
