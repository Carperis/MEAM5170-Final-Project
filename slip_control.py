import time
import numpy as np
import pybullet as p
import pybullet_data as pd

def contact():
    tip_z_position = p.getLinkState(hopperID, tipLinkIndex)[0][2]
    if tip_z_position < 0.0504:
        return True
    else:
        return False

def getVelocity():
    hopper_vel = np.array(p.getBaseVelocity(hopperID)[0])
    return hopper_vel

def getTargetLegDisplacement():
    vel = getVelocity()[0:2]
    vel_error = vel - targetVelocity
    vel_gain = 0.027 # this value is determined by trial and error

    neutral_point = (vel * stance_duration) / 2.0
    if True and len(stance_velocities) != 0:
        neutral_point = (np.mean(stance_velocities, axis=0)[0:2] * stance_duration) / 2.0

    x_y_disp = neutral_point + vel_gain * vel_error
    z_disp = -np.sqrt(getLegLength() ** 2 - x_y_disp[0] ** 2 - x_y_disp[1] ** 2)

    if np.isnan(z_disp):
        print('legs too short')
        z_disp = 0
        x_y_disp /= np.linalg.norm(x_y_disp)

    disp = np.append(x_y_disp, z_disp) # disp contains the x, y, z displacement of the tip of the leg in the base frame
    return disp

def getLegLength():
    # 0.15 is the rest length of the pneumatic joint
    return 0.5 - p.getJointState(hopperID, pneumatic_joint_index)[0]

def transform_H_to_B(vec):
    HB_matrix_row_form = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(hopperID)[1])
    HB_matrix = np.zeros((4, 4))
    HB_matrix[3, 3] = 1
    HB_matrix[0, 0:3] = HB_matrix_row_form[0:3]
    HB_matrix[1, 0:3] = HB_matrix_row_form[3:6]
    HB_matrix[2, 0:3] = HB_matrix_row_form[6:9]
    HB_matrix = np.matrix(HB_matrix)
    BH_matrix = np.linalg.inv(HB_matrix)
    return BH_matrix * vec

# -----------Start Setup---------------
k_flight = 6000
k_stance = 6000
state = 0
legForce = 0
tipLinkIndex = 2

outer_hip_joint_index = 0
inner_hip_joint_index = 1
pneumatic_joint_index = 2

hip_joint_kp = 2
hip_joint_kd = 0.5

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

planeID = p.loadURDF("plane.urdf")
p.changeDynamics(planeID, -1, lateralFriction=60) # frction coefficient is set to 60 (unitless)
# p.resetDebugVisualizerCamera(cameraDistance=1.62, cameraYaw=47.6, cameraPitch=-30.8,
#                              cameraTargetPosition=[0.43, 1.49, -0.25])

# hopperID = p.loadURDF("./slip/urdf/slip.urdf", [0, 0, 1], [0.00, 0.001, 0, 1])
hopperID = p.loadURDF("./slip_flat/urdf/slip_flat.urdf", [0, 0, 1], [0.00, 0.001, 0, 1])
p.setJointMotorControl2(hopperID, pneumatic_joint_index, p.VELOCITY_CONTROL, force=0) # set the pneumatic joint to be in position control mode
p.setGravity(0, 0, -9.81)

num_joint = p.getNumJoints(hopperID)
for i in range(num_joint):
    print(p.getJointInfo(hopperID, i))

curtime = 0
dt = 1. / 240.

# while 1:
#     position = p.getJointState(hopperID, pneumatic_joint_index)[0]
#     legForce = -(k_flight) * position
#     p.setJointMotorControl2(
#         hopperID, pneumatic_joint_index, p.TORQUE_CONTROL, force=legForce
#     )
#     print(p.getJointState(hopperID, pneumatic_joint_index))
#     time.sleep(dt)
#     p.stepSimulation()

prev_orientation = np.array([0, 0, 0])
count = 0

stance_made = False
stance_duration = 0.17 # this value is determined by trial and error
stance_velocities = []

targetVelocity = np.array([0.3, 0.3])

def getSystemEnergy() -> float:
    m = 12.761

    velocity = getVelocity()
    kinetic_energy = 0.5 * m * np.linalg.norm(velocity) ** 2

    height = p.getLinkState(hopperID, 0)[0][2]  # TODO
    potential_energy = m * 9.81 * height

    spring_length = getLegLength()
    spring_potential_energy = 0.5 * k_stance * (spring_length - 0.5) ** 2

    return kinetic_energy + potential_energy + spring_potential_energy

original_energy = getSystemEnergy()
original_energy *= 1
previous_energy_loss = None

start_time = time.perf_counter()
step_count = 0

while 1:
    step_count += 1

    keys = p.getKeyboardEvents()
    key_pressed = False
    # External force in 4 directions when shift + arrow key is pressed

    if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN:
        print("Shift key is pressed")
        force_magnitude = 5
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            p.applyExternalForce(hopperID, 0, [0, force_magnitude, 0], [0, 0, 0], p.WORLD_FRAME)
            key_pressed = True
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            p.applyExternalForce(hopperID, 0, [0, -force_magnitude, 0], [0, 0, 0], p.WORLD_FRAME)
            key_pressed = True
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            p.applyExternalForce(hopperID, 0, [-force_magnitude, 0, 0], [0, 0, 0], p.WORLD_FRAME)
            key_pressed = True
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            p.applyExternalForce(hopperID, 0, [force_magnitude, 0, 0], [0, 0, 0], p.WORLD_FRAME)
            key_pressed = True
    else:
        speed = 0.4
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            targetVelocity = np.array([0.0, 1]) * speed
            key_pressed = True
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            targetVelocity = np.array([0.0, -1]) * speed
            key_pressed = True
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            targetVelocity = np.array([-1, 0.0]) * speed
            key_pressed = True
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            targetVelocity = np.array([1, 0.0]) * speed
            key_pressed = True

    if not key_pressed:
        targetVelocity = np.array([0.0, 0.0])

    count = count + 1
    curtime = curtime + dt
    position = p.getJointState(hopperID, pneumatic_joint_index)[0] # get the position on axis z of the pneumatic joint

    old_state = state
    if contact():
        state = 0
    else:
        state = 1

    if state == 1:
        if old_state == 0:
            liftoff_energy = getSystemEnergy()
            previous_energy_loss = original_energy - liftoff_energy
            print("Energy loss:", original_energy - liftoff_energy)
        # Flight phase

        stance_made = False
        legForce = -(k_flight) * position
        targetLegDisplacement_H = getTargetLegDisplacement()
        targetLegDisplacement_H = np.append(targetLegDisplacement_H, 1)
        targetLegDisplacement_H = np.matrix(targetLegDisplacement_H)
        targetLegDisplacement_H = targetLegDisplacement_H.T
        targetLegDisplacement_B = transform_H_to_B(targetLegDisplacement_H)
        x_disp_B = targetLegDisplacement_B[0, 0]
        y_disp_B = targetLegDisplacement_B[1, 0]
        d = getLegLength()
        d = 0.5
        theta_inner = np.arcsin(x_disp_B / d)
        theta_outer = np.arcsin(y_disp_B / (-d * np.cos(theta_inner)))
        p.setJointMotorControl2(hopperID, outer_hip_joint_index, p.POSITION_CONTROL,
                                targetPosition=theta_outer)
        p.setJointMotorControl2(hopperID, inner_hip_joint_index, p.POSITION_CONTROL,
                                targetPosition=theta_inner)
    else:
        # Stance phase
        if old_state == 1:
            touchdown_energy = getSystemEnergy()
            energy_loss = original_energy - touchdown_energy + previous_energy_loss

        if not stance_made:
            stance_made = True
            stance_duration = 0
            stance_velocities = []
        stance_duration = stance_duration + dt
        stance_velocities.append(getVelocity())

        base_orientation = p.getBasePositionAndOrientation(hopperID)[1]
        # base_orientation = p.getLinkState(hopperID, 0)[1]
        base_orientation_euler = np.array(p.getEulerFromQuaternion(base_orientation))
        orientation_change = base_orientation_euler - prev_orientation
        orientation_velocity = orientation_change / dt
        outer_hip_joint_target_vel = -hip_joint_kp * base_orientation_euler[0] - hip_joint_kd * orientation_velocity[0]
        inner_hip_joint_target_vel = -hip_joint_kp * base_orientation_euler[1] - hip_joint_kd * orientation_velocity[1]
        p.setJointMotorControl2(hopperID, outer_hip_joint_index, p.VELOCITY_CONTROL,
                                targetVelocity=outer_hip_joint_target_vel)
        p.setJointMotorControl2(hopperID, inner_hip_joint_index, p.VELOCITY_CONTROL,
                                targetVelocity=inner_hip_joint_target_vel)
        prev_orientation = base_orientation_euler
        # energy_loss = original_energy - getSystemEnergy()
        compensation_force = 2 * energy_loss * (0.5 - getLegLength()) * 500
        # compensation_force = 250
        # print(compensation_force)

        legForce = (-(k_stance) * position) - max(compensation_force, 0)

    p.setJointMotorControl2(hopperID, pneumatic_joint_index, p.TORQUE_CONTROL, force=legForce)

    # print(position, legForce)
    # if count % 100 == 0:
    #     print(getVelocity(), getTargetLegDisplacement())

    p.stepSimulation()

    expected_time = start_time + step_count * dt
    actual_time = time.perf_counter()
    if expected_time > actual_time:
        time.sleep(expected_time - actual_time)
