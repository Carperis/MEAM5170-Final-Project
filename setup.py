import pybullet as p
import pybullet_data as pd

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    planeID = p.loadURDF("plane.urdf")
    p.changeDynamics(planeID, -1, lateralFriction=60)
    p.resetDebugVisualizerCamera(cameraDistance=1.62, cameraYaw=47.6, cameraPitch=-30.8,
                                 cameraTargetPosition=[0.43, 1.49, -0.25])
    hopperID = p.loadURDF("hopper.urdf", [0, 0, 0.2], [0.00, 0.001, 0, 1])
    p.setJointMotorControl2(hopperID, 6, p.VELOCITY_CONTROL, force=0)
    p.setGravity(0, 0, -9.81)
    return hopperID
