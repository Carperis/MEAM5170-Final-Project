"""SLIP simulation with PyBullet."""

import logging
import time

import numpy as np
import pybullet as p
import pybullet_data as pd

logger = logging.getLogger(__name__)


class Simulation:

    TIMESTEP = 1 / 240

    NOMINAL_LEG_LENGTH = 0.5
    SPRING_CONSTANT = 6000

    PNEUMATIC_JOINT_INDEX = 2
    TIP_LINK_INDEX = 2

    def __init__(
        self, velocity_gain: float = 0.027, initial_stance_duration: float = 0.17
    ) -> None:
        """Initializes the simulation object.

        Args:
            velocity_gain: Velocity gain used by the flight controller to compute target
                leg displacements.
            initial_stance_duration: Stance duration used by the flight controller
                before the first stance phase.
        """

        self.velocity_gain = velocity_gain
        self.stance_duration = initial_stance_duration

        self.target_velocity = np.array([0.3, 0.3])
        """Array of shape `(2,)` containing the target `[x, y]` velocity."""

        # Initialize PyBullet.
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)

        # TODO: Change some dynamics.

        # Load the plane and hopper.
        self.plane_id = p.loadURDF("plane.urdf")
        self.hopper_id = p.loadURDF("./slip/urdf/slip.urdf", basePosition=[0, 0, 1])

        # Log joint information.
        for joint_index in range(p.getNumJoints(self.hopper_id)):
            logger.info(
                "Joint %d info: %s",
                joint_index,
                p.getJointInfo(self.hopper_id, joint_index),
            )

    def has_contact(self) -> bool:
        """Checks if the tip of the hopper is in contact with the ground.

        Returns:
            A boolean indicating whether the tip is in contact with the ground.
        """

        world_position = p.getLinkState(self.hopper_id, self.TIP_LINK_INDEX)[0]
        # TODO: Explain the 0.0504.
        return world_position[2] < 0.0504

    def get_velocity(self) -> np.ndarray:
        """Gets the linear velocity of the hopper's base.

        Returns:
            Array of shape `(3,)` containing the `[x, y, z]` velocity.
        """

        linear_velocity = p.getBaseVelocity(self.hopper_id)[0]
        return np.array(linear_velocity)

    def get_leg_length(self) -> float:
        """Gets the current length of the leg.

        Returns:
            The current length of the leg.
        """

        position = p.getJointState(self.hopper_id, self.PNEUMATIC_JOINT_INDEX)[0]
        return self.NOMINAL_LEG_LENGTH - position

    def get_target_leg_displacement(self) -> np.ndarray:
        """Computes the target leg displacement based on the target velocity.

        Returns:
            Array of shape `(3,)` containing the target `[x, y, z]` displacement.
        """

        velocity = self.get_velocity()[0:2]
        velocity_error = velocity - self.target_velocity

        # TODO: Should use average velocity to compute neutral point.
        neutral_point = velocity * (self.stance_duration / 2)

        xy_displacement = neutral_point + self.velocity_gain * velocity_error
        leg_length = self.get_leg_length()
        z_displacement = -np.sqrt(
            leg_length**2 - xy_displacement[0] ** 2 - xy_displacement[1] ** 2
        )

        if np.isnan(z_displacement):
            logger.error(
                "Legs are too short: xy_displacement=%s, leg_length=%f",
                xy_displacement,
                leg_length,
            )
            z_displacement = 0
            xy_displacement /= np.linalg.norm(xy_displacement) / leg_length

        return np.append(xy_displacement, z_displacement)

    def set_leg_force(self, compensation: float) -> None:
        """Sets the force applied to the leg joint.

        Args:
            compensation: The force to apply to the leg joint in addition to the spring
                force.
        """

        leg_position = p.getJointState(self.hopper_id, self.PNEUMATIC_JOINT_INDEX)[0]
        leg_force = -(self.SPRING_CONSTANT * leg_position + compensation)
        p.setJointMotorControl2(
            self.hopper_id,
            self.PNEUMATIC_JOINT_INDEX,
            p.TORQUE_CONTROL,
            force=leg_force,
        )

    def stance_control(self) -> None:

        self.set_leg_force(compensation=800)

    def flight_control(self) -> None:
        """TODO"""

        leg_displacement = self.get_target_leg_displacement()

        self.set_leg_force(compensation=0)

    def run(self) -> None:
        start_time = time.perf_counter()
        step_count = 0

        while True:
            try:
                if self.has_contact():
                    self.stance_control()
                else:
                    self.flight_control()
                self.get_velocity()

                p.stepSimulation()
            except p.error as e:
                logger.error("PyBullet error: %s", e)
                break

            # Sleep to maintain real-time simulation.
            step_count += 1
            expected_time = start_time + step_count * self.TIMESTEP
            actual_time = time.perf_counter()
            if expected_time > actual_time:
                time.sleep(expected_time - actual_time)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
