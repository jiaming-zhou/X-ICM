from pyrep.robots.arms.arm import Arm


class Panda(Arm):

    def __init__(
        self,
        count: int = 0,
        max_velocity: float = 1.0,
        max_acceleration: float = 4.0
    ):
        super().__init__(
            count,
            'Panda',
            7,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
