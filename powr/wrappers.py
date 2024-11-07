from gymnasium import RewardWrapper


class RewardRangeWrapper(RewardWrapper):
    def __init__(self, env ):
        super().__init__(env)

    def reward(self, reward):
        # Modify the rewards:
        # - reward of 1 (reaching the goal) becomes 0
        # - reward of 0 (falling into hole or stepping safely) becomes -1
        if reward == 1:
            return 0  # Reaching the goal becomes 0
        elif reward == 0:
            return -1  # Falling into a hole or staying safe becomes -1
        else:
            return reward