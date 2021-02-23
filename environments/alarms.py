import sys
from contextlib import closing
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

MAP = [
    "ooo",
    "ooo",
    "ooo",
]


class AlarmsEnv(discrete.DiscreteEnv):
    """
    Alarm Dashboard

    Description:
    Simple grid with random alarm that the agent need to turn off. The episode ends if all the alarm are on.
    At each step each alarm have a probability to turn on off 1/nb_alarm

    Observations:
    There are `2**(nrow*ncol)` discrete states alarm off (0) or alarm on (1) for each grid locations.

    Actions:
    There are `nrow*ncol + 1` discrete deterministic actions:
    - X: try to turn off the alarm at location X
    - Do nothing.

    Rewards:
    There is a default per-step reward of -1 for each alarm on,
    +10 to turn off an alarm,
    -5 to trying to turn off an alarm thats not on.

    Rendering:
    - o: alarm off
    - *: alarm on

    state space is represented by the grid
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")
        self.nrow = nrow = 3
        self.ncol = ncol = 3
        num_states = 2 ** (nrow * ncol)
        num_actions = (nrow * ncol) + 1

        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[0] = 1  # Initial state is always all the alarm are turned off
        P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}

        for state in range(num_states):
            grid = self.decode(state)
            for p, next_grid in self._next_grids(grid):
                for action in range(num_actions):
                    reward = -1 * grid.sum()  # default reward depending on the number of alarm on
                    new_grid = next_grid.copy()
                    if action != num_actions - 1:  # For all except the last action
                        i, j = self._action_to_pos(action)
                        # if the alarm we're trying to turn off is on turn it off else punish
                        if grid[i, j] == 1:
                            reward += 10
                            new_grid[i, j] = 0
                        else:
                            reward -= 5
                    # Encode new grid into it's corresponding state
                    new_state = self.encode(new_grid)
                    done = new_state == num_states - 1  # The episode is done when all the alarms are on. grid full of 1
                    P[state][action].append((p, new_state, reward, done))
        # Verify P
        for state in P.keys():
            for action in P[state].keys():
                assert np.array(P[state][action])[:, 0].sum() == 1, "Sum of probability should be equal to 1"

        initial_state_distrib /= initial_state_distrib.sum()
        super(AlarmsEnv, self).__init__(num_states, num_actions, P, initial_state_distrib)

    def _next_grids(self, grid):
        next_grids = []
        for row, col in zip(*np.where(grid == 0)):
            copy = grid.copy()
            copy[row][col] = 1
            next_grids.append((1 / grid.size, copy))
        if len(next_grids) < grid.size:
            next_grids.append((grid.sum() / grid.size, grid))
        next_grids = np.array(next_grids)
        # In the assert below, `round` accept the float error induced by python. for example `sum([1/9] * 9) == 1 -> False`
        assert round(next_grids[:, 0].sum()) == 1, "Sum of probability should be equal to 1"
        return next_grids

    def _action_to_pos(self, action):
        return action // self.ncol, action % self.ncol

    def encode(self, grid):
        """Transform grid into a state number"""
        if not hasattr(self, "encode_arr"):
            # Construct the array to transform the grid into an int
            self.encode_arr = (
                1 << np.arange(grid.ravel().size)[::-1]
            )  # -1 reverses array of powers of 2 of same length as bits
        return grid.ravel() @ self.encode_arr  # this matmult using @ is way fatser than using A.dot(B)

    def decode(self, num):
        """Transform a state number into a grid"""
        return (
            np.fromstring(np.binary_repr(num).zfill(self.nrow * self.ncol), dtype="S1")
            .astype(int)
            .reshape(self.nrow, self.ncol)
        )

    def render(self, mode="human", plot=False):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        grid = self.decode(self.s)
        for row, col in zip(*np.where(grid == 1)):
            desc[row][col] = utils.colorize("*", "red", highlight=True, bold=True)

        if self.lastaction is not None:
            outfile.write(" Turn off ({},{})\n".format(*self._action_to_pos(self.lastaction)))
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if plot:
            outfile.write("\n")
            plt.pcolormesh(grid, edgecolors="w", linewidth=2, cmap="seismic")
            plt.axis("off")  # remove axis
            plt.gca().invert_yaxis()  # pcolormesh invert y axis so we re-invert it
            plt.gca().set_aspect("equal")  # display the grid as a square
            plt.show()

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
