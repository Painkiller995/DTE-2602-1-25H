"""
This module solves the exercises Karaktersatt oppgave 1: Q-learning in DTE-2602-1 25H

This implementation could be improved in the feature please check github for the latest version.
https://github.com/Painkiller995/DTE-2602-1-25H

"""

import numpy as np
import random

ALPHA = 1.0  # Learning rate
GAMMA = 0.8  # Discount factor
EPSILON = 0.2  # Exploration rate

COLS = 6
ROWS = 6
NUMBER_OF_ACTIONS = 4  # 0: up, 1: down, 2: left, 3: right
NUMBER_OF_STATES = COLS * ROWS

START_CELL = (0, 3)
GOAL_CELL = (5, 0)

REWARDS = [
    [-50, -20, -20, -1.0, -1.0, -50],
    [-50, -50, -1.0, -20, -1.0, -1.0],
    [-20, -1.0, -1.0, -20, -1.0, -20],
    [-20, -1.0, -1.0, -1.0, -1.0, -1.0],
    [-20, -1.0, -20, -1.0, -20, -1.0],
    [100, -50, -50, -50, -50, -50],
]

# Action moves: up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Robot:
    """
    A reinforcement-learning robot that navigates a 6x6 grid using Q-learning.

    The robot learns optimal paths from a start cell to a goal cell
    through iterative updates to a Q-value table.
    """

    def __init__(self):
        self.q_matrix = np.zeros((NUMBER_OF_STATES, NUMBER_OF_ACTIONS))
        self.start_state = self.pos_to_state(START_CELL)
        self.goal_state = self.pos_to_state(GOAL_CELL)
        self.current_state = None

    def state_to_pos(self, state):
        """
        Convert a numeric state index into (row, col) grid coordinates.

        Args:
            state (int): The linear index of the state.

        Returns:
            tuple[int, int]: (row, col) position on the grid.
        """
        row = state // COLS
        col = state % COLS
        return (row, col)

    def pos_to_state(self, pos):
        """
        Convert a (row, col) grid position into a single numeric state index.

        Args:
            pos (tuple[int, int]): The (row, col) grid coordinates.

        Returns:
            int: Linear index representing the state.
        """
        row, col = pos
        return row * COLS + col

    def get_x(self):
        """Return the current x-coordinate (column index)."""
        _, col = self.state_to_pos(self.current_state)
        return col

    def get_y(self):
        """Return the current y-coordinate (row index)."""
        row, _ = self.state_to_pos(self.current_state)
        return row

    def select_action(self):
        """Select a random action (pure exploration)."""
        return random.randint(0, NUMBER_OF_ACTIONS - 1)

    def select_action_eg(self, state):
        """
        Select an action using the epsilon-greedy strategy.

        Args:
            state (int): Current state index.

        Returns:
            int: Chosen action index.
        """
        if random.random() < EPSILON:
            return self.select_action()
        else:
            return np.argmax(self.q_matrix[state])

    def get_next_state(self, state, action):
        """
        Compute the next state given the current state and an action.

        Ensures the agent does not move outside grid boundaries.

        Args:
            state (int): Current state index.
            action (int): Action index (0–3).

        Returns:
            int: Next state index after the move.
        """
        row, col = self.state_to_pos(state)
        d_row, d_col = ACTIONS[action]

        # Move with boundary check
        new_row = max(0, min(ROWS - 1, row + d_row))
        new_col = max(0, min(COLS - 1, col + d_col))

        return self.pos_to_state((new_row, new_col))

    def get_reward(self, state):
        """
        Retrieve the reward associated with a specific state.

        Args:
            state (int): State index.

        Returns:
            float: Reward value from the REWARDS grid.
        """
        row, col = self.state_to_pos(state)
        return REWARDS[row][col]

    def has_reached_goal(self):
        """Check if the robot has reached the goal state."""
        return self.current_state == self.goal_state

    def one_step_q_learning(self, use_eg=True):
        """
        Perform one iteration (step) of Q-learning.

        Args:
            use_eg (bool): If True, uses epsilon-greedy action selection.
                           If False, selects actions purely at random.
        """
        if self.current_state is None:
            self.current_state = self.start_state

        if self.has_reached_goal():
            return

        # Choose action
        if use_eg:
            action = self.select_action_eg(self.current_state)
        else:
            action = self.select_action()

        # Get next state and reward
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(next_state)

        # Q-learning update
        self.q_matrix[self.current_state][action] = (1 - ALPHA) * self.q_matrix[
            self.current_state
        ][action] + ALPHA * (reward + GAMMA * max(self.q_matrix[next_state]))

        # Move to next state
        self.current_state = next_state

    def q_learning(self, epochs, use_eg=False):
        """
        Train the robot over multiple episodes using Q-learning.

        Args:
            epochs (int): Number of training iterations (episodes).
            use_eg (bool): If True, uses epsilon-greedy exploration.
        """
        for _ in range(epochs):
            self.reset_start()
            while not self.has_reached_goal():
                self.one_step_q_learning(use_eg=use_eg)

    def reset_start(self):
        """Reset the robot's position to the start state."""
        self.current_state = self.start_state

    def reset_random(self):
        """Reset the robot to a random non-goal state."""
        while True:
            rand_state = random.randint(0, NUMBER_OF_STATES - 1)
            if rand_state != self.goal_state:
                self.current_state = rand_state
                break

    def greedy_path(self):
        """
        Compute the best (greedy) path from start to goal based on learned Q-values.

        Returns:
            list[tuple[int, int]]: Sequence of (row, col) positions representing the path.
        """
        state = self.start_state
        path = [self.state_to_pos(state)]
        visited = set([state])
        steps = 0
        while state != self.goal_state and steps < 200:
            # Velger handlingen med høyest Q-verdi i gjeldende tilstand (beste lærte handling)
            action = np.argmax(self.q_matrix[state])
            next_state = self.get_next_state(state, action)

            if next_state in visited:  # prevent infinite loops
                break

            path.append(self.state_to_pos(next_state))
            visited.add(next_state)
            state = next_state
            steps += 1

        return path

    def print_path(self):
        """Print the robot's greedy path from start to goal."""
        path = self.greedy_path()
        print("Path from start to goal:")
        print(" -> ".join(str(p) for p in path))


if __name__ == "__main__":
    robot_one = Robot()
    robot_one.q_learning(epochs=10000, use_eg=False)
    robot_one.print_path()

    robot_two = Robot()
    robot_two.q_learning(epochs=10000, use_eg=True)
    robot_two.print_path()
