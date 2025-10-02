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
    def __init__(self):
        self.q_matrix = np.zeros((NUMBER_OF_STATES, NUMBER_OF_ACTIONS))
        self.start_state = self.pos_to_state(START_CELL)
        self.goal_state = self.pos_to_state(GOAL_CELL)
        self.current_state = None

    def state_to_pos(self, state):
        row = state // COLS
        col = state % COLS
        return (row, col)

    def pos_to_state(self, pos):
        row, col = pos
        return row * COLS + col

    def select_action(self):
        return random.randint(0, NUMBER_OF_ACTIONS - 1)

    def select_action_eg(self, state):
        if random.random() < EPSILON:
            return self.select_action()
        else:
            return np.argmax(self.q_matrix[state])

    def get_reward(self, state):
        row, col = self.state_to_pos(state)
        return REWARDS[row][col]

    def get_next_state(self, state, action):
        row, col = self.state_to_pos(state)
        d_row, d_col = ACTIONS[action]

        # Move with boundary check
        new_row = max(0, min(ROWS - 1, row + d_row))
        new_col = max(0, min(COLS - 1, col + d_col))

        return self.pos_to_state((new_row, new_col))

    def one_episode_q_learning(self, use_eg=False):
        self.current_state = self.start_state
        while not self.has_reached_goal():
            if use_eg:
                action = self.select_action_eg(self.current_state)
            else:
                action = self.select_action()

            next_state = self.get_next_state(self.current_state, action)
            reward = self.get_reward(next_state)

            # Q-learning update
            self.q_matrix[self.current_state][action] = (1 - ALPHA) * self.q_matrix[
                self.current_state
            ][action] + ALPHA * (reward + GAMMA * max(self.q_matrix[next_state]))

            self.current_state = next_state

    def q_learning(self, epochs, use_eg=False):
        for _ in range(epochs):
            self.one_episode_q_learning(use_eg=use_eg)

    def has_reached_goal(self):
        return self.current_state == self.goal_state

    def greedy_path(self):
        """Return the greedy path following max-Q actions."""
        state = self.start_state
        path = [self.state_to_pos(state)]
        visited = set([state])
        steps = 0
        while state != self.goal_state and steps < 200:
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
        path = self.greedy_path()
        print("Path from start to goal:")
        print(" -> ".join(str(p) for p in path))


if __name__ == "__main__":
    robot_one = Robot()
    robot_one.q_learning(epochs=10000, use_eg=False)
    robot_one.print_path()

    robot_two = Robot()
    robot_two.q_learning(epochs=50000, use_eg=True)
    robot_two.print_path()
