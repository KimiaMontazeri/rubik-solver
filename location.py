import numpy as np


def solved_location():
    return np.array([
        [[1, 2],
        [3, 4]],
        [[5, 6],
        [7, 8]],
    ], dtype=np.uint8)


def next_location(location, action):

    location = np.copy(location)

    # left to up
    if action == 1:
        location[:, :, 0] = np.rot90(location[:, :, 0], 1)

    # right to up
    elif action == 2:
        location[:, :, 1] = np.rot90(location[:, :, 1], 1)

    # down to left
    elif action == 3:
        location[:, 1, :] = np.rot90(location[:, 1, :], 1)

    # up to left
    elif action == 4:
        location[:, 0, :] = np.rot90(location[:, 0, :], 1)

    # back to right
    elif action == 5:
        location[1, :, :] = np.rot90(location[1, :, :], -1)

    # front to right
    elif action == 6:
        location[0, :, :] = np.rot90(location[0, :, :], -1)

    # left to down
    if action == 7:
        location[:, :, 0] = np.rot90(location[:, :, 0], -1)

    # right to down
    if action == 8:
        location[:, :, 1] = np.rot90(location[:, :, 1], -1)
        
    # down to right
    elif action == 9:
        location[:, 1, :] = np.rot90(location[:, 1, :], -1)

    # up to right
    elif action == 10:
        location[:, 0, :] = np.rot90(location[:, 0, :], -1)

    # back to left
    elif action == 11:
        location[1, :, :] = np.rot90(location[1, :, :], 1)

    # front to left
    elif action == 12:
        location[0, :, :] = np.rot90(location[0, :, :], 1)

    return location


def manhattan_distance(a, b):
    return np.sum(np.abs(np.subtract(a, b)))


def calculate_heuristic(current_location):
    goal_location = solved_location()
    total_distance = 0

    for goal_value in range(1, np.max(goal_location) + 1):
        goal_index = np.argwhere(goal_location == goal_value)[0]
        current_index = np.argwhere(current_location == goal_value)[0]

        distance = manhattan_distance(goal_index, current_index)
        total_distance += distance

    return total_distance // 4


if __name__ == '__main__':
    initial_location = solved_location()
    print('intial location:')
    print(initial_location)
    print()
    child_location = next_location(initial_location, action=4)
    print('next location:')
    print(child_location)