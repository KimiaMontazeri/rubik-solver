from collections import deque
import numpy as np
from state import next_state, solved_state
from location import next_location


def is_goal_state(state):
    return np.array_equal(state, solved_state())


def generate_successors(state):
    successors = []
    for action in range(12):
        next = next_state(state, action)
        successors.append((next, action))
    return successors


def depth_limited_dfs(start_state, depth_limit):
    visited = set()
    stack = deque([(start_state, 0, [])])
    explored_states = 0

    while stack:
        current_state, depth, path = stack.pop()

        if is_goal_state(current_state):
            return path, explored_states

        if depth < depth_limit:
            visited.add(tuple(current_state.flatten()))
            explored_states += 1
            successors = generate_successors(current_state)

            for successor, action in successors:
                state_tuple = tuple(successor.flatten())

                if state_tuple not in visited:
                    visited.add(state_tuple)
                    stack.append((successor, depth + 1, path + [action]))

    return None, explored_states


def iterative_deepening_dfs(initial_state, max_depth=1000):
    depth_limit = 1
    while depth_limit <= max_depth:
        result, explored_states = depth_limited_dfs(start_state=np.copy(initial_state), depth_limit=depth_limit)
        if result is not None:
            print('path: ', result)
            print('Number of explored states: ', explored_states)
            print('Depth to reach the goal: ', depth_limit)
            return result
        depth_limit += 1


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.
 
    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == 'Random':
        return list(np.random.randint(1, 12+1, 10))
    
    elif method == 'IDS-DFS':
        return list(iterative_deepening_dfs(init_state))
    
    elif method == 'A*':
        ...

    elif method == 'BiBFS':
        ...
    
    else:
        return []