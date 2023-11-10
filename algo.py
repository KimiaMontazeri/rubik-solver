import numpy as np
from collections import deque
from queue import PriorityQueue
from state import next_state, solved_state
from location import next_location, calculate_heuristic


NUM_OF_ACTIONS = 12

opposite_action_mapping = {
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 11,
    6: 12,
    7: 1,
    8: 2,
    9: 3,
    10: 4,
    11: 5,
    12: 6
}


def is_goal_state(state):
    return np.array_equal(state, solved_state())


def generate_successors(state):
    successors = []
    for action in range(NUM_OF_ACTIONS):
        new_state = next_state(state, action)
        successors.append((new_state, action))
    return successors


def generate_successors_with_location(state, location):
    successors = []
    for action in range(NUM_OF_ACTIONS):
        new_state = next_state(state, action)
        new_location = next_location(location, action)
        successors.append((new_state, new_location, action))
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


def a_star_search(initial_state, initial_location):
    visited = set()
    # Dictionary to store costs associated with states
    visited_costs = {}
    priority_queue = PriorityQueue()

    initial_cost = calculate_heuristic(initial_location)
    priority_queue.put((initial_cost, tuple(initial_state.flatten()), tuple(initial_location.flatten()), []))

    while not priority_queue.empty():
        current_cost, current_state_tuple, current_location_tuple, path = priority_queue.get()

        current_state = np.array(current_state_tuple).reshape((12, 2))
        current_location = np.array(current_location_tuple).reshape((2, 2, 2))
        if is_goal_state(current_state):
            return path

        if tuple(current_state.flatten()) not in visited or current_cost < visited_costs[tuple(current_state.flatten())]:
            visited.add(tuple(current_state.flatten()))
            visited_costs[tuple(current_state.flatten())] = current_cost

            successors = generate_successors_with_location(current_state, current_location)
            for successor_state, successor_location, action in successors:
                successor_cost = calculate_heuristic(successor_location) + len(path) + 1
                priority_queue.put((successor_cost, tuple(successor_state.flatten()), tuple(successor_location.flatten()), path + [action]))

    return None


def bidirectional_bfs(initial_state):
    visited_forward = {}  # Dictionary to store paths associated with states
    visited_backward = {}  # Dictionary to store paths associated with states

    queue_forward = deque([(initial_state, [])])
    queue_backward = deque([(solved_state(), [])])

    while queue_forward and queue_backward:
        # Forward BFS
        current_state_forward, path_forward = queue_forward.popleft()

        if tuple(current_state_forward.flatten()) in visited_backward:
            # Paths meet in the middle
            meeting_point = tuple(current_state_forward.flatten())
            path_backward = visited_backward[meeting_point]
            return path_forward + path_backward[::-1]

        visited_forward[tuple(current_state_forward.flatten())] = path_forward

        successors_forward = generate_successors(current_state_forward)
        for successor_state, action in successors_forward:
            if tuple(successor_state.flatten()) not in visited_forward:
                queue_forward.append((successor_state, path_forward + [action]))

        # Backward BFS
        current_state_backward, path_backward = queue_backward.popleft()

        if tuple(current_state_backward.flatten()) in visited_forward:
            # Paths meet in the middle
            meeting_point = tuple(current_state_backward.flatten())
            path_forward = visited_forward[meeting_point]
            return path_forward + path_backward[::-1]

        visited_backward[tuple(current_state_backward.flatten())] = path_backward

        successors_backward = generate_successors(current_state_backward)
        for successor_state, action in successors_backward:
            if tuple(successor_state.flatten()) not in visited_backward:
                queue_backward.append((successor_state, path_backward + [opposite_action_mapping[action]]))

    return None


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
        return list(a_star_search(init_state, init_location))

    elif method == 'BiBFS':
        return list(bidirectional_bfs(init_state))
    
    else:
        return []