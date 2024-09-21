from collections import deque

# Define the initial and goal states
initial_state = (0, 0, 0, 0)  # (farmer, wolf, goat, cabbage)
goal_state = (1, 1, 1, 1)  # (farmer, wolf, goat, cabbage)

# Define the possible moves
moves = [
    (1, 0, 0, 0),  # Farmer moves alone
    (1, 1, 0, 0),  # Farmer moves with wolf
    (1, 0, 1, 0),  # Farmer moves with goat
    (1, 0, 0, 1)   # Farmer moves with cabbage
]

# Check if a state is valid
def is_valid(state):
    farmer, wolf, goat, cabbage = state
    if (wolf == goat and farmer != wolf) or (goat == cabbage and farmer != goat):
        return False
    return True

# Generate the next states from the current state
def get_next_states(state):
    next_states = []
    farmer, wolf, goat, cabbage = state
    for move in moves:
        new_state = (
            farmer ^ move[0],
            wolf ^ move[1],
            goat ^ move[2],
            cabbage ^ move[3]
        )
        if is_valid(new_state):
            next_states.append(new_state)
    return next_states

# Map state values to names
def state_to_names(state):
    names = ['farmer', 'wolf', 'goat', 'cabbage']
    left_shore = [names[i] for i in range(4) if state[i] == 0]
    right_shore = [names[i] for i in range(4) if state[i] == 1]
    return left_shore, right_shore

# Use BFS to find the solution
def solve_riddle():
    queue = deque([(initial_state, [])])
    visited = set()
    visited.add(initial_state)

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path + [current_state]

        for next_state in get_next_states(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [current_state]))

    return None

# Solve the riddle and print the solution
solution = solve_riddle()
if solution:
    print("Solution found:")
    for step in solution:
        left_shore, right_shore = state_to_names(step)
        print(f"Left shore: {left_shore}, Right shore: {right_shore}")
else:
    print("No solution found.")