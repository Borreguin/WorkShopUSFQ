# Define the Tower of Hanoi function with move counting
def tower_of_hanoi(n, source, auxiliary, target, move_counter):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        move_counter[0] += 1
        return
    tower_of_hanoi(n-1, source, target, auxiliary, move_counter)
    print(f"Move disk {n} from {source} to {target}")
    move_counter[0] += 1
    tower_of_hanoi(n-1, auxiliary, source, target, move_counter)

# Get the number of disks from the user
n = int(input("Enter the number of disks: "))

# Initialize move counter
move_counter = [0]

# Solve the Tower of Hanoi problem
print(f"Solving Tower of Hanoi for {n} disks:")
tower_of_hanoi(n, 'A', 'B', 'C', move_counter)

# Print the total number of moves
print(f"Total number of moves: {move_counter[0]}")

