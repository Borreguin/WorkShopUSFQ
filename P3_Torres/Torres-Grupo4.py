import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time

def plot_towers(towers, move_number):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Source", "Auxiliary", "Target"))

    for i, tower in enumerate(towers):
        fig.add_trace(go.Bar(x=[f"Disk {disk}" for disk in tower], y=tower, name=f"Tower {i+1}"), row=1, col=i+1)

    fig.update_layout(title_text=f"Move {move_number}", showlegend=False)
    fig.show()
    time.sleep(1)

def tower_of_hanoi(n, source, auxiliary, target, towers, move_counter):
    if n == 1:
        disk = towers[source].pop()
        towers[target].append(disk)
        move_counter[0] += 1
        plot_towers(towers, move_counter[0])
        return
    tower_of_hanoi(n-1, source, target, auxiliary, towers, move_counter)
    disk = towers[source].pop()
    towers[target].append(disk)
    move_counter[0] += 1
    plot_towers(towers, move_counter[0])
    tower_of_hanoi(n-1, auxiliary, source, target, towers, move_counter)

# Get the number of disks from the user
n = int(input("Enter the number of disks: "))

# Initialize move counter and towers
move_counter = [0]
towers = [list(range(n, 0, -1)), [], []]

# Solve the Tower of Hanoi problem
print(f"Solving Tower of Hanoi for {n} disks:")
plot_towers(towers, move_counter[0])
tower_of_hanoi(n, 0, 1, 2, towers, move_counter)

# Print the total number of moves
print(f"Total number of moves: {move_counter[0]}")