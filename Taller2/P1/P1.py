import os
import sys
from collections import deque
import matplotlib.pyplot as plt
import time
import heapq

project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_MazeLoader import MazeLoader
from P1_util import define_color

def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start: None}
    nodes_visited = 0

    while queue:
        current = queue.popleft()
        nodes_visited += 1
        if current == goal:
            break
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    path = []
    while goal is not None:
        path.append(goal)
        goal = visited[goal]
    path.reverse()
    return path, nodes_visited

def dfs(graph, start, goal):
    stack = [start]
    visited = {start: None}
    nodes_visited = 0

    while stack:
        current = stack.pop()
        nodes_visited += 1
        if current == goal:
            break
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                stack.append(neighbor)

    path = []
    while goal is not None:
        path.append(goal)
        goal = visited[goal]
    path.reverse()
    return path, nodes_visited

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    nodes_visited = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_visited += 1

        if current == goal:
            break

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    path = []
    while goal is not None:
        path.append(goal)
        goal = came_from[goal]
    path.reverse()
    return path, nodes_visited

def visualize_solution(maze, path, output_file):
    height = len(maze)
    width = len(maze[0])

    fig = plt.figure(figsize=(width / 4, height / 4))
    for y in range(height):
        for x in range(width):
            cell = maze[y][x]
            if (y, x) in path:
                color = 'yellow'
            else:
                color = define_color(cell)
            plt.fill([x, x + 1, x + 1, x], [y, y, y + 1, y + 1], color=color, edgecolor='black')

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()

def visualize_results(times, nodes_visited):
    algorithms = ['BFS', 'DFS', 'A*']
    labyrinths = ['Labyrinth 1', 'Labyrinth 2', 'Labyrinth 3']

    # Time visualization
    fig, ax = plt.subplots()
    for i, lab in enumerate(labyrinths):
        ax.bar([x + i * 0.2 for x in range(len(algorithms))], times[i], width=0.2, label=lab)
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks([x + 0.2 for x in range(len(algorithms))])
    ax.set_xticklabels(algorithms)
    ax.legend()
    plt.title('Time Comparison of BFS, DFS, and A*')
    plt.show()

    # Nodes visited visualization
    fig, ax = plt.subplots()
    for i, lab in enumerate(labyrinths):
        ax.bar([x + i * 0.2 for x in range(len(algorithms))], nodes_visited[i], width=0.2, label=lab)
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Nodes Visited')
    ax.set_xticks([x + 0.2 for x in range(len(algorithms))])
    ax.set_xticklabels(algorithms)
    ax.legend()
    plt.title('Nodes Visited Comparison of BFS, DFS, and A*')
    plt.show()

def study_case(maze_file, output_file_bfs, output_file_dfs, output_file_astar):
    maze = MazeLoader(maze_file).load_Maze()
    graph = maze.get_graph()

    start = next((y, x) for y, row in enumerate(maze.maze) for x, cell in enumerate(row) if cell == 'E')
    goal = next((y, x) for y, row in enumerate(maze.maze) for x, cell in enumerate(row) if cell == 'S')

    # Measure BFS performance
    start_time = time.time()
    path_bfs, nodes_visited_bfs = bfs(graph, start, goal)
    bfs_time = time.time() - start_time

    # Measure DFS performance
    start_time = time.time()
    path_dfs, nodes_visited_dfs = dfs(graph, start, goal)
    dfs_time = time.time() - start_time

    # Measure A* performance
    start_time = time.time()
    path_astar, nodes_visited_astar = a_star(graph, start, goal)
    astar_time = time.time() - start_time

    # Visualize solutions
    visualize_solution(maze.maze, path_bfs, output_file_bfs)
    visualize_solution(maze.maze, path_dfs, output_file_dfs)
    visualize_solution(maze.maze, path_astar, output_file_astar)

    return [bfs_time, dfs_time, astar_time], [nodes_visited_bfs, nodes_visited_dfs, nodes_visited_astar]

def study_case_1():
    print("This is study case 1")
    return study_case('laberinto1.txt', 'solution_bfs_1.jpg', 'solution_dfs_1.jpg', 'solution_astar_1.jpg')

def study_case_2():
    print("This is study case 2")
    return study_case('laberinto2.txt', 'solution_bfs_2.jpg', 'solution_dfs_2.jpg', 'solution_astar_2.jpg')

def study_case_3():
    print("This is study case 3")
    return study_case('laberinto3.txt', 'solution_bfs_3.jpg', 'solution_dfs_3.jpg', 'solution_astar_3.jpg')

if __name__ == '__main__':
    times_1, nodes_visited_1 = study_case_1()
    times_2, nodes_visited_2 = study_case_2()
    times_3, nodes_visited_3 = study_case_3()

    times = [times_1, times_2, times_3]
    nodes_visited = [nodes_visited_1, nodes_visited_2, nodes_visited_3]

    visualize_results(times, nodes_visited)