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

    while queue:
        current = queue.popleft()
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
    return path

def dfs(graph, start, goal):
    stack = [start]
    visited = {start: None}

    while stack:
        current = stack.pop()
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
    return path

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

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
    return path

def visualize_solution(maze, path, output_file):
    height = len(maze)
    width = len(maze[0])

    fig = plt.figure(figsize=(width/4, height/4))
    for y in range(height):
        for x in range(width):
            cell = maze[y][x]
            if (y, x) in path:
                color = 'yellow'
            else:
                color = define_color(cell)
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color=color, edgecolor='black')

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()

def study_case(maze_file, output_file_bfs, output_file_dfs, output_file_astar):
    maze = MazeLoader(maze_file).load_Maze()
    graph = maze.get_graph()
    print(graph)

    start = next((y, x) for y, row in enumerate(maze.maze) for x, cell in enumerate(row) if cell == 'E')
    goal = next((y, x) for y, row in enumerate(maze.maze) for x, cell in enumerate(row) if cell == 'S')

    # Measure BFS performance
    start_time = time.time()
    path_bfs = bfs(graph, start, goal)
    bfs_time = time.time() - start_time

    # Measure DFS performance
    start_time = time.time()
    path_dfs = dfs(graph, start, goal)
    dfs_time = time.time() - start_time

    # Measure A* performance
    start_time = time.time()
    path_astar = a_star(graph, start, goal)
    astar_time = time.time() - start_time

    # Visualize solutions
    visualize_solution(maze.maze, path_bfs, output_file_bfs)
    visualize_solution(maze.maze, path_dfs, output_file_dfs)
    visualize_solution(maze.maze, path_astar, output_file_astar)

    # Compare results
    print(f"BFS: Path length = {len(path_bfs)}, Time = {bfs_time:.6f} seconds")
    print(f"DFS: Path length = {len(path_dfs)}, Time = {dfs_time:.6f} seconds")
    print(f"A*: Path length = {len(path_astar)}, Time = {astar_time:.6f} seconds")

def study_case_1():
    print("This is study case 1")
    study_case('laberinto1.txt', 'solution_bfs_1.jpg', 'solution_dfs_1.jpg', 'solution_astar_1.jpg')

def study_case_2():
    print("This is study case 2")
    study_case('laberinto2.txt', 'solution_bfs_2.jpg', 'solution_dfs_2.jpg', 'solution_astar_2.jpg')

def study_case_3():
    print("This is study case 3")
    study_case('laberinto3.txt', 'solution_bfs_3.jpg', 'solution_dfs_3.jpg', 'solution_astar_3.jpg')

if __name__ == '__main__':
    study_case_1()
    study_case_2()
    study_case_3()