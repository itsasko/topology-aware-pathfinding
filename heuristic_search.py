import heapq
import numpy as np


def euclidean_heuristic(u, v, points):
    return np.linalg.norm(points[u] - points[v])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def astar(G, points, start, goal):
    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float("inf") for node in G.nodes}
    g_score[start] = 0

    f_score = {node: float("inf") for node in G.nodes}
    f_score[start] = euclidean_heuristic(start, goal, points)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in G.neighbors(current):
            tentative_g = g_score[current] + G[current][neighbor]['weight']
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + euclidean_heuristic(neighbor, goal, points)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


def weighted_astar(G, points, start, goal, w=1.5):
    open_set = [(0, start)]
    came_from = {}
    g_score = {node: float("inf") for node in G.nodes}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in G.neighbors(current):
            tentative_g = g_score[current] + G[current][neighbor]['weight']
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + w * euclidean_heuristic(neighbor, goal, points)
                heapq.heappush(open_set, (f, neighbor))
    return None


def greedy_bfs(G, points, start, goal):
    open_set = [(0, start)]
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                came_from[neighbor] = current
                h = euclidean_heuristic(neighbor, goal, points)
                heapq.heappush(open_set, (h, neighbor))
    return None
