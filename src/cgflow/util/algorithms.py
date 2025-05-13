import random

import torch


def pairwise_distance_matrix(x):
    """
    Compute the pairwise distance matrix between a set of vectors.
    
    Args:
        x: A tensor of shape (B, N, D) representing N vectors of dimension D.
    """
    return (x.unsqueeze(-2) - x.unsqueeze(-3)).norm(dim=-1) 




def kabsch_alignment(P, Q):
    """
    Perform Kabsch alignment ensuring all tensors are on the same device.

    Args:
        P: Predicted coordinates (N, 3)
        Q: Target coordinates (N, 3)

    Returns:
        Aligned predicted coordinates (N, 3)
    """
    # Ensure P and Q are on the same device
    device = P.device

    # Center the coordinates (subtract mean)
    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # Singular Value Decomposition (SVD)
    U, S, Vt = torch.svd(H)

    # Ensure a right-handed coordinate system (determinant of the rotation matrix = 1)
    d = torch.det(U @ Vt).to(device)  # Ensure d is on the same device
    D = torch.eye(3, device=device)  # Create D on the same device
    D[2, 2] = d

    # Optimal rotation matrix
    R = U @ D @ Vt.T

    # Rotate the predicted coordinates
    P_aligned = P_centered @ R

    return P_aligned


def sample_connected_trajectory_bfs(graph):
    """
    Samples a connected trajectory using BFS, starting from a random node.

    Args:
        graph (dict): A dictionary where keys are nodes and values are lists of neighboring nodes.

    Returns:
        list: A list representing the connected trajectory.
    """
    if not graph:
        return [0]

    # Choose a random start node from the graph
    start_node = random.choice(list(graph.keys()))
    trajectory = [start_node]

    vistable = set(graph[start_node])
    visited = set([start_node])

    while len(trajectory) < len(graph):
        if not vistable:
            # Graph is disconnected. Picking a random node to continue the trajectory
            vistable = set(graph.keys()).difference(visited)

        # Choose a random neighbor to add to the trajectory
        current_node = random.choice(list(vistable))
        trajectory.append(current_node)
        visited.add(current_node)

        # Update the vistable with the neighbors of the current node
        neighbors = graph.get(current_node, [])
        vistable = vistable.union(neighbors).difference(visited)

    return trajectory
