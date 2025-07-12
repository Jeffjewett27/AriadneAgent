from dataclasses import dataclass
from typing import NewType, Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import networkx as nx

from motion import MotionPrimitive
from physics.environment import Segment, Terrain
from physics.hero_controller import FIXED_UPDATE_FRAMERATE, forward_dynamics
from physics.hero_controller_state import HeroControllerStates
from physics.player_input import PlayerInput


MotionStateType = NewType("MotionStateType", np.ndarray)


@dataclass
class RRTNode:
    id: int  # Unique identifier for the node
    rrt_state: MotionStateType  # Position and velocity state
    sim_state: HeroControllerStates  # Full simulation state
    segment: int | None # the hash of the segment it rests on (if grounded) else None
    reachable: Dict[int, float] = None  # Maps segment hash of reachable segments to distance
    sources: Dict[int, float] = None # Maps segment hash of source segments to air_time
    air_time: float = 0


class RRTGraph:
    def __init__(self, segments: list[Segment] = None):
        self.graph = nx.DiGraph()
        self.next_id = 0
        self.nodes: Dict[int, RRTNode] = {}
        self.bounds_min = None
        self.bounds_max = None
        self.floor_connection_graph = None
        self.floor_segments = {} if segments is None else {hash(seg): seg for seg in segments if seg.type == 'floor'}
        # Private state for sampling
        self._priv_graph = None
        self._is_sampling = False
        self._terrain = None

    def add_node(self, rrt_state, sim_state, parent_id=None, control=None, duration=None, segment=None):
        node_id = self.next_id
        self.next_id += 1
        
        node = RRTNode(
            id=node_id,
            rrt_state=rrt_state,
            sim_state=sim_state,
            segment=segment,
            reachable={},
            sources={},
            air_time=0
        )
        if duration is None:
            duration = 0

        if parent_id is not None:
            # add parent sources
            parent = self.nodes[parent_id]
            for source, time in parent.sources.items():
                if parent.segment is not None and source != parent.segment:
                    # grounded parents only propagate their segment
                    continue
                node.sources[source] = time + duration
        if segment is not None:
            # is grounded
            node.sources[segment] = 0
            node.reachable[segment] = 0
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        
        if parent_id is not None and control is not None and duration is not None:
            # Store control and duration as edge attributes
            self.graph.add_edge(
                parent_id, 
                node_id, 
                control=control, 
                duration=duration,
                weight=duration  # Use duration as the edge weight for shortest path
            )
        
        # Update bounds
        if self.bounds_min is None:
            self.bounds_min = rrt_state.copy()
            self.bounds_max = rrt_state.copy()
        else:
            self.bounds_min = np.minimum(self.bounds_min, rrt_state)
            self.bounds_max = np.maximum(self.bounds_max, rrt_state)
        
        return node_id
    
    def get_nearest_node(self, target_state):
        """Find the ID of the node closest to the target state"""
        return min(
            self.nodes.items(), 
            key=lambda item: np.linalg.norm(item[1].rrt_state - target_state)
        )[0]
    
    def get_k_nearest_nodes(self, target_state, k=5):
        """Find the IDs of the k nodes closest to the target state"""
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda item: np.linalg.norm(item[1].rrt_state - target_state)
        )
        return [node_id for node_id, _ in sorted_nodes[:k]]
    
    def update_segment_reachability(self, terrain):
        """
        Updates the reachable dictionary for each node based on grounded states
        and propagates distances up the graph using Bellman backups.
        """
        # First pass: Identify grounded states and their segments
        grounded_nodes = {}
        for node_id, node in self.nodes.items():
            if node.segment is not None:
                node.reachable[node.segment] = 0.0  # Zero distance to the segment it's on
                grounded_nodes[node_id] = node.segment
        
        # Second pass: Propagate distances using Bellman backups
        # We'll do this by working backwards through the graph
        # First, build a reverse graph for easier parent access
        reverse_graph = defaultdict(list)
        for u, v in self.graph.edges():
            reverse_graph[v].append(u)
        
        # Process nodes in topological order (approximate with BFS from grounded nodes)
        visited = set(grounded_nodes.keys())
        reachable = set()
        queue = list(visited)
        self.temp_multi = dict()
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]

            # Update parents of this node
            for parent_id in reverse_graph[node_id]:
                parent = self.nodes[parent_id]

                if node.segment is not None and parent.segment == node.segment:
                    # don't propagate same platform
                    continue
                
                # Only propagate to non-grounded parents
                edge_data = self.graph.get_edge_data(parent_id, node_id)
                edge_cost = edge_data.get('duration', 0.0)
                
                # Update reachability information
                updated = False
                for segment_hash, distance in node.reachable.items():
                    if node.segment is not None and segment_hash != node.segment:
                        # grounded items should only propapate own segment
                        continue
                    new_distance = distance + edge_cost
                    
                    if (segment_hash not in parent.reachable or 
                        new_distance < parent.reachable[segment_hash]):
                        parent.reachable[segment_hash] = new_distance
                        updated = True
                
                # Add parent to queue if it was updated and not already processed
                if updated and parent_id not in visited:
                    visited.add(parent_id)
                    queue.append(parent_id)
                
                if updated:
                    reachable.add(parent_id)
                    reachable.add(node_id)

                if len(parent.reachable) > 1:
                    print('reachable', node_id, parent_id, parent.segment, parent.reachable)
                    self.temp_multi[parent.id] = parent.reachable

        for node_id in reachable:
            node = self.nodes[node_id]
            reachable_child = False
            for child_id in self.graph[node_id]:
                if child_id in reachable:
                    reachable_child = True
                # assert self.nodes[child_id].segment != node.segment
            assert node.segment is not None or reachable_child
        return reachable

    def connect_same_segment_nodes(self, floor_segments: list[Segment], terrain: Terrain, run_speed: float = 8.3):
        """
        Connect nodes that are on the same floor segment with a direct edge.
        The cost is calculated as horizontal distance / RUN_SPEED.
        
        Args:
            run_speed: The run speed constant (default from hero_controller.py)
        """
        # Group nodes by segment
        segment_nodes = {}
        for node_id, node in self.nodes.items():
            if node.segment is None:
                continue
                
            if node.segment not in segment_nodes:
                segment_nodes[node.segment] = []
            segment_nodes[node.segment].append(node_id)
        
        SEG_SPACING = 3
        # Sort nodes by x position for consistent connectivity
        for segment_id, nodes in segment_nodes.items():
            nodes.sort(key=lambda node_id: self.nodes[node_id].rrt_state[0])
            segment = self.floor_segments[segment_id]
            n0: RRTNode = self.nodes[nodes[0]]
            while n0.rrt_state[0] - segment.x_min > SEG_SPACING:
                n0: RRTNode = self.nodes[nodes[0]]
                new_x = n0.rrt_state[0] - SEG_SPACING
                if new_x < 0 or new_x < segment.x_min:
                    break
                rrt_state = np.array([new_x, n0.rrt_state[1], n0.rrt_state[2]])
                new_state = n0.sim_state.copy()
                new_state.x_pos = new_x
                new_id = self.add_node(rrt_state, new_state, segment=segment_id)
                nodes.insert(0, new_id)

            n_last: RRTNode = self.nodes[nodes[-1]]
            while segment.x_max - n_last.rrt_state[0] > SEG_SPACING:
                n_last: RRTNode = self.nodes[nodes[-1]]
                new_x = n_last.rrt_state[0] + SEG_SPACING
                if new_x > terrain.scene_width or new_x > segment.x_max:
                    break
                rrt_state = np.array([new_x, n_last.rrt_state[1], n_last.rrt_state[2]])
                new_state = n_last.sim_state.copy()
                new_state.x_pos = new_x
                new_id = self.add_node(rrt_state, new_state, segment=segment_id)
                nodes.append(new_id)
            
            # Connect nodes within the same segment
            for i in range(len(nodes) - 1):
                node_id1 = nodes[i]
                node_id2 = nodes[i + 1]
                node1 = self.nodes[node_id1]
                node2 = self.nodes[node_id2]
                
                # Calculate direct cost based on horizontal distance
                x_displacement = abs(node1.rrt_state[0] - node2.rrt_state[0])
                cost = x_displacement / run_speed
                
                # Add bidirectional edges
                self.graph.add_edge(
                    node_id1, 
                    node_id2,
                    control=None,  # No specific control, just direct movement
                    duration=cost,
                    weight=cost
                )
                self.graph.add_edge(
                    node_id2, 
                    node_id1,
                    control=None,
                    duration=cost,
                    weight=cost
                )

    def prune_non_essential_nodes(self, essential_node_ids: set[int]):
        """
        Prune nodes that are not essential for floor connectivity.
        A node is essential if it can reach multiple different floor segments.
        """
        # Keep track of nodes to keep
        essential_node_ids = set()
        
        # # Nodes that can reach multiple floor segments are essential
        for node_id, node in self.nodes.items():
            # Nodes that can reach another floor segment
            new_segments = set(node.reachable) | set(node.sources)
            # if node.segment is not None:
            #     new_segments.discard(node.segment)
            if len(new_segments) > 1:
                essential_node_ids.add(node_id)
        
        # Remove non-essential nodes
        non_essential_nodes = set(self.nodes.keys()) - essential_node_ids
        for node_id in non_essential_nodes:
            # Remove node and its edges
            self.graph.remove_node(node_id)
            del self.nodes[node_id]
            # self.nodes[node_id].rrt_state[2] = 100
            
        print(f"Pruned graph from {len(self.nodes) + len(non_essential_nodes)} to {len(self.nodes)} nodes")

    def find_path(self, start_state: np.ndarray | HeroControllerStates, goal_state: np.ndarray | HeroControllerStates, k_nearest=1):
        """
        Find a path through the RRT graph from start to goal without modifying the graph
        
        Args:
            graph: The RRT graph
            start_state: Starting state
            goal_state: Goal state
            k_nearest: Number of nearest neighbors to consider
            
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if len(self.nodes) == 0:
            print('no nodes in graph')
            return [], []
        # Find nearest nodes to start and goal
        if isinstance(start_state, HeroControllerStates):
            start_state = sim_state_to_rrt_state(start_state)
        start_state = np.array([*start_state, 0])[:3]
        if isinstance(goal_state, HeroControllerStates):
            goal_state = sim_state_to_rrt_state(goal_state)
        goal_state = np.array([*goal_state, 0])[:3]
        start_neighbors = self.get_k_nearest_nodes(start_state, k=k_nearest)
        goal_neighbors = self.get_k_nearest_nodes(goal_state, k=k_nearest)
        appx_start_pos = self.nodes[start_neighbors[0]].rrt_state
        appx_goal_pos = self.nodes[goal_neighbors[0]].rrt_state
        print(f'finding path from {start_state} to {goal_state} via {appx_start_pos} and {appx_goal_pos}')
        
        # Try to find a path between any pair of neighbors
        best_path = None
        best_path_length = float('inf')
        
        for start_id in start_neighbors:
            for goal_id in goal_neighbors:
                try:
                    path = nx.shortest_path(self.graph, start_id, goal_id, weight='weight')
                    path_length = sum(
                        self.graph[path[i]][path[i+1]]['weight'] 
                        for i in range(len(path)-1)
                    )
                    
                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                except nx.NetworkXNoPath:
                    continue
        
        if best_path is None:
            print('no path found')
            return [], []
        
        trajectory = []
        target_segments = []
        curr_segment = None
        for u, v in zip(best_path[:-1], best_path[1:]):
            node_u = self.nodes[u]
            control = self.graph[u][v]['control']
            trajectory.append((node_u.sim_state, control))
            if node_u.segment != curr_segment and node_u.segment is not None:
                target_segments.append(node_u.segment)
                curr_segment = node_u.segment

        return trajectory, target_segments

    def sample_connections_sync(self, terrain: Terrain, num_samples: int = 1000, 
                          samples_per_segment: int = 5, max_edge_duration: float = 0.8,
                          prune_nodes: bool = True, connect_same_segment: bool = True,
                          publish_interval: int = 500):
        """
        Synchronously build a connected graph of motion primitives across all floor segments.
        
        Args:
            terrain: The terrain containing segments
            num_samples: Total number of random samples to generate
            samples_per_segment: Number of equally spaced root nodes per segment
            max_edge_duration: Maximum duration for edge connections
            prune_nodes: Whether to prune non-essential nodes
            connect_same_segment: Whether to connect nodes on the same floor segment
            publish_interval: How often to prune and update the public graph
            
        Returns:
            RRTGraph: Self, with the final graph
        """
        if self._is_sampling:
            raise RuntimeError("Already sampling connections")
            
        floor_segments = [seg for seg in terrain.segments if seg.type == 'floor']
        if not floor_segments:
            return self
            
        # Initialize private sampling state
        self._is_sampling = True
        self._terrain = terrain
        self._priv_graph = RRTGraph(floor_segments)
        
        # Create root nodes along each floor segment
        self._initialize_floor_nodes(terrain, samples_per_segment)
        
        # Sample nodes
        print(f'Sampling RRT graph for terrain with {num_samples} samples.')
        for i in range(num_samples):
            if i % publish_interval == 0 and i > 0:
                print(f'{i} samples done. Publishing intermediate results...')
                self._publish_pruned_graph(prune_nodes, connect_same_segment)
                
            self._sample_one_node(terrain, i, num_samples, max_edge_duration)
        
        # Final update
        if num_samples % publish_interval != 0:
            self._publish_pruned_graph(prune_nodes, connect_same_segment)
            
        # Clean up
        self._is_sampling = False
        self._priv_graph = None
        self._terrain = None
        
        return self
        
    async def sample_connections_async(self, terrain: Terrain, num_samples: int = 1000, 
                                samples_per_segment: int = 5, max_edge_duration: float = 0.8,
                                prune_nodes: bool = True, connect_same_segment: bool = True,
                                publish_interval: int = 500):
        """
        Asynchronously build a connected graph of motion primitives across all floor segments.
        
        Args:
            terrain: The terrain containing segments
            num_samples: Total number of random samples to generate
            samples_per_segment: Number of equally spaced root nodes per segment
            max_edge_duration: Maximum duration for edge connections
            prune_nodes: Whether to prune non-essential nodes
            connect_same_segment: Whether to connect nodes on the same floor segment
            publish_interval: How often to prune and update the public graph
            
        Returns:
            RRTGraph: Self, with the final graph
        """
        if self._is_sampling:
            raise RuntimeError("Already sampling connections")
            
        import asyncio
        
        floor_segments = [seg for seg in terrain.segments if seg.type == 'floor']
        if not floor_segments:
            return self
            
        # Initialize private sampling state
        self._is_sampling = True
        self._terrain = terrain
        self._priv_graph = RRTGraph(floor_segments)
        
        # Create root nodes along each floor segment
        self._initialize_floor_nodes(terrain, samples_per_segment)
        
        # Sample nodes
        print(f'Sampling RRT graph for terrain with {num_samples} samples.')
        for i in range(num_samples):
            if i % publish_interval == 0 and i > 0:
                print(f'{i} samples done. Publishing intermediate results...')
                self._publish_pruned_graph(prune_nodes, connect_same_segment)
                # Yield to event loop periodically
                await asyncio.sleep(0)
                
            self._sample_one_node(terrain, i, num_samples, max_edge_duration)
            
            # Yield to event loop occasionally to prevent blocking
            if i % 50 == 0:
                await asyncio.sleep(0)
        
        # Final update
            if num_samples % publish_interval != 0:
                self._publish_pruned_graph(prune_nodes, connect_same_segment)
            
        # Clean up
        self._is_sampling = False
        self._priv_graph = None
        self._terrain = None
        
        return self
    
    def _initialize_floor_nodes(self, terrain: Terrain, samples_per_segment: int):
        """Initialize root nodes on each floor segment"""
        for segment in self._priv_graph.floor_segments.values():
            # Sample points along the segment
            x_coords = np.linspace(segment.x_min, segment.x_max, samples_per_segment)
            y_coords = np.ones_like(x_coords) * segment.y_min  # Assuming floor segments are flat
            
            for i in range(samples_per_segment):
                root_state = np.array([x_coords[i], y_coords[i], 0.0])  # x, y, y_velocity
                
                # Skip if outside scene bounds
                if not is_within_scene_bounds(root_state, terrain):
                    continue
                    
                sim_state = HeroControllerStates(
                    x_pos=root_state[0],
                    y_pos=root_state[1],
                    y_velocity=root_state[2],
                    onGround=True
                )
                segment_hash = hash(segment)
                self._priv_graph.add_node(root_state, sim_state, segment=segment_hash)
    
    def _sample_one_node(self, terrain: Terrain, i: int, num_samples: int, max_edge_duration: float):
        """Sample a single node and add it to the private graph"""
        def duration_schedule(i):
            # Potentially decrease duration as we sample more nodes
            max_duration = 1.0
            min_duration = 0.2
            return max_duration - (max_duration - min_duration) * (i / num_samples)
            
        # Sample random state within scene bounds
        state_rand = np.array([
            np.random.uniform(0, terrain.scene_width),
            np.random.uniform(0, terrain.scene_height),
            0.0  # reasonable y_velocity range
        ])

        if np.random.uniform() < 0.5:
            state_rand[2] = np.random.beta(4, 1) * 20
        else:
            state_rand[2] = np.random.beta(1, 4) * -10 + 2
        
        # Find closest node in the graph
        nearest_id = self._priv_graph.get_nearest_node(state_rand)
        parent_node = self._priv_graph.nodes[nearest_id]
        
        # Sample control and simulate
        duration = min(duration_schedule(i), max_edge_duration)
        control = MotionPrimitive.sample(duration)
        full_state = parent_node.sim_state
        
        # Forward simulation
        for action in PlayerInput.get_action_iterator(
            control.control_sequence(), FIXED_UPDATE_FRAMERATE
        ):
            full_state = forward_dynamics(
                full_state, terrain, action, 1 / FIXED_UPDATE_FRAMERATE
            )
        
        # Create new node
        new_state = sim_state_to_rrt_state(full_state)
        
        # Skip if outside scene bounds
        if not is_within_scene_bounds(new_state, terrain):
            return
        
        if full_state.onGround:
            floor_segment = hash(get_ground_segment(full_state, terrain))
        else:
            floor_segment = None
            
        # Add node to private graph
        self._priv_graph.add_node(
            rrt_state=new_state,
            sim_state=full_state,
            parent_id=nearest_id,
            control=control,
            duration=duration,
            segment=floor_segment
        )
    
    def _publish_pruned_graph(self, prune_nodes: bool, connect_same_segment: bool):
        """Prune the private graph and publish it to the public interface"""
        # Update reachability in the private graph
        essential_node_ids = self._priv_graph.update_segment_reachability(self._terrain)
        
        # Copy of the graph for pruning and processing
        working_graph = RRTGraph(list(self._priv_graph.floor_segments.values()))
        working_graph.graph = self._priv_graph.graph.copy()
        working_graph.nodes = {k: v for k, v in self._priv_graph.nodes.items()}
        working_graph.next_id = self._priv_graph.next_id
        working_graph.bounds_min = self._priv_graph.bounds_min.copy() if self._priv_graph.bounds_min is not None else None
        working_graph.bounds_max = self._priv_graph.bounds_max.copy() if self._priv_graph.bounds_max is not None else None
        
        # Optionally prune non-essential nodes
        if prune_nodes:
            working_graph.prune_non_essential_nodes(essential_node_ids)
            
        # Optionally connect nodes on the same floor segment
        if connect_same_segment:
            working_graph.connect_same_segment_nodes(list(working_graph.floor_segments.values()), self._terrain)
        
        # Publish to the public interface
        self.graph = working_graph.graph
        self.nodes = working_graph.nodes
        self.next_id = working_graph.next_id
        self.bounds_min = working_graph.bounds_min
        self.bounds_max = working_graph.bounds_max
        self.floor_segments = working_graph.floor_segments
        self.floor_connection_graph = compute_floor_connections(self, self._terrain)


def get_ground_segment(sim_state, terrain):
    """
    Returns the floor segment that the state is grounded on.
    This is a stub function - you'll need to implement the actual logic.
    """
    if not sim_state.onGround:
        return None
    
    segments = terrain.find_touching_segments(
        np.array([sim_state.x_pos, sim_state.y_pos])
    )
    segments = [seg for seg in segments if seg.type == 'floor']
    if len(segments) == 0:
        return None
    
    return segments[0]


def is_within_scene_bounds(state: np.ndarray, terrain: Terrain) -> bool:
    """
    Check if a state's position is within the scene boundaries.
    
    Args:
        state: numpy array containing [x_pos, y_pos, ...]
        terrain: Terrain object containing scene dimensions
        
    Returns:
        bool: True if position is within bounds, False otherwise
    """
    return (0 <= state[0] <= terrain.scene_width and 
            0 <= state[1] <= terrain.scene_height)


def compute_floor_connections(rrt_graph: RRTGraph, terrain: Terrain) -> nx.DiGraph:
    """
    Compute a graph of floor segment connectivity based on RRT graph reachability.
    
    Args:
        rrt_graph: The RRT graph with reachability information
        terrain: The terrain containing floor segments
        
    Returns:
        nx.DiGraph: A directed graph where nodes are floor segment hashes and edges contain minimum costs
    """
    floor_connection_graph = nx.DiGraph()
    
    # Add nodes for each floor segment
    floor_segments = [seg for seg in terrain.segments if seg.type == 'floor']
    segment_hashes = [hash(seg) for seg in floor_segments]
    for seg_hash in segment_hashes:
        floor_connection_graph.add_node(seg_hash)
    
    # Build connectivity between floor segments
    connectivity = {}  # (from_segment_hash, to_segment_hash) -> min_cost
    
    # Find the minimum cost to reach each floor segment from each node
    for node_id, node in rrt_graph.nodes.items():
        # Skip if no reachability information or no segment
        if not node.reachable or node.segment is None:
            continue
            
        # For each reachable segment from this node
        for target_seg_hash, cost in node.reachable.items():
            if target_seg_hash == node.segment:
                continue  # Skip self-connections
                
            key = (node.segment, target_seg_hash)
            if key not in connectivity or cost < connectivity[key]:
                connectivity[key] = cost
    
    # Add edges to the floor connection graph
    for (from_hash, to_hash), cost in connectivity.items():
        floor_connection_graph.add_edge(from_hash, to_hash, weight=cost)
    
    return floor_connection_graph


def sample_connections(terrain: Terrain, num_samples: int = 1000, samples_per_segment: int = 5, 
                       k_neighbors: int = 3, max_edge_duration: float = 0.8,
                       prune_nodes: bool = True, connect_same_segment: bool = True):
    """
    Build a connected graph of motion primitives across all floor segments in the terrain.
    
    Args:
        terrain: The terrain containing segments
        num_samples: Total number of random samples to generate
        samples_per_segment: Number of equally spaced root nodes per segment
        k_neighbors: Number of nearest neighbors to connect to for creating cycles
        max_edge_duration: Maximum duration for edge connections
        prune_nodes: Whether to prune non-essential nodes
        connect_same_segment: Whether to connect nodes on the same floor segment
        
    Returns:
        RRTGraph: A graph representing the motion connections
    """
    # Create and sample the RRT graph
    graph = RRTGraph()
    graph.sample_connections_sync(
        terrain=terrain,
        num_samples=num_samples,
        samples_per_segment=samples_per_segment,
        max_edge_duration=max_edge_duration,
        prune_nodes=prune_nodes,
        connect_same_segment=connect_same_segment
    )
    return graph

def sim_state_to_rrt_state(sim_state: HeroControllerStates) -> np.ndarray:
    return np.array([sim_state.x_pos, sim_state.y_pos, sim_state.y_velocity])

def visualize_graph(graph: RRTGraph, terrain: Terrain, show_edges=True):
    """
    Visualize the RRT graph and terrain segments.
    
    Args:
        graph: The RRT graph to visualize
        terrain: The terrain containing segments to visualize
        show_edges: Whether to show graph edges (default: True)
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot terrain segments
    for segment in terrain.segments:
        color = {
            'floor': 'gray',
            'wall': 'darkgray',
            'ceiling': 'lightgray',
            'hazard': 'red'
        }.get(segment.type, 'black')
        
        ax.plot(
            [segment.x0, segment.x1],
            [segment.y0, segment.y1],
            color=color,
            linewidth=2,
            alpha=0.7,
            label=segment.type
        )
    
    # Get node positions and velocities for coloring
    node_positions = []
    velocities = []
    for node in graph.nodes.values():
        node_positions.append([node.rrt_state[0], node.rrt_state[1]])
        # velocities.append(node.rrt_state[2])
        velocities.append(len(node.reachable))
        if node.segment is None:
            assert len(node.sources) == 1
        # velocities.append(node.air_time)
        # velocities.append((len(node.reachable)) * (1 if node.sim_state.onGround else -1))
    
    node_positions = np.array(node_positions)
    velocities = np.array(velocities)
    
    if len(node_positions) > 0:
        # Plot nodes
        scatter = ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=velocities,
            # cmap='coolwarm',
            s=30,
            alpha=0.6,
            zorder=2
        )
        plt.colorbar(scatter, label='Y Velocity')
        
        # Plot edges if requested
        if show_edges:
            edges = []
            for u, v in graph.graph.edges():
                node_u = graph.nodes[u]
                node_v = graph.nodes[v]
                start_pos = [graph.nodes[u].rrt_state[0], graph.nodes[u].rrt_state[1]]
                end_pos = [graph.nodes[v].rrt_state[0], graph.nodes[v].rrt_state[1]]
                edges.append([start_pos, end_pos])

                u2, v2 = graph.nodes[u].rrt_state[2], graph.nodes[v].rrt_state[2]
                suspect = u2 > 80 and v2 < 80
                keep = u2 < 80 and v2 < 80
                col = 'black'
                if suspect:
                    col = 'orange'
                if keep:
                    col = 'green'
                ax.annotate(
                    "",
                    xy=(end_pos[0], end_pos[1]),
                    xytext=(start_pos[0], start_pos[1]),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2,
                        color=col,
                        alpha=0.3
                    )
                )
                if node_u.segment is not None and node_v.segment is not None and node_u.segment != node_v.segment:
                    assert keep
            
            # if edges:
            #     lc = LineCollection(
            #         edges,
            #         colors='lightblue',
            #         alpha=0.3,
            #         zorder=1
            #     )
            #     ax.add_collection(lc)
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('RRT Graph Visualization')
    
    # Add legend for segment types
    handles = [plt.Line2D([0], [0], color=c, label=t) for t, c in [
        ('Floor', 'gray'),
        ('Wall', 'darkgray'),
        ('Ceiling', 'lightgray'),
        ('Hazard', 'red')
    ]]
    ax.legend(handles=handles, loc='upper right')
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    # Set viewing bounds
    ax.set_xlim(0, terrain.scene_width)
    ax.set_ylim(0, terrain.scene_height)
    
    plt.tight_layout()
    plt.show()

def visualize_floor_connections(graph: RRTGraph, terrain: Terrain):
    """
    Visualize the connections between floor segments based on reachability.
    Edge colors represent the minimum cost to reach from one segment to another.
    
    Args:
        graph: The RRT graph with floor_connection_graph
        terrain: The terrain containing segments to visualize
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Get all floor segments
    floor_segments = [seg for seg in terrain.segments if seg.type == 'floor']
    segment_to_idx = {hash(seg): i for i, seg in enumerate(floor_segments)}
    
    # Plot terrain segments
    segment_centers = {}
    for segment in terrain.segments:
        color = {
            'floor': 'gray',
            'ceiling': 'lightgray',
            'left_wall': 'darkgray',
            'right_wall': 'darkgray'
        }.get(segment.type, 'black')
        
        ax.plot(
            [segment.x_min, segment.x_max],
            [segment.y_min, segment.y_max],
            color=color,
            linewidth=2,
            alpha=0.7
        )
        
        # Calculate and store segment centers for floor segments
        if segment.type == 'floor':
            center_x = (segment.x_min + segment.x_max) / 2
            center_y = segment.y_min  # Using y_min as this is the top of the floor
            segment_hash = hash(segment)
            segment_centers[segment_hash] = (center_x, center_y)
            
            # Label the floor segment
            idx = segment_to_idx[segment_hash]
            ax.text(center_x, center_y + 0.5, f"F{idx}", ha='center', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Draw connections from the floor connection graph
    if hasattr(graph, 'floor_connection_graph'):
        # Get all costs for normalization
        costs = [data['weight'] for _, _, data in graph.floor_connection_graph.edges(data=True)]
        
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)
            
            # Draw each edge
            for from_hash, to_hash, data in graph.floor_connection_graph.edges(data=True):
                if from_hash not in segment_centers or to_hash not in segment_centers:
                    continue
                
                cost = data['weight']
                from_pos = segment_centers[from_hash]
                to_pos = segment_centers[to_hash]
                
                # Normalize cost for color
                norm_cost = (cost - min_cost) / (max_cost - min_cost) if max_cost > min_cost else 0.5
                color = plt.cm.viridis_r(norm_cost)  # viridis_r: yellow=low, purple=high
                
                # Draw the arrow
                ax.annotate(
                    "",
                    xy=(to_pos[0], to_pos[1]),
                    xytext=(from_pos[0], from_pos[1]),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2,
                        color=color,
                        alpha=0.7
                    )
                )
                
                # Add cost label at midpoint
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                ax.text(mid_x, mid_y, f"{cost:.1f}", fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=mcolors.Normalize(vmin=min_cost, vmax=max_cost))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Reachability Cost')
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Floor Segment Reachability')
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    # Set viewing bounds
    ax.set_xlim(0, terrain.scene_width)
    ax.set_ylim(0, terrain.scene_height)
    
    plt.tight_layout()
    plt.show()