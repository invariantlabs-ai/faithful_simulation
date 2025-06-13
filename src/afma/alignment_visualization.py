"""
Alignment Visualization Module

This module provides tools for creating and visualizing sequence alignment graphs.
It supports:
- Creating directed graphs representing sequence alignments with Levenshtein distance
- Adding start and end nodes to visualize sequence boundaries
- Color-coded visualization showing reference backbone vs sequence variations
- Export to both matplotlib figures and base64 encoded images for web apps
"""

import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any, Tuple, Optional


def ref_node(value) -> str:
    """Create reference node name."""
    return f"ref_{value}"


def seq_node(value, alignment_position) -> str:
    """Create sequence node name based on value and alignment position."""
    return f"seq_{value}_pos{alignment_position}"


def levenshtein_alignment(ref: List, seq: List) -> Tuple[List, List, int]:
    """
    Compute Levenshtein alignment between reference and sequence.
    Returns the aligned sequences showing insertions, deletions, and substitutions.
    
    Args:
        ref: Reference sequence
        seq: Sequence to align
    
    Returns:
        Tuple of (aligned_ref, aligned_seq, distance)
    """
    m, n = len(ref), len(seq)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            position_bias = 0.0001 * j
            if ref[i-1] == seq[j-1]:
                dp[i][j] = dp[i-1][j-1] + position_bias  # Match gets more expensive later
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],                      # Deletion
                    dp[i][j-1],     # Insertion (cheaper later)
                    dp[i-1][j-1]                     # Substitution
                )
    
    # Backtrack to find alignment
    aligned_ref = []
    aligned_seq = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == seq[j-1]:
            # Match - check with position bias
            position_bias = 0.0001 * j
            if abs(dp[i][j] - (dp[i-1][j-1] + position_bias)) < 1e-10:
                aligned_ref.append(ref[i-1])
                aligned_seq.append(seq[j-1])
                i -= 1
                j -= 1
                continue
        
        # Check all possible moves and pick the one that led to current cost
        candidates = []
        
        # Substitution
        if i > 0 and j > 0:
            if abs(dp[i][j] - (dp[i-1][j-1] + 1)) < 1e-10:
                candidates.append(('substitute', i-1, j-1))
        
        # Deletion
        if i > 0:
            if abs(dp[i][j] - (dp[i-1][j] + 1)) < 1e-10:
                candidates.append(('delete', i-1, j))
        
        # Insertion - check with insertion bias
        if j > 0:
            if abs(dp[i][j] - (dp[i][j-1] + 1)) < 1e-10:
                candidates.append(('insert', i, j-1))
        
        # Choose the first valid candidate (preference order matters)
        if candidates:
            move, new_i, new_j = candidates[0]
            if move == 'substitute':
                aligned_ref.append(ref[i-1])
                aligned_seq.append(seq[j-1])
            elif move == 'delete':
                aligned_ref.append(ref[i-1])
                aligned_seq.append('-')
            elif move == 'insert':
                aligned_ref.append('-')
                aligned_seq.append(seq[j-1])
            i, j = new_i, new_j
        else:
            # Fallback - shouldn't happen with correct implementation
            break
    
    # Reverse since we built backwards
    aligned_ref.reverse()
    aligned_seq.reverse()
    
    return aligned_ref, aligned_seq, dp[m][n]


def create_alignment_graph(reference: List, actual_sequences: List[List]) -> nx.DiGraph:
    """
    Create a directed graph representing sequence alignments with start and end nodes.
    
    The algorithm works as follows:
    1. Create start and end nodes
    2. Create reference backbone nodes (ref_tool1, ref_tool2, etc.)
    3. For each sequence alignment:
       - Follow the alignment path
       - Create new nodes for substitutions and insertions (seq_tool_pos{position})
       - Merge nodes that have the same value at the same alignment position
       - Connect nodes to show the alignment flow
       - Track edge usage counts (weights)
    4. Connect start node to first nodes and last nodes to end node
    5. Result: A graph showing how sequences branch from and merge back to reference
    
    Args:
        reference: Reference sequence
        actual_sequences: List of actual sequences to align
    
    Returns:
        NetworkX DiGraph representing the alignments with edge weights
    """
    graph = nx.DiGraph()
    edge_weights = {}  # Track how many traces use each edge
    
    # Step 1: Create start and end nodes
    start_node = "START"
    end_node = "END"
    graph.add_node(start_node)
    graph.add_node(end_node)
    
    # Step 2: Create reference backbone
    ref_nodes = [ref_node(val) for val in reference]
    for node in ref_nodes:
        graph.add_node(node)
    
    # Connect reference nodes sequentially
    for i in range(len(ref_nodes) - 1):
        edge = (ref_nodes[i], ref_nodes[i + 1])
        graph.add_edge(*edge)
        edge_weights[edge] = 0  # Initialize with 0, will be incremented by traces that use it
    
    # Step 3: Process each actual sequence
    first_node_counts = {}  # Track how many sequences start at each first node
    last_node_counts = {}   # Track how many sequences end at each last node
    
    for trace_idx, seq in enumerate(actual_sequences):
        ref_aligned, seq_aligned, distance = levenshtein_alignment(reference, seq)
        
        # Track the path through this alignment
        prev_node = None
        first_node_in_trace = None
        last_node_in_trace = None
        
        # Track reference position separately from alignment position
        ref_pos = 0
        
        for align_pos, (ref_val, seq_val) in enumerate(zip(ref_aligned, seq_aligned)):
            current_node = None
            
            if ref_val == seq_val and ref_val != '-':
                # Match: use reference node
                current_node = ref_node(ref_val)
                ref_pos += 1
                
            elif ref_val == '-':
                # Insertion: create new sequence node using reference position
                # For insertions, we use the current reference position (where it would insert)
                current_node = seq_node(seq_val, ref_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
                # Don't increment ref_pos for insertions
                
            elif seq_val == '-':
                # Deletion: skip this alignment position, don't create edges
                ref_pos += 1
                continue
                
            else:
                # Substitution: create new sequence node using reference position
                current_node = seq_node(seq_val, ref_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
                ref_pos += 1
            
            # Track first and last nodes in this trace
            if current_node is not None:
                if first_node_in_trace is None:
                    first_node_in_trace = current_node
                last_node_in_trace = current_node
            
            # Add edge from previous node if exists and track weight
            if prev_node is not None and current_node is not None:
                edge = (prev_node, current_node)
                if not graph.has_edge(*edge):
                    graph.add_edge(*edge)
                    edge_weights[edge] = 0
                edge_weights[edge] += 1
            
            prev_node = current_node
        
        # Track first and last nodes for start/end connections with counts
        if first_node_in_trace is not None:
            first_node_counts[first_node_in_trace] = first_node_counts.get(first_node_in_trace, 0) + 1
        if last_node_in_trace is not None:
            last_node_counts[last_node_in_trace] = last_node_counts.get(last_node_in_trace, 0) + 1
    
    # Step 4: Connect start node to first nodes and last nodes to end node with correct weights
    for first_node, count in first_node_counts.items():
        edge = (start_node, first_node)
        graph.add_edge(*edge)
        edge_weights[edge] = count
    
    for last_node, count in last_node_counts.items():
        edge = (last_node, end_node)
        graph.add_edge(*edge)
        edge_weights[edge] = count
    
    # Add weights as edge attributes
    for edge, weight in edge_weights.items():
        if graph.has_edge(*edge):
            graph[edge[0]][edge[1]]['weight'] = weight
    
    return graph


def visualize_alignment_graph_matplotlib(graph: nx.DiGraph, reference: List, 
                                       figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Visualize the alignment graph using matplotlib with color-coded nodes and weighted edges.
    Enhanced to support merged nodes with purple coloring based on reference ratio.
    
    Args:
        graph: NetworkX DiGraph to visualize
        reference: Reference sequence (used to identify backbone edges)
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        pos = nx.planar_layout(graph)
    except:
        pos = nx.spring_layout(graph, k=3, iterations=50)
    
    # Separate nodes by type using node metadata
    start_end_nodes = []
    ref_nodes = []
    var_nodes = []
    merged_nodes = []
    merged_colors = []
    
    for node in graph.nodes():
        node_type = graph.nodes[node].get('node_type', 'unknown')
        
        if node_type == 'boundary':
            start_end_nodes.append(node)
        elif node_type == 'reference':
            ref_nodes.append(node)
        elif node_type == 'variation':
            var_nodes.append(node)
        elif node_type == 'merged':
            merged_nodes.append(node)
            # Calculate purple color based on reference ratio
            ref_ratio = graph.nodes[node].get('ref_ratio', 0.5)
            # Create purple color: blend between red (1,0,0) and blue (0,0,1)
            # ref_ratio = 1.0 -> pure blue, ref_ratio = 0.0 -> pure red
            red_component = 1.0 - ref_ratio
            blue_component = ref_ratio
            purple_color = (red_component, 0.0, blue_component)
            merged_colors.append(purple_color)
        else:
            # Fallback for unknown node types
            if node.startswith('ref_'):
                ref_nodes.append(node)
            elif node.startswith('seq_'):
                var_nodes.append(node)
            elif node.startswith('merged_'):
                merged_nodes.append(node)
                merged_colors.append((0.5, 0.0, 0.5))  # Default purple
            else:
                start_end_nodes.append(node)
    
    # Separate edges by type using edge metadata and node types
    ref_edges = []
    var_edges = []
    merged_edges = []  # Edges involving merged nodes
    ref_edge_weights = []
    var_edge_weights = []
    merged_edge_weights = []
    
    for edge in graph.edges():
        edge_type = graph[edge[0]][edge[1]].get('edge_type', 'unknown')
        edge_weight = max(graph[edge[0]][edge[1]].get('weight', 1) * 2, 2)
        
        source_node_type = graph.nodes[edge[0]].get('node_type', 'unknown')
        target_node_type = graph.nodes[edge[1]].get('node_type', 'unknown')
        
        # Classify edges based on node types and edge metadata
        if source_node_type == 'merged' or target_node_type == 'merged':
            # Edge involving merged nodes - use purple/mixed coloring
            merged_edges.append(edge)
            merged_edge_weights.append(edge_weight)
        elif edge_type == 'reference' or (source_node_type == 'reference' and target_node_type == 'reference'):
            ref_edges.append(edge)
            ref_edge_weights.append(edge_weight)
        else:
            var_edges.append(edge)
            var_edge_weights.append(edge_weight)
    
    # Draw nodes with different colors
    if start_end_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=start_end_nodes, node_color="grey", 
                              node_size=6000, alpha=0.8, ax=ax)
    if ref_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=ref_nodes, node_color="blue", 
                              node_size=6000, alpha=0.8, ax=ax)
    if var_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=var_nodes, node_color="red", 
                              node_size=6000, alpha=0.8, ax=ax)
    if merged_nodes and merged_colors:
        nx.draw_networkx_nodes(graph, pos, nodelist=merged_nodes, node_color=merged_colors, 
                              node_size=6000, alpha=0.8, ax=ax)
    
    # Draw edges with different colors and weights
    if ref_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=ref_edges, edge_color="blue", 
                              arrows=True, alpha=0.6, arrowsize=40, width=ref_edge_weights, 
                              min_source_margin=30, min_target_margin=30, ax=ax)
    if var_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=var_edges, edge_color="red", 
                              arrows=True, alpha=0.6, arrowsize=40, width=var_edge_weights,
                              min_source_margin=30, min_target_margin=30, ax=ax)
    if merged_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=merged_edges, edge_color="purple", 
                              arrows=True, alpha=0.6, arrowsize=40, width=merged_edge_weights,
                              min_source_margin=30, min_target_margin=30, ax=ax)
    
    # Create custom labels showing only the tool names
    custom_labels = {}
    for node in graph.nodes():
        if node in ['START', 'END']:
            custom_labels[node] = node
        else:
            # Extract tool name from node metadata or node name
            tool_name = graph.nodes[node].get('tool_name', None)
            if tool_name is None:
                # Fallback to parsing node name
                if node.startswith('ref_'):
                    tool_name = node[4:]
                elif node.startswith('seq_'):
                    value_part = node[4:]
                    if '_pos' in value_part:
                        tool_name = value_part.split('_pos')[0]
                    else:
                        tool_name = value_part
                elif node.startswith('merged_'):
                    value_part = node[7:]  # Remove 'merged_' prefix
                    if '_pos' in value_part:
                        tool_name = value_part.split('_pos')[0]
                    elif '_ref' in value_part:
                        tool_name = value_part.replace('_ref', '')
                    else:
                        tool_name = value_part
                else:
                    tool_name = node
            
            # Format tool name: first 8 chars on first line, next 8 on second line
            if len(tool_name) <= 8:
                custom_labels[node] = tool_name
            else:
                first_line = tool_name[:8]
                second_line = tool_name[8:16]
                custom_labels[node] = f"{first_line}\n{second_line}"
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, labels=custom_labels, font_size=14, 
                           font_color="white", font_weight="bold", ax=ax)
    
    ax.set_title("Sequence Alignment Graph", fontsize=18)
    ax.axis("off")
    plt.tight_layout()
    
    return fig


def create_alignment_graph_from_prealigned(reference: List, aligned_sequences: List[List[Tuple]]) -> nx.DiGraph:
    """
    Create a directed graph from pre-aligned sequences (where alignment is already computed).
    
    Args:
        reference: Reference sequence
        aligned_sequences: List of aligned sequences, where each aligned sequence is a list of 
                          (ref_element, seq_element) tuples. None values represent gaps.
    
    Returns:
        NetworkX DiGraph representing the alignments with edge weights
    """
    graph = nx.DiGraph()
    edge_weights = {}  # Track how many traces use each edge
    
    # Step 1: Create start and end nodes
    start_node = "START"
    end_node = "END"
    graph.add_node(start_node)
    graph.add_node(end_node)
    
    # Step 2: Create reference backbone nodes
    ref_nodes = [ref_node(val) for val in reference]
    for node in ref_nodes:
        graph.add_node(node)
    
    # Connect reference nodes sequentially
    for i in range(len(ref_nodes) - 1):
        edge = (ref_nodes[i], ref_nodes[i + 1])
        graph.add_edge(*edge)
        edge_weights[edge] = 0  # Initialize with 0, will be incremented by traces that use it
    
    # Step 3: Process each pre-aligned sequence
    first_node_counts = {}  # Track how many sequences start at each first node
    last_node_counts = {}   # Track how many sequences end at each last node
    
    for trace_idx, aligned_seq in enumerate(aligned_sequences):
        # Track the path through this alignment
        prev_node = None
        first_node_in_trace = None
        last_node_in_trace = None
        
        # Track reference position separately from alignment position
        ref_pos = 0
        
        for align_pos, (ref_val, seq_val) in enumerate(aligned_seq):
            current_node = None
            
            if ref_val == seq_val and ref_val is not None:
                # Match: use reference node
                current_node = ref_node(ref_val)
                ref_pos += 1
                
            elif ref_val is None:
                # Insertion: create new sequence node using reference position
                current_node = seq_node(seq_val, ref_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
                # Don't increment ref_pos for insertions
                
            elif seq_val is None:
                # Deletion: skip this alignment position, don't create edges
                ref_pos += 1
                continue
                
            else:
                # Substitution: create new sequence node using reference position
                current_node = seq_node(seq_val, ref_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
                ref_pos += 1
            
            # Track first and last nodes in this trace
            if current_node is not None:
                if first_node_in_trace is None:
                    first_node_in_trace = current_node
                last_node_in_trace = current_node
            
            # Add edge from previous node if exists and track weight
            if prev_node is not None and current_node is not None:
                edge = (prev_node, current_node)
                if not graph.has_edge(*edge):
                    graph.add_edge(*edge)
                    edge_weights[edge] = 0
                edge_weights[edge] += 1
            
            prev_node = current_node
        
        # Track first and last nodes for start/end connections with counts
        if first_node_in_trace is not None:
            first_node_counts[first_node_in_trace] = first_node_counts.get(first_node_in_trace, 0) + 1
        if last_node_in_trace is not None:
            last_node_counts[last_node_in_trace] = last_node_counts.get(last_node_in_trace, 0) + 1
    
    # Step 4: Connect start node to first nodes and last nodes to end node with correct weights
    for first_node, count in first_node_counts.items():
        edge = (start_node, first_node)
        graph.add_edge(*edge)
        edge_weights[edge] = count
    
    for last_node, count in last_node_counts.items():
        edge = (last_node, end_node)
        graph.add_edge(*edge)
        edge_weights[edge] = count
    
    # Add weights as edge attributes
    for edge, weight in edge_weights.items():
        if graph.has_edge(*edge):
            graph[edge[0]][edge[1]]['weight'] = weight
    
    return graph


def create_trace_alignment_graph_base64(alignment_data: Dict[str, Any]) -> str:
    """
    Create an alignment graph visualization for web apps and return it as base64 encoded image.
    
    Args:
        alignment_data: Dictionary containing alignment data with 'alignments' and 'reference_sequence'
    
    Returns:
        Base64 encoded PNG image string
    """
    alignments = alignment_data["alignments"]
    reference_sequence = alignment_data["reference_sequence"]
    
    if not alignments or len(alignments) <= 1:
        # Create a simple message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough instances for visualization\n(Need at least 2 sequences)", 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        
        # Convert to base64 for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_base64
    
    # Check if we have pre-aligned data (alignment field) or need to extract from operations
    if "alignment" in alignments[0]:
        # Use pre-aligned data directly
        aligned_sequences = []
        for alignment_entry in alignments:
            aligned_seq = alignment_entry["alignment"]
            aligned_sequences.append(aligned_seq)
        
        # Create alignment graph from pre-aligned data
        graph = create_alignment_graph_from_prealigned(reference_sequence, aligned_sequences)
    else:
        # Extract actual sequences from operations and compute alignment
        actual_sequences = []
        for alignment in alignments:
            actual_sequence = []
            for op_type, tool1, tool2 in alignment["operations"]:
                if op_type == "match":
                    actual_sequence.append(tool2)  # tool2 is the actual tool used
                elif op_type == "substitute":
                    actual_sequence.append(tool2)  # substituted tool
                elif op_type == "insert":
                    actual_sequence.append(tool2)  # inserted tool
                # skip delete operations (no tool added)
            actual_sequences.append(actual_sequence)
        
        # Create alignment graph (will compute Levenshtein alignment internally)
        graph = create_alignment_graph(reference_sequence, actual_sequences)
    
    # Visualize using matplotlib
    fig = visualize_alignment_graph_matplotlib(graph, reference_sequence, figsize=(16, 12))
    
    # Convert to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_base64


def print_alignment(ref_aligned: List, seq_aligned: List, distance: int, seq_index: int):
    """Print a formatted alignment."""
    print(f"\nAlignment {seq_index + 1}:")
    print(f"Reference: {' '.join(str(x) if x != '-' else '-' for x in ref_aligned)}")
    print(f"Actual:    {' '.join(str(x) if x != '-' else '-' for x in seq_aligned)}")
    print(f"Distance:  {distance}")
    
    # Show operations
    operations = []
    for i, (r, s) in enumerate(zip(ref_aligned, seq_aligned)):
        if r == s:
            operations.append('M')  # Match
        elif r == '-':
            operations.append('I')  # Insertion
        elif s == '-':
            operations.append('D')  # Deletion
        else:
            operations.append('S')  # Substitution
    print(f"Operations: {' '.join(operations)}")
    print(f"Legend: M=Match, S=Substitution, I=Insertion, D=Deletion")


def print_graph_info(graph: nx.DiGraph):
    """Print information about the graph structure and edge weights."""
    print(f"\nGraph nodes: {sorted(list(graph.nodes()))}")
    print(f"Graph edges: {sorted(list(graph.edges()))}")
    
    # Print edge weights
    print(f"\nEdge weights:")
    for edge in sorted(graph.edges()):
        weight = graph[edge[0]][edge[1]].get('weight', 0)
        print(f"  {edge[0]} -> {edge[1]}: {weight}")


def process_alignments(reference: List, actual_sequences: List[List], verbose: bool = True):
    """
    Process and print all sequence alignments.
    
    Args:
        reference: Reference sequence
        actual_sequences: List of actual sequences to align
        verbose: Whether to print alignment details
    """
    if verbose:
        print("Computing alignments:")
        for i, seq in enumerate(actual_sequences):
            ref_aligned, seq_aligned, distance = levenshtein_alignment(reference, seq)
            print_alignment(ref_aligned, seq_aligned, distance, i)


def save_alignment_graph(graph: nx.DiGraph, reference: List, 
                        output_filename: str = "alignment_graph.png", 
                        figsize: Tuple[int, int] = (16, 12)):
    """
    Save alignment graph visualization to file.
    
    Args:
        graph: NetworkX DiGraph to visualize
        reference: Reference sequence
        output_filename: Output filename for the graph image
        figsize: Figure size tuple
    """
    fig = visualize_alignment_graph_matplotlib(graph, reference, figsize)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph saved as '{output_filename}'")


def merge_alignment_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    """
    Merge multiple alignment graphs into a single graph using BFS traversal.
    Enhanced to merge nodes of the same tool that appear as both reference and variation.
    
    The merging strategy:
    1. Start BFS from START node in each graph
    2. Identify nodes representing the same tool at the same position
    3. Merge ref_tool and seq_tool_pos{X} nodes into merged_tool_pos{X} nodes
    4. Sum edge weights when merging equivalent edges
    5. Track reference vs variation usage ratios for purple coloring
    
    Args:
        graphs: List of NetworkX DiGraphs to merge
    
    Returns:
        Merged NetworkX DiGraph with node type information
    """
    if not graphs:
        return nx.DiGraph()
    
    if len(graphs) == 1:
        return graphs[0].copy()
    
    merged_graph = nx.DiGraph()
    merged_edge_weights = {}
    
    # Track node usage statistics for merging decisions
    node_usage = {}  # {(tool, position): {'ref_count': int, 'var_count': int, 'total_weight': int}}
    
    # Step 1: Analyze all nodes across graphs to identify merge candidates
    for graph in graphs:
        for node in graph.nodes():
            if node in ['START', 'END']:
                continue
                
            tool_name = None
            position = None
            node_type = None
            
            if node.startswith('ref_'):
                tool_name = node[4:]  # Remove 'ref_' prefix
                # For reference nodes, we need to determine their position in the reference sequence
                # We'll use a special position marker for reference nodes
                position = 'ref'
                node_type = 'reference'
            elif node.startswith('seq_'):
                # Extract tool name and position from seq_tool_pos{position}
                value_part = node[4:]  # Remove 'seq_' prefix
                if '_pos' in value_part:
                    tool_name, pos_part = value_part.split('_pos', 1)
                    try:
                        position = int(pos_part)
                        node_type = 'variation'
                    except ValueError:
                        position = pos_part
                        node_type = 'variation'
                else:
                    tool_name = value_part
                    position = 'unknown'
                    node_type = 'variation'
            
            if tool_name is not None and position is not None:
                key = (tool_name, position)
                if key not in node_usage:
                    node_usage[key] = {'ref_count': 0, 'var_count': 0, 'total_weight': 0}
                
                if node_type == 'reference':
                    node_usage[key]['ref_count'] += 1
                elif node_type == 'variation':
                    node_usage[key]['var_count'] += 1
    
    # Step 2: Create merged nodes with type information
    node_mapping = {}  # Maps original nodes to merged node names
    
    for (tool, position), usage in node_usage.items():
        ref_count = usage['ref_count']
        var_count = usage['var_count']
        total_count = ref_count + var_count
        
        # Determine merged node name and type
        if ref_count > 0 and var_count > 0:
            # Both reference and variation exist - create purple merged node
            if position == 'ref':
                merged_node = f"merged_{tool}_ref"
            else:
                merged_node = f"merged_{tool}_pos{position}"
            node_type = 'merged'
            ref_ratio = ref_count / total_count
        elif ref_count > 0:
            # Only reference - keep as blue
            merged_node = f"ref_{tool}"
            node_type = 'reference'
            ref_ratio = 1.0
        else:
            # Only variation - keep as red
            if position == 'ref':
                merged_node = f"seq_{tool}_ref"
            else:
                merged_node = f"seq_{tool}_pos{position}"
            node_type = 'variation'
            ref_ratio = 0.0
        
        # Add merged node to graph with metadata
        merged_graph.add_node(merged_node, node_type=node_type, ref_ratio=ref_ratio, 
                             tool_name=tool, position=position)
        
        # Create mappings from original nodes to merged nodes
        if ref_count > 0:
            orig_ref_node = f"ref_{tool}"
            node_mapping[orig_ref_node] = merged_node
        
        if var_count > 0:
            if position == 'ref':
                orig_var_node = f"seq_{tool}_ref"
            else:
                orig_var_node = f"seq_{tool}_pos{position}"
            node_mapping[orig_var_node] = merged_node
    
    # Add START and END nodes
    merged_graph.add_node("START", node_type="boundary")
    merged_graph.add_node("END", node_type="boundary")
    node_mapping["START"] = "START"
    node_mapping["END"] = "END"
    
    # Step 3: Merge edges using BFS-like approach with node mapping
    for graph in graphs:
        if not graph.nodes():
            continue
            
        # Use BFS to traverse this graph and merge edges
        if "START" in graph.nodes():
            visited = set()
            queue = ["START"]
            
            while queue:
                current_node = queue.pop(0)
                
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                # Get merged node name for current node
                merged_current = node_mapping.get(current_node, current_node)
                
                # Process all outgoing edges from current node
                for successor in graph.successors(current_node):
                    merged_successor = node_mapping.get(successor, successor)
                    edge = (merged_current, merged_successor)
                    
                    # Get edge weight from this graph
                    edge_weight = graph[current_node][successor].get('weight', 1)
                    
                    # Determine edge type based on source and target node types
                    source_type = merged_graph.nodes[merged_current].get('node_type', 'unknown')
                    target_type = merged_graph.nodes[merged_successor].get('node_type', 'unknown')
                    
                    if source_type == 'reference' and target_type == 'reference':
                        edge_type = 'reference'
                    elif source_type in ['boundary'] or target_type in ['boundary']:
                        # Edge connecting to START/END
                        if source_type == 'reference' or target_type == 'reference':
                            edge_type = 'reference'
                        else:
                            edge_type = 'variation'
                    else:
                        edge_type = 'variation'
                    
                    # Add edge to merged graph if not exists
                    if not merged_graph.has_edge(*edge):
                        merged_graph.add_edge(*edge, edge_type=edge_type)
                        merged_edge_weights[edge] = 0
                    
                    # Accumulate edge weight
                    merged_edge_weights[edge] += edge_weight
                    
                    # Add successor to queue for further processing
                    if successor not in visited:
                        queue.append(successor)
        else:
            # Fallback: process all edges if no START node
            for edge in graph.edges():
                orig_source, orig_target = edge
                merged_source = node_mapping.get(orig_source, orig_source)
                merged_target = node_mapping.get(orig_target, orig_target)
                merged_edge = (merged_source, merged_target)
                
                edge_weight = graph[orig_source][orig_target].get('weight', 1)
                
                if not merged_graph.has_edge(*merged_edge):
                    merged_graph.add_edge(*merged_edge, edge_type='variation')
                    merged_edge_weights[merged_edge] = 0
                
                merged_edge_weights[merged_edge] += edge_weight
    
    # Step 4: Add accumulated weights as edge attributes
    for edge, total_weight in merged_edge_weights.items():
        if merged_graph.has_edge(*edge):
            merged_graph[edge[0]][edge[1]]['weight'] = total_weight
    
    return merged_graph


def filter_alignment_graph(graph: nx.DiGraph, min_edge_weight: int = 1, min_node_degree: int = 1) -> nx.DiGraph:
    """
    Filter alignment graph by removing low-weight edges and low-degree nodes.
    
    Args:
        graph: NetworkX DiGraph to filter
        min_edge_weight: Minimum edge weight to keep (edges below this are removed)
        min_node_degree: Minimum node degree to keep (nodes below this are removed)
    
    Returns:
        Filtered NetworkX DiGraph
    """
    if not graph.nodes():
        return graph.copy()
    
    filtered_graph = graph.copy()
    
    # Step 1: Filter edges by weight
    edges_to_remove = []
    for edge in filtered_graph.edges():
        weight = filtered_graph[edge[0]][edge[1]].get('weight', 1)
        if weight < min_edge_weight:
            edges_to_remove.append(edge)
    
    filtered_graph.remove_edges_from(edges_to_remove)
    
    # Step 2: Filter nodes by degree (but preserve START and END)
    nodes_to_remove = []
    for node in filtered_graph.nodes():
        if node in ['START', 'END']:
            continue  # Always keep START and END nodes
        
        degree = filtered_graph.degree(node)  # Total degree (in + out)
        if degree < min_node_degree:
            nodes_to_remove.append(node)
    
    filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # Step 3: Clean up orphaned nodes (nodes with no edges after filtering)
    # but preserve START and END
    orphaned_nodes = []
    for node in filtered_graph.nodes():
        if node in ['START', 'END']:
            continue
        if filtered_graph.degree(node) == 0:
            orphaned_nodes.append(node)
    
    filtered_graph.remove_nodes_from(orphaned_nodes)
    
    return filtered_graph


def get_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Get statistics about the graph for display in UI.
    
    Args:
        graph: NetworkX DiGraph to analyze
    
    Returns:
        Dictionary with graph statistics
    """
    if not graph.nodes():
        return {
            'node_count': 0,
            'edge_count': 0,
            'max_edge_weight': 0,
            'min_edge_weight': 0,
            'avg_edge_weight': 0,
            'max_node_degree': 0,
            'min_node_degree': 0,
            'avg_node_degree': 0
        }
    
    # Edge weights
    edge_weights = []
    for edge in graph.edges():
        weight = graph[edge[0]][edge[1]].get('weight', 1)
        edge_weights.append(weight)
    
    # Node degrees
    node_degrees = []
    for node in graph.nodes():
        if node not in ['START', 'END']:  # Exclude START/END from degree stats
            degree = graph.degree(node)
            node_degrees.append(degree)
    
    return {
        'node_count': len(graph.nodes()),
        'edge_count': len(graph.edges()),
        'max_edge_weight': max(edge_weights) if edge_weights else 0,
        'min_edge_weight': min(edge_weights) if edge_weights else 0,
        'avg_edge_weight': sum(edge_weights) / len(edge_weights) if edge_weights else 0,
        'max_node_degree': max(node_degrees) if node_degrees else 0,
        'min_node_degree': min(node_degrees) if node_degrees else 0,
        'avg_node_degree': sum(node_degrees) / len(node_degrees) if node_degrees else 0
    }


def create_merged_trace_alignment_graph_base64(all_trace_alignments: Dict[str, Any], 
                                             figsize: Tuple[int, int] = (20, 16),
                                             min_edge_weight: int = 1,
                                             min_node_degree: int = 1) -> str:
    """
    Create a merged alignment graph from all trace alignments and return as base64 encoded image.
    
    Args:
        all_trace_alignments: Dictionary of all trace alignment data keyed by trace_set_id
        figsize: Figure size tuple
        min_edge_weight: Minimum edge weight to keep (edges below this are removed)
        min_node_degree: Minimum node degree to keep (nodes below this are removed)
    
    Returns:
        Base64 encoded PNG image string
    """
    if not all_trace_alignments:
        # Create a simple message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No trace alignment data available", 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_base64
    
    # Step 1: Create individual graphs for each trace set
    individual_graphs = []
    all_reference_sequences = []
    
    for trace_set_id, alignment_data in all_trace_alignments.items():
        alignments = alignment_data.get("alignments", [])
        reference_sequence = alignment_data.get("reference_sequence", [])
        
        if not alignments or len(alignments) <= 1:
            continue  # Skip trace sets with insufficient data
        
        try:
            # Check if we have pre-aligned data or need to extract from operations
            if alignments and "alignment" in alignments[0]:
                # Use pre-aligned data directly
                aligned_sequences = []
                for alignment_entry in alignments:
                    aligned_seq = alignment_entry["alignment"]
                    aligned_sequences.append(aligned_seq)
                
                # Create alignment graph from pre-aligned data
                graph = create_alignment_graph_from_prealigned(reference_sequence, aligned_sequences)
            else:
                # Extract actual sequences from operations
                actual_sequences = []
                for alignment in alignments:
                    actual_sequence = []
                    for op_type, tool1, tool2 in alignment["operations"]:
                        if op_type == "match":
                            actual_sequence.append(tool2)
                        elif op_type == "substitute":
                            actual_sequence.append(tool2)
                        elif op_type == "insert":
                            actual_sequence.append(tool2)
                        # skip delete operations
                    actual_sequences.append(actual_sequence)
                
                # Create alignment graph
                graph = create_alignment_graph(reference_sequence, actual_sequences)
            
            individual_graphs.append(graph)
            all_reference_sequences.append(reference_sequence)
            
        except Exception as e:
            print(f"Error processing trace set {trace_set_id}: {e}")
            continue
    
    if not individual_graphs:
        # Create error message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid trace alignments found for merging", 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_base64
    
    # Step 2: Merge all individual graphs
    merged_graph = merge_alignment_graphs(individual_graphs)
    
    # Step 3: Create a combined reference sequence for visualization
    # Use the most common tools across all reference sequences
    from collections import Counter
    all_tools = []
    for ref_seq in all_reference_sequences:
        all_tools.extend(ref_seq)
    
    # Use all unique tools as combined reference (preserving some order)
    combined_reference = []
    seen_tools = set()
    for ref_seq in all_reference_sequences:
        for tool in ref_seq:
            if tool not in seen_tools:
                combined_reference.append(tool)
                seen_tools.add(tool)
    
    # Step 4: Filter the merged graph
    filtered_graph = filter_alignment_graph(merged_graph, min_edge_weight, min_node_degree)
    
    # Step 5: Visualize the filtered graph
    fig = visualize_alignment_graph_matplotlib(filtered_graph, combined_reference, figsize)
    
    # Add title indicating this is a merged view
    fig.suptitle("Merged Alignment Graph - All Trace Sets Combined", fontsize=20, y=0.95)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_base64 