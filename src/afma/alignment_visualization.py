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
            if ref[i-1] == seq[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    # Backtrack to find alignment
    aligned_ref = []
    aligned_seq = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == seq[j-1]:
            # Match
            aligned_ref.append(ref[i-1])
            aligned_seq.append(seq[j-1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            aligned_ref.append(ref[i-1])
            aligned_seq.append(seq[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion from reference
            aligned_ref.append(ref[i-1])
            aligned_seq.append('-')
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion to reference
            aligned_ref.append('-')
            aligned_seq.append(seq[j-1])
            j -= 1
    
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
    first_nodes = set()  # Track first nodes that start node should connect to
    last_nodes = set()   # Track last nodes that should connect to end node
    
    for trace_idx, seq in enumerate(actual_sequences):
        ref_aligned, seq_aligned, distance = levenshtein_alignment(reference, seq)
        
        # Track the path through this alignment
        prev_node = None
        first_node_in_trace = None
        last_node_in_trace = None
        
        for align_pos, (ref_val, seq_val) in enumerate(zip(ref_aligned, seq_aligned)):
            current_node = None
            
            if ref_val == seq_val and ref_val != '-':
                # Match: use reference node
                current_node = ref_node(ref_val)
                
            elif ref_val == '-':
                # Insertion: create new sequence node using alignment position
                current_node = seq_node(seq_val, align_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
                
            elif seq_val == '-':
                # Deletion: skip this alignment position, don't create edges
                continue
                
            else:
                # Substitution: create new sequence node using alignment position
                current_node = seq_node(seq_val, align_pos)
                if not graph.has_node(current_node):
                    graph.add_node(current_node)
            
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
        
        # Track first and last nodes for start/end connections
        if first_node_in_trace is not None:
            first_nodes.add(first_node_in_trace)
        if last_node_in_trace is not None:
            last_nodes.add(last_node_in_trace)
    
    # Step 4: Connect start node to first nodes and last nodes to end node
    for first_node in first_nodes:
        edge = (start_node, first_node)
        graph.add_edge(*edge)
        edge_weights[edge] = len(actual_sequences)  # All sequences start here
    
    for last_node in last_nodes:
        edge = (last_node, end_node)
        graph.add_edge(*edge)
        edge_weights[edge] = len(actual_sequences)  # All sequences end here
    
    # Add weights as edge attributes
    for edge, weight in edge_weights.items():
        if graph.has_edge(*edge):
            graph[edge[0]][edge[1]]['weight'] = weight
    
    return graph


def visualize_alignment_graph_matplotlib(graph: nx.DiGraph, reference: List, 
                                       figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Visualize the alignment graph using matplotlib with color-coded nodes and weighted edges.
    
    Args:
        graph: NetworkX DiGraph to visualize
        reference: Reference sequence (used to identify backbone edges)
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph, k=3, iterations=50)
    
    # Separate nodes by type
    start_end_nodes = [node for node in graph.nodes() if node in ['START', 'END']]
    ref_nodes = [node for node in graph.nodes() if node.startswith('ref_')]
    seq_nodes = [node for node in graph.nodes() if node.startswith('seq_')]
    
    # Separate edges by type and get their weights
    ref_backbone_edges = []  # only the original reference backbone
    seq_edges = []           # all other edges (alignment-derived)
    ref_backbone_weights = []
    seq_edge_weights = []
    
    # Define the reference backbone edges explicitly
    ref_values = reference
    
    # Identify first and last reference nodes
    first_ref_node = ref_node(ref_values[0]) if ref_values else None
    last_ref_node = ref_node(ref_values[-1]) if ref_values else None
    
    for i in range(len(ref_values) - 1):
        backbone_edge = (ref_node(ref_values[i]), ref_node(ref_values[i + 1]))
        if backbone_edge in graph.edges():
            ref_backbone_edges.append(backbone_edge)
            weight = graph[backbone_edge[0]][backbone_edge[1]].get('weight', 1)
            ref_backbone_weights.append(max(weight * 2, 2))  # Scale edge weights for larger nodes
    
    # Separate start/end edges from other sequence edges
    for edge in graph.edges():
        if edge not in ref_backbone_edges:
            if edge[0] in ['START', 'END'] or edge[1] in ['START', 'END']:
                # Check if this start/end edge connects to reference start/end nodes
                connects_to_ref_start_end = False
                if edge[0] == 'START' and edge[1] == first_ref_node:
                    connects_to_ref_start_end = True
                elif edge[0] == last_ref_node and edge[1] == 'END':
                    connects_to_ref_start_end = True
                
                weight = graph[edge[0]][edge[1]].get('weight', 1)
                scaled_weight = max(weight * 2, 2)
                
                if connects_to_ref_start_end:
                    # Add to backbone edges (blue)
                    ref_backbone_edges.append(edge)
                    ref_backbone_weights.append(scaled_weight)
                else:
                    # Add to sequence edges (red)
                    seq_edges.append(edge)
                    seq_edge_weights.append(scaled_weight)
            else:
                seq_edges.append(edge)
                weight = graph[edge[0]][edge[1]].get('weight', 1)
                seq_edge_weights.append(max(weight * 2, 2))  # Scale edge weights for larger nodes
    
    # Draw nodes with different colors
    if start_end_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=start_end_nodes, node_color="grey", 
                              node_size=6000, alpha=0.8, ax=ax)
    if ref_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=ref_nodes, node_color="blue", 
                              node_size=6000, alpha=0.8, ax=ax)
    if seq_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=seq_nodes, node_color="red", 
                              node_size=6000, alpha=0.8, ax=ax)
    
    # Draw edges with different colors and weights
    if ref_backbone_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=ref_backbone_edges, edge_color="blue", 
                              arrows=True, alpha=0.6, arrowsize=40, width=ref_backbone_weights, 
                              min_source_margin=30, min_target_margin=30, ax=ax)
    if seq_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=seq_edges, edge_color="red", 
                              arrows=True, alpha=0.6, arrowsize=40, width=seq_edge_weights,
                              min_source_margin=30, min_target_margin=30, ax=ax)
    
    # Create custom labels showing only the values
    custom_labels = {}
    for node in graph.nodes():
        if node in ['START', 'END']:
            # Keep START and END as is
            custom_labels[node] = node
        elif node.startswith('ref_'):
            # Extract value from ref_tool_name
            tool_name = node[4:]  # Remove 'ref_' prefix
        elif node.startswith('seq_'):
            # Extract value from seq_tool_name_pos{position}
            value_part = node[4:]  # Remove 'seq_' prefix
            if '_pos' in value_part:
                # Split on '_pos' to separate tool name from position
                tool_name = value_part.split('_pos')[0]
            else:
                tool_name = value_part
        else:
            tool_name = node  # Fallback
        
        # Format tool name: first 8 chars on first line, next 8 on second line
        # (Skip this formatting for START/END nodes)
        if node not in ['START', 'END']:
            if len(tool_name) <= 8:
                custom_labels[node] = tool_name
            else:
                first_line = tool_name[:8]
                second_line = tool_name[8:16]  # Take next 8 chars, strip the rest
                custom_labels[node] = f"{first_line}\n{second_line}"
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, labels=custom_labels, font_size=14, 
                           font_color="white", font_weight="bold", ax=ax)
    
    ax.set_title("Sequence Alignment Graph", fontsize=18)
    ax.axis("off")
    plt.tight_layout()
    
    return fig


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
    
    # Extract actual sequences from alignments
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
    
    # Create alignment graph
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