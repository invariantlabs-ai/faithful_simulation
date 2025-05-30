#!/usr/bin/env python3
"""
Conversation Viewer - A simple web app to explore conversation data.
Run with: streamlit run conversation_viewer.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any


@st.cache_data
def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Load conversations from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def similarity_metric(seq1: List[str], seq2: List[str]) -> float:
    """Calculate normalized Levenshtein Distance similarity between two sequences.
    
    Returns a similarity score between 0 and 1, where:
    - 1.0 means sequences are identical
    - 0.0 means sequences are completely different
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    # Calculate Levenshtein distance using dynamic programming
    len1, len2 = len(seq1), len(seq2)
    
    # Create a matrix to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    # Get the Levenshtein distance
    distance = dp[len1][len2]
    
    # Normalize to similarity score (0-1 range)
    max_len = max(len1, len2)
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    return similarity


def get_similarity_score(conv: Dict[str, Any]) -> tuple[float, str]:
    """Get similarity score from conversation data, preferring weighted if available."""
    # Check for weighted Levenshtein metric score first
    if "weighted_levenshtein_score" in conv:
        return conv["weighted_levenshtein_score"], "Weighted Levenshtein"
    # Check for tool subsequence metric score
    elif "tool_subsequence_score" in conv:
        return conv["tool_subsequence_score"], "Tool Subsequence"
    else:
        # Calculate basic similarity as fallback
        user_source_tools = [tool.get('name', '') for tool in conv.get('user_source', [])]
        used_tools = conv.get('used_tools', [])
        similarity = similarity_metric(user_source_tools, used_tools)
        return similarity, "Basic Levenshtein"


def main():
    st.set_page_config(
        page_title="Conversation Viewer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Conversation Dataset Viewer")
    st.markdown("---")
    
    # Initialize session state for conversation selection
    if 'selected_conv_index' not in st.session_state:
        st.session_state.selected_conv_index = 0
    if 'show_conversation_details' not in st.session_state:
        st.session_state.show_conversation_details = False
    if 'selected_conversation' not in st.session_state:
        st.session_state.selected_conversation = None
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a JSON file with conversations",
        type=['json'],
        help="Upload a JSON file containing conversation data"
    )
    
    # Load data
    conversations = []
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            conversations = json.load(uploaded_file)
            st.success(f"Loaded {len(conversations)} conversations from uploaded file")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return
    else:        
        st.warning("No file uploaded. Please upload a JSON file with conversations.")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Get unique personalities
    user_personalities = list(set(conv['user_personality'] for conv in conversations))
    env_personalities = list(set(conv['environment_personality'] for conv in conversations))
    
    selected_user_personality = st.sidebar.selectbox(
        "User Personality",
        ["All"] + user_personalities,
        index=0
    )
    
    selected_env_personality = st.sidebar.selectbox(
        "Environment Personality", 
        ["All"] + env_personalities,
        index=0
    )
    
    # Filter conversations
    filtered_conversations = conversations
    if selected_user_personality != "All":
        filtered_conversations = [c for c in filtered_conversations if c['user_personality'] == selected_user_personality]
    if selected_env_personality != "All":
        filtered_conversations = [c for c in filtered_conversations if c['environment_personality'] == selected_env_personality]
    
    st.sidebar.markdown(f"**Filtered: {len(filtered_conversations)} conversations**")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Browse Conversations", "📈 Similarity Analysis"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Conversations", len(conversations))
            st.metric("Filtered Conversations", len(filtered_conversations))
        
        with col2:
            st.metric("User Personalities", len(user_personalities))
            st.metric("Environment Personalities", len(env_personalities))
        
        with col3:
            # Calculate average tools
            avg_source_tools = sum(len(c.get('user_source', [])) for c in filtered_conversations) / len(filtered_conversations) if filtered_conversations else 0
            avg_used_tools = sum(len(c.get('used_tools', [])) for c in filtered_conversations) / len(filtered_conversations) if filtered_conversations else 0
            
            # Calculate average similarity
            avg_similarity = 0
            similarity_type = "Basic"
            if filtered_conversations:
                similarities = []
                for conv in filtered_conversations:
                    similarity, sim_type = get_similarity_score(conv)
                    similarities.append(similarity)
                    similarity_type = sim_type  # Use the type from the last conversation (they should all be the same)
                avg_similarity = sum(similarities) / len(similarities)
            
            st.metric("Avg Source Tools", f"{avg_source_tools:.1f}")
            st.metric("Avg Used Tools", f"{avg_used_tools:.1f}")
            st.metric(f"Avg {similarity_type} Similarity", f"{avg_similarity:.3f}")
    
    with tab2:
        st.header("Browse Conversations")
        
        if not filtered_conversations:
            st.warning("No conversations match the current filters.")
            return
        
        # Calculate similarities and create sorted list
        conversations_with_scores = []
        for i, conv in enumerate(filtered_conversations):
            similarity, similarity_type = get_similarity_score(conv)
            conversations_with_scores.append({
                'original_index': i,
                'conversation': conv,
                'similarity': similarity,
                'similarity_type': similarity_type
            })
        
        # Sort by similarity score (ascending - worst to best)
        conversations_with_scores.sort(key=lambda x: x['similarity'])
        
        # Conversation selector
        conv_index = st.selectbox(
            "Select Conversation",
            range(len(conversations_with_scores)),
            format_func=lambda x: f"Conversation {conversations_with_scores[x]['original_index']+1} ({conversations_with_scores[x]['similarity']:.2f}): {conversations_with_scores[x]['conversation']['user_goal'][:50]}...",
            index=st.session_state.selected_conv_index if st.session_state.selected_conv_index < len(conversations_with_scores) else 0
        )
        
        conv = conversations_with_scores[conv_index]['conversation']
        
        # Display conversation details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Conversation Details")
            
            # User goal
            st.markdown("**User Goal:**")
            st.write(conv['user_goal'])
            
            # History
            st.markdown("**Conversation History:**")
            for i, message in enumerate(conv.get('history', [])):
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
                tool_call_id = message.get('tool_call_id', '')
                
                if role == 'system':
                    # Skip system messages
                    continue
                elif role == 'user':
                    st.chat_message("user").write(content)
                elif role == 'assistant':
                    with st.chat_message("assistant"):
                        if content:
                            st.write(content)
                        
                        # Display tool calls if present
                        if tool_calls:
                            st.markdown("**🔧 Tool Calls:**")
                            for tool_call in tool_calls:
                                tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                tool_id = tool_call.get('id', 'unknown')
                                
                                # Try to format JSON arguments nicely
                                try:
                                    parsed_args = json.loads(tool_args)
                                    formatted_args = json.dumps(parsed_args, indent=2)
                                except:
                                    formatted_args = tool_args
                                
                                with st.expander(f"🔧 {tool_name} ({tool_id[:8]}...)", expanded=True):
                                    st.code(formatted_args, language="json")
                elif role == 'tool':
                    # Tool result - format nicely with indentation
                    tool_name = message.get('name', 'unknown_tool')
                    with st.container():
                        st.markdown(f"**🔧 Tool Result: `{tool_name}`** `{tool_call_id[:8] if tool_call_id else 'unknown'}...`")
                        
                        # Format tool result content with proper indentation
                        if content:
                            # Split content into lines and add proper indentation
                            lines = content.split('\n')
                            formatted_content = ""
                            for line in lines:
                                if line.strip():
                                    formatted_content += f"    {line}\n"
                                else:
                                    formatted_content += "\n"
                            
                            st.code(formatted_content.rstrip(), language="text")
                        else:
                            st.code("(No content)", language="text")
                else:
                    st.write(f"**{role.title()}:** {content}")
        
        with col2:
            st.subheader("Metadata")
            
            # Tools
            user_source_tools = [tool.get('name', '') for tool in conv.get('user_source', [])]
            used_tools = conv.get('used_tools', [])
            similarity, similarity_type = get_similarity_score(conv)
            
            st.markdown("**Available Tools:**")
            for tool in user_source_tools:
                st.code(tool)
            
            st.markdown("**Used Tools:**")
            for tool in used_tools:
                st.code(tool)
            
            st.metric(f"{similarity_type} Similarity", f"{similarity:.3f}")
            
            # Personalities
            st.markdown("**User Personality:**")
            st.write(conv['user_personality'])
            
            st.markdown("**Environment Personality:**")
            st.write(conv['environment_personality'])
    
    with tab3:
        st.header("Similarity Analysis")
        
        if not filtered_conversations:
            st.warning("No conversations match the current filters.")
            return
        
        # Calculate similarities for filtered data
        similarities = []
        similarity_type = "Basic"
        for i, conv in enumerate(filtered_conversations):
            similarity, sim_type = get_similarity_score(conv)
            similarity_type = sim_type  # Use the type from conversations
            similarities.append({
                'similarity': similarity,
                'user_personality': conv['user_personality'],
                'environment_personality': conv['environment_personality'],
                'user_goal': conv['user_goal'][:100] + "..." if len(conv['user_goal']) > 100 else conv['user_goal'],
                'original_index': i
            })
        
        # Create DataFrame
        df = pd.DataFrame(similarities)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{similarity_type} Similarity Distribution")
            # Create histogram using pandas and streamlit
            hist_data = np.histogram(df['similarity'], bins=20)
            hist_df = pd.DataFrame({
                'similarity_range': [f"{hist_data[1][i]:.2f}-{hist_data[1][i+1]:.2f}" for i in range(len(hist_data[0]))],
                'count': hist_data[0]
            })
            st.bar_chart(hist_df.set_index('similarity_range'))
            
            st.metric(f"Mean {similarity_type} Similarity", f"{df['similarity'].mean():.3f}")
            st.metric(f"Std {similarity_type} Similarity", f"{df['similarity'].std():.3f}")
        
        with col2:
            st.subheader("Top/Bottom Similarities")
            
            # Top similarities
            st.markdown("**Highest Similarities:**")
            top_sims = df.nlargest(5, 'similarity')
            for idx, row in top_sims.iterrows():
                if st.button(f"Conversation {row['original_index']+1}: {row['similarity']:.3f} - {row['user_goal']}", 
                           key=f"view_top_{idx}", 
                           help="Click to view conversation details"):
                    st.session_state.selected_conversation = filtered_conversations[row['original_index']]
                    st.session_state.show_conversation_details = True
            
            # Bottom similarities
            st.markdown("**Lowest Similarities:**")
            bottom_sims = df.nsmallest(5, 'similarity')
            for idx, row in bottom_sims.iterrows():
                if st.button(f"Conversation {row['original_index']+1}: {row['similarity']:.3f} - {row['user_goal']}", 
                           key=f"view_bottom_{idx}", 
                           help="Click to view conversation details"):
                    st.session_state.selected_conversation = filtered_conversations[row['original_index']]
                    st.session_state.show_conversation_details = True
        
        # Show conversation details if a conversation is selected
        if st.session_state.show_conversation_details and st.session_state.selected_conversation:
            st.markdown("---")
            st.subheader("📖 Conversation Details")
            
            # Add a close button
            if st.button("❌ Close", key="close_details", help="Close conversation details"):
                st.session_state.show_conversation_details = False
                st.session_state.selected_conversation = None
            
            conv = st.session_state.selected_conversation
            
            # Display conversation details in two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # User goal
                st.markdown("**User Goal:**")
                st.write(conv['user_goal'])
                
                # History
                st.markdown("**Conversation History:**")
                for i, message in enumerate(conv.get('history', [])):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    tool_calls = message.get('tool_calls', [])
                    tool_call_id = message.get('tool_call_id', '')
                    
                    if role == 'system':
                        # Skip system messages
                        continue
                    elif role == 'user':
                        st.chat_message("user").write(content)
                    elif role == 'assistant':
                        with st.chat_message("assistant"):
                            if content:
                                st.write(content)
                            
                            # Display tool calls if present
                            if tool_calls:
                                st.markdown("**🔧 Tool Calls:**")
                                for tool_call in tool_calls:
                                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                                    tool_args = tool_call.get('function', {}).get('arguments', '{}')
                                    tool_id = tool_call.get('id', 'unknown')
                                    
                                    # Try to format JSON arguments nicely
                                    try:
                                        parsed_args = json.loads(tool_args)
                                        formatted_args = json.dumps(parsed_args, indent=2)
                                    except:
                                        formatted_args = tool_args
                                    
                                    with st.expander(f"🔧 {tool_name} ({tool_id[:8]}...)", expanded=True):
                                        st.code(formatted_args, language="json")
                    elif role == 'tool':
                        # Tool result - format nicely with indentation
                        tool_name = message.get('name', 'unknown_tool')
                        with st.container():
                            st.markdown(f"**🔧 Tool Result: `{tool_name}`** `{tool_call_id[:8] if tool_call_id else 'unknown'}...`")
                            
                            # Format tool result content with proper indentation
                            if content:
                                # Split content into lines and add proper indentation
                                lines = content.split('\n')
                                formatted_content = ""
                                for line in lines:
                                    if line.strip():
                                        formatted_content += f"    {line}\n"
                                    else:
                                        formatted_content += "\n"
                                
                                st.code(formatted_content.rstrip(), language="text")
                            else:
                                st.code("(No content)", language="text")
                    else:
                        st.write(f"**{role.title()}:** {content}")
            
            with col2:
                st.markdown("**Metadata**")
                
                # Tools
                user_source_tools = [tool.get('name', '') for tool in conv.get('user_source', [])]
                used_tools = conv.get('used_tools', [])
                similarity, similarity_type = get_similarity_score(conv)
                
                st.markdown("**Available Tools:**")
                for tool in user_source_tools:
                    st.code(tool)
                
                st.markdown("**Used Tools:**")
                for tool in used_tools:
                    st.code(tool)
                
                st.metric(f"{similarity_type} Similarity", f"{similarity:.3f}")
                
                # Personalities
                st.markdown("**User Personality:**")
                st.write(conv['user_personality'])
                
                st.markdown("**Environment Personality:**")
                st.write(conv['environment_personality'])


if __name__ == "__main__":
    main() 