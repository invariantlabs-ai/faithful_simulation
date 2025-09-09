import json
import html
from datetime import datetime
from typing import Dict, List, Any
import os

def create_simple_trace_html(trace_data: List[Dict[str, Any]], output_file: str = "agent_trace.html"):
    """
    Create a simple HTML visualization of agent traces.
    
    Args:
        trace_data: List of conversation turns (as parsed from JSON)
        output_file: Output HTML file name
    """
    
    def escape_html(text):
        """Escape HTML characters in text"""
        if text is None:
            return ""
        return html.escape(str(text))
    
    def get_role_color(role):
        """Get background color for different roles"""
        colors = {
            'system': '#e3f2fd',
            'user': '#f3e5f5', 
            'assistant': '#e8f5e8',
            'tool': '#fff3e0'
        }
        return colors.get(role, '#f5f5f5')
    
    def get_role_icon(role):
        """Get icon for different roles"""
        icons = {
            'system': '⚙️',
            'user': '👤',
            'assistant': '🤖',
            'tool': '🔧'
        }
        return icons.get(role, '❓')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Trace Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        
        .trace-item {{
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .trace-header {{
            padding: 15px 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .trace-content {{
            padding: 20px;
            background-color: white;
        }}
        
        .content-text {{
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-bottom: 15px;
        }}
        
        .tool-calls {{
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .tool-call {{
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }}
        
        .tool-name {{
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        
        .tool-args {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre;
        }}
        
        .metadata {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 15px;
        }}
        
        .step-number {{
            background-color: rgba(0,0,0,0.1);
            color: rgba(0,0,0,0.7);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Agent Trace Visualization</h1>
            <p>Complete conversation flow and tool usage</p>
        </div>
"""
    
    # Generate HTML for each trace item
    for i, item in enumerate(trace_data):
        role = item.get('role', 'unknown')
        role_color = get_role_color(role)
        role_icon = get_role_icon(role)
        
        html_content += f"""
        <div class="trace-item">
            <div class="trace-header" style="background-color: {role_color};">
                <span>{role_icon}</span>
                <span>{role.upper()}</span>
                <span class="step-number">#{i+1}</span>
            </div>
            <div class="trace-content">
        """
        
        # Add metadata if present
        metadata_items = []
        if item.get('tool_call_id'):
            metadata_items.append(f"Tool Call ID: {escape_html(item['tool_call_id'])}")
        if item.get('name'):
            metadata_items.append(f"Tool Name: {escape_html(item['name'])}")
        
        if metadata_items:
            html_content += f'<div class="metadata">{" | ".join(metadata_items)}</div>'
        
        # Add main content
        if item.get('content'):
            html_content += f'<div class="content-text">{escape_html(item["content"])}</div>'
        
        # Add tool calls
        if item.get('tool_calls'):
            html_content += '<div class="tool-calls"><strong>🔧 Tool Calls:</strong>'
            for j, tool_call in enumerate(item['tool_calls']):
                function_name = tool_call.get('function', {}).get('name', 'Unknown')
                function_args = tool_call.get('function', {}).get('arguments', '{}')
                tool_id = tool_call.get('id', 'N/A')
                
                html_content += f"""
                    <div class="tool-call">
                        <div class="tool-name">🔧 {escape_html(function_name)}</div>
                        <div style="font-size: 11px; color: #666; margin-bottom: 8px;">ID: {escape_html(tool_id)}</div>
                        <div class="tool-args">{escape_html(function_args)}</div>
                    </div>
                """
            html_content += '</div>'
        
        html_content += """
            </div>
        </div>
        """
    
    # Close HTML
    html_content += """
    </div>
</body>
</html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Simple trace visualization saved to {output_file}")
    return output_file

# Command line usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        print("Usage: python create_simple_trace_html.py <trace_file.json>")
        print("Example: python create_simple_trace_html.py my_trace.json")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.json', '_visualization.html')
    
    if len(sys.argv) == 1:
        traces_files = [file for file in os.listdir("traces") if file.endswith(".json")]
        traces_files.sort()
        input_file = os.path.join("traces", traces_files[-1])
        output_file = input_file.replace('.json', '_visualization.html')

    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        
        create_simple_trace_html(trace_data, output_file)
        print(f"🎉 Open {output_file} in your browser to view the visualization")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{input_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)