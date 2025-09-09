#!/usr/bin/env python3
"""
Convert agent traces (JSON format) to minimal HTML visualization.
"""

import json
import html
import re
from pathlib import Path
from typing import Dict, Any, List


def is_json_content(content: str) -> bool:
    """Check if content is JSON by trying to parse it."""
    if not content or not content.strip():
        return False
    
    content = content.strip()
    if not (content.startswith('{') or content.startswith('[')):
        return False
    
    try:
        json.loads(content)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def format_json_content(content: str) -> str:
    """Format JSON content with proper indentation."""
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2)
    except:
        return content


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Format tool calls into readable HTML."""
    if not tool_calls:
        return ""
    
    html_parts = []
    for call in tool_calls:
        func_name = call.get('function', {}).get('name', 'unknown')
        func_args = call.get('function', {}).get('arguments', '{}')
        
        # Try to format arguments as JSON
        try:
            args_obj = json.loads(func_args) if func_args else {}
            formatted_args = json.dumps(args_obj, indent=2)
        except:
            formatted_args = func_args
        
        html_parts.append(f"""
        <div class="tool-call">
            <strong>Function:</strong> {html.escape(func_name)}
            <pre class="json-content">{html.escape(formatted_args)}</pre>
        </div>
        """)
    
    return ''.join(html_parts)


def convert_message_to_html(message: Dict[str, Any]) -> str:
    """Convert a single message to HTML."""
    role = message.get('role', 'unknown')
    content = message.get('content', '')
    tool_calls = message.get('tool_calls', [])
    tool_call_id = message.get('tool_call_id')
    name = message.get('name')
    
    # Determine message type and styling
    if role == 'system':
        css_class = 'message-system'
        header = 'System'
    elif role == 'user':
        css_class = 'message-user'
        header = 'User'
    elif role == 'assistant':
        css_class = 'message-assistant'
        header = 'Assistant'
    elif role == 'tool':
        css_class = 'message-tool'
        header = f'Tool Response'
        if name:
            header += f' ({name})'
    else:
        css_class = 'message-other'
        header = role.title()
    
    # Format content
    content_html = ''
    if content:
        if is_json_content(content):
            formatted_content = format_json_content(content)
            content_html = f'<pre class="json-content">{html.escape(formatted_content)}</pre>'
        else:
            # Convert newlines to <br> and escape HTML
            formatted_content = html.escape(content).replace('\n', '<br>')
            content_html = f'<div class="text-content">{formatted_content}</div>'
    
    # Format tool calls
    tool_calls_html = format_tool_calls(tool_calls) if tool_calls else ''
    
    # Build message HTML
    message_html = f"""
    <div class="message {css_class}">
        <div class="message-header">{header}</div>
        {content_html}
        {tool_calls_html}
    </div>
    """
    
    return message_html


def generate_html(messages: List[Dict[str, Any]], title: str = "Agent Trace") -> str:
    """Generate complete HTML document from messages."""
    
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            color: #333;
        }
        
        .container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        h1 {
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            font-size: 24px;
            font-weight: 600;
        }
        
        .message {
            border-bottom: 1px solid #e9ecef;
            padding: 16px 20px;
        }
        
        .message:last-child {
            border-bottom: none;
        }
        
        .message-header {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .message-system {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }
        
        .message-system .message-header {
            color: #856404;
        }
        
        .message-user {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
        }
        
        .message-user .message-header {
            color: #0c5460;
        }
        
        .message-assistant {
            background: #d4edda;
            border-left: 4px solid #28a745;
        }
        
        .message-assistant .message-header {
            color: #155724;
        }
        
        .message-tool {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        
        .message-tool .message-header {
            color: #721c24;
        }
        
        .message-other {
            background: #e2e3e5;
            border-left: 4px solid #6c757d;
        }
        
        .text-content {
            line-height: 1.5;
            margin-top: 4px;
        }
        
        .json-content {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 12px;
            margin-top: 8px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre;
            color: #495057;
        }
        
        .tool-call {
            margin-top: 12px;
            padding: 12px;
            background: rgba(255,255,255,0.7);
            border-radius: 4px;
            border: 1px solid rgba(0,0,0,0.1);
        }
        
        .tool-call strong {
            color: #495057;
        }
    </style>
    """
    
    messages_html = ''.join(convert_message_to_html(msg) for msg in messages)
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        {messages_html}
    </div>
</body>
</html>"""
    
    return html_template


def convert_trace_file(input_file: str, output_file: str = None, title: str = None):
    """Convert a JSON trace file to HTML."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output file name
    if output_file is None:
        output_file = input_path.with_suffix('.html')
    
    # Determine title
    if title is None:
        title = f"Agent Trace - {input_path.stem}"
    
    # Load and parse JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_file}: {e}")
    
    # Extract messages - handle different possible structures
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
    elif isinstance(data, dict) and any(key in data for key in ['role', 'content']):
        messages = [data]  # Single message
    else:
        raise ValueError("Could not find messages in the JSON structure")
    
    # Generate HTML
    html_content = generate_html(messages, title)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Converted {input_file} -> {output_file}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert agent traces to HTML')
    parser.add_argument('input_file', help='Input JSON file containing agent trace')
    parser.add_argument('-o', '--output', help='Output HTML file (default: same name with .html extension)')
    parser.add_argument('-t', '--title', help='Title for the HTML page')
    
    args = parser.parse_args()
    
    try:
        convert_trace_file(args.input_file, args.output, args.title)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())