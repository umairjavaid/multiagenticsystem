import json
from typing import List, Dict, Any, Optional
from ..core.base_tool import ToolCallRequest
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ToolCallParser:
    """Parse tool calls from agent responses using JSON parsing."""
    
    @staticmethod
    def extract_tool_calls(response: str) -> List[ToolCallRequest]:
        """
        Extract tool calls from agent response.
        Looks for JSON objects with 'name' and 'args' fields.
        """
        tool_calls = []
        
        # Try to parse the entire response as JSON first
        try:
            data = json.loads(response)
            if isinstance(data, dict) and 'name' in data:
                tool_calls.append(ToolCallRequest.from_dict({
                    'name': data['name'],
                    'arguments': data.get('args', data.get('arguments', {}))
                }))
                return tool_calls
        except json.JSONDecodeError:
            pass
        
        # Look for JSON-like structures in the text
        # Split by lines and look for JSON objects
        lines = response.split('\n')
        current_json = []
        brace_count = 0
        
        for line in lines:
            # Count braces to find complete JSON objects
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            if open_braces > 0 or brace_count > 0:
                current_json.append(line)
                brace_count += open_braces - close_braces
                
                if brace_count == 0 and current_json:
                    # Try to parse the accumulated JSON
                    json_str = '\n'.join(current_json)
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict) and 'name' in data:
                            tool_calls.append(ToolCallRequest.from_dict({
                                'name': data['name'],
                                'arguments': data.get('args', data.get('arguments', {}))
                            }))
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse potential JSON: {json_str[:100]}...")
                    current_json = []
        
        return tool_calls
