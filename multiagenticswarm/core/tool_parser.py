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
        
        # Look for JSON-like structures in the text using a character-by-character approach
        i = 0
        while i < len(response):
            if response[i] == '{':
                # Found start of potential JSON object
                brace_count = 1
                start_pos = i
                i += 1
                in_string = False
                escape_next = False
                
                while i < len(response) and brace_count > 0:
                    char = response[i]
                    
                    if escape_next:
                        escape_next = False
                    elif char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    
                    i += 1
                
                # If we have a complete JSON object (brace_count == 0)
                if brace_count == 0:
                    json_str = response[start_pos:i]
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict) and 'name' in data:
                            tool_calls.append(ToolCallRequest.from_dict({
                                'name': data['name'],
                                'arguments': data.get('args', data.get('arguments', {}))
                            }))
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse potential JSON: {json_str[:100]}...")
                else:
                    # Incomplete JSON, skip it
                    logger.debug(f"Skipping incomplete JSON starting at position {start_pos}")
            else:
                i += 1
        
        return tool_calls
