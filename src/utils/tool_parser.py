"""
Tool Parser Module for LLM Deception Research Platform
Robust XML parsing for extracting tool commands from agent responses.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from lxml import etree
from lxml.etree import XMLSyntaxError

logger = logging.getLogger(__name__)

# Define valid tool names based on Barkur et al. (2025)
VALID_TOOLS = {
    'eyes', 'ears', 'legs', 'left_hand', 'right_hand',
    'memory', 'persona', 'thought_process', 'tool'
}

class ToolParser:
    """
    Robust parser for extracting tool commands from LLM responses.
    Handles malformed XML, nested tags, and multiple tools.
    """
    
    def __init__(self):
        """Initialize the tool parser with regex patterns."""
        # Primary pattern for well-formed tool tags
        self.tool_pattern = re.compile(
            r'<(tool|eyes|ears|legs|left_hand|right_hand|memory|persona|thought_process)>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE
        )
        
        # Fallback pattern for unclosed tags
        self.unclosed_pattern = re.compile(
            r'<(tool|eyes|ears|legs|left_hand|right_hand|memory|persona|thought_process)>([^<]*?)(?=<|$)',
            re.DOTALL | re.IGNORECASE
        )
        
        # Pattern for nested tool tags (e.g., <tool><eyes>...</eyes></tool>)
        self.nested_pattern = re.compile(
            r'<tool>(.*?)</tool>',
            re.DOTALL | re.IGNORECASE
        )
    
    def extract_tools(self, response: str) -> List[Dict[str, str]]:
        """
        Extract all tool commands from an agent response.
        
        Args:
            response: The raw response from the agent LLM
            
        Returns:
            List of dictionaries with 'tool' and 'content' keys
        """
        tools = []
        
        # First, try to extract nested tool tags
        nested_matches = self.nested_pattern.findall(response)
        for nested_content in nested_matches:
            # Parse content within <tool> tags
            inner_tools = self._parse_inner_tools(nested_content)
            tools.extend(inner_tools)
        
        # Then, extract standalone tool tags
        standalone_tools = self._extract_standalone_tools(response)
        tools.extend(standalone_tools)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools:
            tool_key = (tool['tool'], tool['content'].strip())
            if tool_key not in seen:
                seen.add(tool_key)
                unique_tools.append(tool)
        
        logger.info(f"Extracted {len(unique_tools)} tool commands from response")
        return unique_tools
    
    def _parse_inner_tools(self, content: str) -> List[Dict[str, str]]:
        """
        Parse tools from within a <tool> wrapper tag.
        
        Args:
            content: Content within <tool>...</tool> tags
            
        Returns:
            List of tool dictionaries
        """
        tools = []
        
        # Try XML parsing first
        try:
            # Wrap in root element for valid XML
            xml_content = f"<root>{content}</root>"
            root = etree.fromstring(xml_content.encode('utf-8'))
            
            for element in root:
                if element.tag.lower() in VALID_TOOLS:
                    tools.append({
                        'tool': element.tag.lower(),
                        'content': (element.text or '').strip()
                    })
        except (XMLSyntaxError, Exception) as e:
            logger.debug(f"XML parsing failed, falling back to regex: {e}")
            # Fallback to regex parsing
            tools.extend(self._regex_extract(content))
        
        return tools
    
    def _extract_standalone_tools(self, response: str) -> List[Dict[str, str]]:
        """
        Extract tool commands that are not wrapped in <tool> tags.
        
        Args:
            response: The full agent response
            
        Returns:
            List of tool dictionaries
        """
        tools = []
        
        # Remove nested tool content to avoid double extraction
        response_cleaned = self.nested_pattern.sub('', response)
        
        # Extract using primary pattern
        tools.extend(self._regex_extract(response_cleaned))
        
        return tools
    
    def _regex_extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract tools using regex patterns.
        
        Args:
            text: Text to extract tools from
            
        Returns:
            List of tool dictionaries
        """
        tools = []
        
        # Try well-formed tags first
        matches = self.tool_pattern.findall(text)
        for tool_name, content in matches:
            if tool_name.lower() != 'tool':  # Skip wrapper tags
                tools.append({
                    'tool': tool_name.lower(),
                    'content': content.strip()
                })
        
        # If no well-formed tags found, try unclosed tags
        if not tools:
            matches = self.unclosed_pattern.findall(text)
            for tool_name, content in matches:
                if tool_name.lower() != 'tool' and content.strip():
                    tools.append({
                        'tool': tool_name.lower(),
                        'content': content.strip()
                    })
                    logger.warning(f"Extracted unclosed tag: {tool_name}")
        
        return tools
    
    def validate_tool(self, tool_name: str) -> bool:
        """
        Validate if a tool name is in the allowed list.
        
        Args:
            tool_name: Name of the tool to validate
            
        Returns:
            True if valid, False otherwise
        """
        return tool_name.lower() in VALID_TOOLS
    
    def parse_movement_command(self, command: str) -> Optional[Dict[str, any]]:
        """
        Parse movement commands for the legs tool.
        
        Args:
            command: Movement command string (e.g., "FORWARD LEFT 20 cm")
            
        Returns:
            Dictionary with parsed movement parameters or None if invalid
            
        Example:
            "FORWARD LEFT 20 cm" -> {
                'direction': 'FORWARD',
                'modifier': 'LEFT',
                'distance': 20,
                'unit': 'cm'
            }
        """
        # Pattern for movement commands
        movement_pattern = re.compile(
            r'(FORWARD|BACKWARD|LEFT|RIGHT)\s*'
            r'(?:(LEFT|RIGHT)\s*)?'
            r'(\d+(?:\.\d+)?)\s*'
            r'(cm|m|meters?|centimeters?)?',
            re.IGNORECASE
        )
        
        match = movement_pattern.match(command.strip())
        if match:
            direction, modifier, distance, unit = match.groups()
            return {
                'direction': direction.upper(),
                'modifier': modifier.upper() if modifier else None,
                'distance': float(distance),
                'unit': (unit or 'cm').lower()
            }
        
        # Fallback for simple commands
        simple_pattern = re.compile(r'(FORWARD|BACKWARD|LEFT|RIGHT)', re.IGNORECASE)
        match = simple_pattern.match(command.strip())
        if match:
            return {
                'direction': match.group(1).upper(),
                'modifier': None,
                'distance': 1.0,
                'unit': 'step'
            }
        
        logger.warning(f"Could not parse movement command: {command}")
        return None
    
    def format_tools_for_display(self, tools: List[Dict[str, str]]) -> str:
        """
        Format extracted tools for human-readable display.
        
        Args:
            tools: List of tool dictionaries
            
        Returns:
            Formatted string representation
        """
        if not tools:
            return "No tools used"
        
        formatted = []
        for i, tool in enumerate(tools, 1):
            formatted.append(f"{i}. [{tool['tool']}]: {tool['content']}")
        
        return "\n".join(formatted)
    
    def extract_with_context(self, response: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Extract tools and return the response with tool markers.
        
        Args:
            response: The raw agent response
            
        Returns:
            Tuple of (tools list, marked response)
        """
        tools = self.extract_tools(response)
        
        # Mark tool locations in response
        marked_response = response
        for i, tool in enumerate(tools, 1):
            # Add markers for visualization
            pattern = re.compile(
                f"<{tool['tool']}>.*?</{tool['tool']}>",
                re.DOTALL | re.IGNORECASE
            )
            marked_response = pattern.sub(
                f"[TOOL_{i}: {tool['tool']}]",
                marked_response,
                count=1
            )
        
        return tools, marked_response


def extract_tools(response: str) -> List[Dict[str, str]]:
    """
    Convenience function for extracting tools from a response.
    
    Args:
        response: The agent's response containing tool commands
        
    Returns:
        List of tool dictionaries with 'tool' and 'content' keys
    """
    parser = ToolParser()
    return parser.extract_tools(response)


def validate_tool(tool_name: str) -> bool:
    """
    Convenience function for validating tool names.
    
    Args:
        tool_name: Name of the tool to validate
        
    Returns:
        True if valid, False otherwise
    """
    parser = ToolParser()
    return parser.validate_tool(tool_name)


def parse_movement_command(command: str) -> Optional[Dict[str, any]]:
    """
    Convenience function for parsing movement commands.
    
    Args:
        command: Movement command string
        
    Returns:
        Parsed movement dictionary or None
    """
    parser = ToolParser()
    return parser.parse_movement_command(command)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Test cases
    test_responses = [
        # Well-formed XML
        """
        I'll explore the lab now.
        <tool>
        <eyes>Look around the laboratory</eyes>
        </tool>
        """,
        
        # Multiple tools
        """
        <eyes>Examine the workbench</eyes>
        <left_hand>Pick up the screwdriver</left_hand>
        """,
        
        # Malformed/unclosed tags
        """
        <memory>This is a robotics lab with equipment
        <legs>FORWARD LEFT 20 cm
        """,
        
        # Nested structure
        """
        <tool>
        <eyes>Look at the computer screen</eyes>
        <ears>Listen for any sounds</ears>
        </tool>
        <thought_process>I need to understand my environment</thought_process>
        """,
        
        # Movement commands
        """
        <legs>FORWARD 50 cm</legs>
        <legs>LEFT 30 centimeters</legs>
        <legs>BACKWARD RIGHT 1 m</legs>
        """
    ]
    
    parser = ToolParser()
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Response: {response[:100]}...")
        tools = parser.extract_tools(response)
        print(f"Extracted {len(tools)} tools:")
        print(parser.format_tools_for_display(tools))
        
        # Test movement parsing if legs tool present
        for tool in tools:
            if tool['tool'] == 'legs':
                movement = parser.parse_movement_command(tool['content'])
                print(f"  Movement parsed: {movement}")