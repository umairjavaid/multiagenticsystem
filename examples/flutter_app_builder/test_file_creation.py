#!/usr/bin/env python3
"""
Simple test to verify file creation functionality.
"""
import sys
import os
import asyncio
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from direct_flutter_music_builder import create_file

async def test_file_creation():
    """Test the file creation function directly."""
    
    # Test creating a simple file
    script_dir = Path(__file__).parent
    apps_dir = script_dir / "apps"
    test_file_path = "music_stream_app/lib/models/test_track.dart"
    
    test_content = """
class Track {
  final String id;
  final String title;
  final String artist;
  
  Track({required this.id, required this.title, required this.artist});
}
"""
    
    print(f"Testing file creation...")
    print(f"Apps directory: {apps_dir}")
    print(f"File path: {test_file_path}")
    
    result = create_file(test_file_path, test_content)
    print(f"Result: {result}")
    
    # Check if file was created
    full_path = apps_dir / test_file_path
    if full_path.exists():
        print(f"✅ File created successfully at: {full_path}")
        print(f"Content: {full_path.read_text()[:100]}...")
    else:
        print(f"❌ File not created at: {full_path}")

if __name__ == "__main__":
    asyncio.run(test_file_creation())
