import os
import time
from typing import List
from pathlib import Path
from multiagenticsystem.utils.logger import get_logger

logger = get_logger(__name__)

class BuildVerifier:
    """Verify build steps completed successfully."""
    
    @staticmethod
    def verify_project_exists(project_path: str, wait_timeout: int = 30) -> bool:
        """Check if Flutter project structure exists, with optional wait for creation."""
        required_files = [
            'pubspec.yaml',
            'lib/main.dart',
            'android/build.gradle',
            'ios/Runner.xcodeproj'
        ]
        
        # First, wait for the project directory to exist
        if not BuildVerifier.wait_for_file(project_path, wait_timeout):
            logger.error(f"Project path does not exist after {wait_timeout}s: {project_path}")
            return False
        
        # Wait a bit more for Flutter project creation to complete
        time.sleep(2)
        
        # Check for essential files (some may not exist immediately on all platforms)
        essential_files = ['pubspec.yaml', 'lib/main.dart']
        for file in essential_files:
            file_path = os.path.join(project_path, file)
            if not BuildVerifier.wait_for_file(file_path, 10):
                logger.error(f"Missing essential file: {file}")
                return False
        
        # Check for other files but don't fail if they don't exist (platform-specific)
        optional_files = ['android/build.gradle', 'ios/Runner.xcodeproj']
        missing_optional = []
        for file in optional_files:
            file_path = os.path.join(project_path, file)
            if not os.path.exists(file_path):
                missing_optional.append(file)
        
        if missing_optional:
            logger.warning(f"Optional files not found (may be platform-specific): {missing_optional}")
        
        logger.info(f"Project structure verified: {project_path}")
        return True
    
    @staticmethod
    def verify_file_content(file_path: str, required_content: List[str] = None) -> bool:
        """Verify file exists and optionally contains required content."""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
            
        if required_content:
            with open(file_path, 'r') as f:
                content = f.read()
                for req in required_content:
                    if req not in content:
                        logger.error(f"File {file_path} missing required content: {req}")
                        return False
                        
        logger.info(f"File content verified: {file_path}")
        return True
    
    @staticmethod
    def wait_for_file(file_path: str, timeout: int = 5) -> bool:
        """Wait for file or directory to be created."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                logger.info(f"File/directory found: {file_path}")
                return True
            time.sleep(0.5)  # Check every 0.5 seconds instead of 0.1
        logger.error(f"Timeout waiting for file/directory: {file_path}")
        return False
    
    @staticmethod
    def verify_dependencies(project_path: str, dependencies: List[str]) -> bool:
        """Verify pubspec.yaml contains required dependencies."""
        pubspec_path = os.path.join(project_path, 'pubspec.yaml')
        
        if not os.path.exists(pubspec_path):
            logger.error("pubspec.yaml not found")
            return False
            
        with open(pubspec_path, 'r') as f:
            content = f.read()
            
        missing = []
        for dep in dependencies:
            if dep not in content:
                missing.append(dep)
                
        if missing:
            logger.error(f"Missing dependencies: {missing}")
            return False
            
        logger.info("All dependencies verified")
        return True
    
    @staticmethod
    def verify_files_exist(file_paths: List[str]) -> bool:
        """Verify that multiple files exist."""
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
        logger.info(f"All files verified: {file_paths}")
        return True
    
    @staticmethod
    def verify_file_contains(file_path: str, content: str) -> bool:
        """Verify that a file exists and contains specific content."""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if content in file_content:
                    logger.info(f"File {file_path} contains required content: {content}")
                    return True
                else:
                    logger.error(f"File {file_path} does not contain required content: {content}")
                    return False
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return False
    
    @staticmethod
    def verify_command_succeeds(command: str, working_dir: str = None) -> bool:
        """Verify that a command executes successfully."""
        import subprocess
        try:
            cwd = working_dir if working_dir else os.getcwd()
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                logger.info(f"Command succeeded: {command}")
                return True
            else:
                logger.error(f"Command failed: {command} (exit code: {result.returncode})")
                logger.error(f"STDERR: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return False
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return False
    
    @staticmethod
    def verify_flutter_project_created(project_path: str, wait_timeout: int = 30) -> bool:
        """Verify that basic Flutter project structure was created successfully."""
        logger.info(f"Verifying Flutter project creation at: {project_path}")
        
        # Wait for project directory to exist
        if not BuildVerifier.wait_for_file(project_path, wait_timeout):
            logger.error(f"Project directory not created: {project_path}")
            return False
        
        # Essential Flutter files that should always exist
        essential_files = [
            'pubspec.yaml',
            'lib/main.dart',
            'README.md',
            'analysis_options.yaml'
        ]
        
        # Wait for essential files
        for file in essential_files:
            file_path = os.path.join(project_path, file)
            if not BuildVerifier.wait_for_file(file_path, 15):
                logger.error(f"Essential Flutter file not created: {file}")
                return False
        
        # Verify pubspec.yaml contains Flutter SDK reference
        pubspec_path = os.path.join(project_path, 'pubspec.yaml')
        try:
            with open(pubspec_path, 'r') as f:
                content = f.read()
                if 'flutter:' not in content or 'sdk: flutter' not in content:
                    logger.error("pubspec.yaml does not contain proper Flutter configuration")
                    return False
        except Exception as e:
            logger.error(f"Error reading pubspec.yaml: {e}")
            return False
        
        logger.info(f"Flutter project successfully verified: {project_path}")
        return True
