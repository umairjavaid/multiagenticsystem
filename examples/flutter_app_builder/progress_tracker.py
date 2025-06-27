from datetime import datetime
from typing import List, Dict, Any

class ProgressTracker:
    """Track build progress and provide detailed feedback."""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def start_step(self, name: str, description: str):
        """Start tracking a new step."""
        self.steps.append({
            "name": name,
            "description": description,
            "start_time": datetime.now(),
            "status": "running",
            "details": []
        })
        print(f"\n🔄 {name}: {description}")
        
    def add_detail(self, detail: str):
        """Add detail to current step."""
        if self.steps:
            self.steps[-1]["details"].append(detail)
            print(f"   • {detail}")
            
    def complete_step(self, success: bool = True, error: str = None):
        """Mark current step as complete."""
        if self.steps:
            step = self.steps[-1]
            step["end_time"] = datetime.now()
            step["duration"] = (step["end_time"] - step["start_time"]).total_seconds()
            step["status"] = "success" if success else "failed"
            if error:
                step["error"] = error
                
            if success:
                print(f"   ✅ Completed in {step['duration']:.2f}s")
            else:
                print(f"   ❌ Failed: {error}")
                
    def get_summary(self) -> Dict[str, Any]:
        """Get build summary."""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for s in self.steps if s["status"] == "success")
        failed = sum(1 for s in self.steps if s["status"] == "failed")
        
        return {
            "total_duration": total_duration,
            "total_steps": len(self.steps),
            "successful": successful,
            "failed": failed,
            "steps": self.steps
        }
