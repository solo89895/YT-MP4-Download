import subprocess
import sys
import os
import signal
import time
from typing import List, Optional

class ServerManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        
    def start_backend(self) -> None:
        """Start the backend server"""
        print("Starting Backend Server...")
        backend_cmd = [
            "cmd", "/c",
            "cd backend && .\\.venv\\Scripts\\activate && uvicorn main:app --reload"
        ]
        backend_process = subprocess.Popen(
            backend_cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        self.processes.append(backend_process)
        print("Backend server started on http://localhost:8000")
        
    def start_frontend(self) -> None:
        """Start the frontend server"""
        print("Starting Frontend Server...")
        frontend_cmd = ["npm", "run", "dev"]
        frontend_process = subprocess.Popen(
            frontend_cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        self.processes.append(frontend_process)
        print("Frontend server started on http://localhost:8080")
        
    def start_all(self) -> None:
        """Start all servers"""
        self.start_backend()
        # Wait for backend to initialize
        time.sleep(5)
        self.start_frontend()
        print("\nAll servers have been started!")
        print("Frontend: http://localhost:8080")
        print("Backend: http://localhost:8000")
        
    def stop_all(self) -> None:
        """Stop all running servers"""
        print("\nStopping all servers...")
        for process in self.processes:
            try:
                # On Windows, we need to terminate the process group
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)])
                else:
                    process.terminate()
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        self.processes.clear()
        print("All servers have been stopped.")
        
    def is_running(self) -> bool:
        """Check if any servers are running"""
        return len(self.processes) > 0

def main():
    manager = ServerManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        manager.stop_all()
    else:
        if manager.is_running():
            print("Servers are already running!")
            return
        manager.start_all()
        
        try:
            # Keep the script running and handle Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
            manager.stop_all()

if __name__ == "__main__":
    main() 