"""
Genesis Client Examples
Examples for connecting from different client types
Author: eddy
"""

import requests
import json
from typing import Dict, Any, Optional


# ==================== HTTP Client ====================

class GenesisHTTPClient:
    """
    Simple HTTP client for Genesis Server
    Use for: Simple scripts, batch processing
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
    
    def create_session(self, client_type: str = "http"):
        """Create session"""
        response = requests.post(f"{self.base_url}/api/session/create", json={
            'client_type': client_type
        })
        data = response.json()
        self.session_id = data['session_id']
        return data
    
    def submit_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """Submit task and return task_id"""
        response = requests.post(f"{self.base_url}/api/task/submit", json={
            'task_type': task_type,
            'params': params,
            'session_id': self.session_id
        })
        data = response.json()
        return data['task_id']
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        response = requests.get(f"{self.base_url}/api/task/{task_id}")
        return response.json()['task']
    
    def wait_for_task(self, task_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Wait for task completion"""
        import time
        
        while True:
            task = self.get_task_status(task_id)
            
            if task['status'] in ['completed', 'failed', 'cancelled']:
                return task
            
            time.sleep(poll_interval)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image (blocking)"""
        task_id = self.submit_task('generate', {
            'prompt': prompt,
            **kwargs
        })
        
        return self.wait_for_task(task_id)


# ==================== WebSocket Client ====================

try:
    import socketio
    
    class GenesisWebSocketClient:
        """
        WebSocket client for Genesis Server
        Use for: Real-time updates, interactive UIs
        """
        
        def __init__(self, base_url: str = "http://localhost:5000"):
            self.base_url = base_url
            self.sio = socketio.Client()
            self.connected = False
            self.task_results = {}
            
            # Setup event handlers
            self._setup_handlers()
        
        def _setup_handlers(self):
            """Setup event handlers"""
            
            @self.sio.on('connect')
            def on_connect():
                self.connected = True
                print("Connected to Genesis Server")
            
            @self.sio.on('disconnect')
            def on_disconnect():
                self.connected = False
                print("Disconnected from Genesis Server")
            
            @self.sio.on('connected')
            def on_connected(data):
                print(f"Server confirmed connection: {data}")
            
            @self.sio.on('progress')
            def on_progress(data):
                task_id = data['task_id']
                progress = data['progress']
                message = data.get('message', '')
                print(f"Progress [{task_id}]: {progress}% - {message}")
            
            @self.sio.on('task_complete')
            def on_task_complete(data):
                task_id = data['task_id']
                self.task_results[task_id] = {
                    'status': 'completed',
                    'result': data['result']
                }
                print(f"Task completed: {task_id}")
            
            @self.sio.on('task_error')
            def on_task_error(data):
                task_id = data['task_id']
                self.task_results[task_id] = {
                    'status': 'failed',
                    'error': data['error']
                }
                print(f"Task failed: {task_id} - {data['error']}")
        
        def connect(self):
            """Connect to server"""
            if not self.connected:
                self.sio.connect(self.base_url)
        
        def disconnect(self):
            """Disconnect from server"""
            if self.connected:
                self.sio.disconnect()
        
        def submit_task(self, task_type: str, params: Dict[str, Any]) -> None:
            """Submit task"""
            self.sio.emit('submit_task', {
                'task_type': task_type,
                'params': params
            })
        
        def generate_async(self, prompt: str, **kwargs):
            """Generate image (async)"""
            self.submit_task('generate', {
                'prompt': prompt,
                **kwargs
            })
        
        def wait_for_result(self, task_id: str, timeout: float = 60.0):
            """Wait for task result"""
            import time
            start = time.time()
            
            while time.time() - start < timeout:
                if task_id in self.task_results:
                    return self.task_results[task_id]
                time.sleep(0.1)
            
            raise TimeoutError(f"Task {task_id} timeout")

except ImportError:
    print("Warning: socketio not installed. WebSocket client unavailable.")
    print("Install: pip install python-socketio[client]")


# ==================== Tkinter Example ====================

def create_tkinter_client():
    """
    Example Tkinter GUI client
    """
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        
        class GenesisTkinterApp:
            def __init__(self, root):
                self.root = root
                self.root.title("Genesis Client - Tkinter")
                self.root.geometry("800x600")
                
                self.client = GenesisHTTPClient()
                self.client.create_session("tkinter")
                
                self._create_widgets()
            
            def _create_widgets(self):
                # Prompt input
                ttk.Label(self.root, text="Prompt:").pack(pady=5)
                self.prompt_entry = ttk.Entry(self.root, width=80)
                self.prompt_entry.pack(pady=5)
                
                # Parameters
                params_frame = ttk.Frame(self.root)
                params_frame.pack(pady=10)
                
                ttk.Label(params_frame, text="Steps:").grid(row=0, column=0)
                self.steps_var = tk.IntVar(value=20)
                ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.steps_var, width=10).grid(row=0, column=1)
                
                ttk.Label(params_frame, text="CFG:").grid(row=0, column=2, padx=10)
                self.cfg_var = tk.DoubleVar(value=7.0)
                ttk.Spinbox(params_frame, from_=1.0, to=20.0, textvariable=self.cfg_var, width=10, increment=0.5).grid(row=0, column=3)
                
                # Generate button
                self.gen_button = ttk.Button(self.root, text="Generate", command=self.generate)
                self.gen_button.pack(pady=10)
                
                # Status
                self.status_label = ttk.Label(self.root, text="Ready")
                self.status_label.pack(pady=5)
                
                # Progress
                self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
                self.progress.pack(pady=10)
                
                # Log
                ttk.Label(self.root, text="Log:").pack()
                self.log_text = scrolledtext.ScrolledText(self.root, height=15, width=90)
                self.log_text.pack(pady=5)
            
            def log(self, message: str):
                self.log_text.insert(tk.END, f"{message}\n")
                self.log_text.see(tk.END)
            
            def generate(self):
                prompt = self.prompt_entry.get()
                if not prompt:
                    self.log("Error: Please enter a prompt")
                    return
                
                self.gen_button.config(state='disabled')
                self.status_label.config(text="Generating...")
                self.progress['value'] = 0
                
                # Submit task
                try:
                    task_id = self.client.submit_task('generate', {
                        'prompt': prompt,
                        'steps': self.steps_var.get(),
                        'cfg_scale': self.cfg_var.get()
                    })
                    
                    self.log(f"Task submitted: {task_id}")
                    
                    # Poll for status (in real app, use threading)
                    self.root.after(1000, self.check_status, task_id)
                    
                except Exception as e:
                    self.log(f"Error: {e}")
                    self.gen_button.config(state='normal')
                    self.status_label.config(text="Error")
            
            def check_status(self, task_id: str):
                try:
                    task = self.client.get_task_status(task_id)
                    status = task['status']
                    progress = task['progress']
                    
                    self.progress['value'] = progress
                    
                    if status == 'completed':
                        self.log("Generation completed!")
                        self.status_label.config(text="Completed")
                        self.gen_button.config(state='normal')
                    elif status == 'failed':
                        self.log(f"Generation failed: {task.get('error')}")
                        self.status_label.config(text="Failed")
                        self.gen_button.config(state='normal')
                    else:
                        # Keep checking
                        self.root.after(1000, self.check_status, task_id)
                
                except Exception as e:
                    self.log(f"Error checking status: {e}")
                    self.gen_button.config(state='normal')
        
        root = tk.Tk()
        app = GenesisTkinterApp(root)
        return root
    
    except ImportError:
        print("Tkinter not available")
        return None


# ==================== PyQt Example ====================

def create_pyqt_client():
    """
    Example PyQt GUI client
    """
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QLabel, QLineEdit, QPushButton,
                                     QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar)
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class GenerationThread(QThread):
            progress_signal = pyqtSignal(int, str)
            finished_signal = pyqtSignal(dict)
            
            def __init__(self, client, params):
                super().__init__()
                self.client = client
                self.params = params
            
            def run(self):
                try:
                    task_id = self.client.submit_task('generate', self.params)
                    self.progress_signal.emit(10, f"Task submitted: {task_id}")
                    
                    # Poll status
                    import time
                    while True:
                        task = self.client.get_task_status(task_id)
                        status = task['status']
                        progress = task['progress']
                        
                        self.progress_signal.emit(progress, f"Status: {status}")
                        
                        if status in ['completed', 'failed', 'cancelled']:
                            self.finished_signal.emit(task)
                            break
                        
                        time.sleep(1)
                
                except Exception as e:
                    self.finished_signal.emit({
                        'status': 'failed',
                        'error': str(e)
                    })
        
        class GenesisPyQtWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Genesis Client - PyQt5")
                self.setGeometry(100, 100, 800, 600)
                
                self.client = GenesisHTTPClient()
                self.client.create_session("pyqt")
                self.generation_thread = None
                
                self._create_ui()
            
            def _create_ui(self):
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)
                
                # Prompt
                layout.addWidget(QLabel("Prompt:"))
                self.prompt_input = QLineEdit()
                layout.addWidget(self.prompt_input)
                
                # Parameters
                params_layout = QHBoxLayout()
                params_layout.addWidget(QLabel("Steps:"))
                self.steps_spin = QSpinBox()
                self.steps_spin.setRange(1, 100)
                self.steps_spin.setValue(20)
                params_layout.addWidget(self.steps_spin)
                
                params_layout.addWidget(QLabel("CFG:"))
                self.cfg_spin = QDoubleSpinBox()
                self.cfg_spin.setRange(1.0, 20.0)
                self.cfg_spin.setValue(7.0)
                self.cfg_spin.setSingleStep(0.5)
                params_layout.addWidget(self.cfg_spin)
                
                layout.addLayout(params_layout)
                
                # Generate button
                self.gen_button = QPushButton("Generate")
                self.gen_button.clicked.connect(self.generate)
                layout.addWidget(self.gen_button)
                
                # Progress
                self.progress_bar = QProgressBar()
                layout.addWidget(self.progress_bar)
                
                # Log
                layout.addWidget(QLabel("Log:"))
                self.log_text = QTextEdit()
                self.log_text.setReadOnly(True)
                layout.addWidget(self.log_text)
            
            def log(self, message: str):
                self.log_text.append(message)
            
            def generate(self):
                prompt = self.prompt_input.text()
                if not prompt:
                    self.log("Error: Please enter a prompt")
                    return
                
                self.gen_button.setEnabled(False)
                self.progress_bar.setValue(0)
                
                params = {
                    'prompt': prompt,
                    'steps': self.steps_spin.value(),
                    'cfg_scale': self.cfg_spin.value()
                }
                
                self.generation_thread = GenerationThread(self.client, params)
                self.generation_thread.progress_signal.connect(self.on_progress)
                self.generation_thread.finished_signal.connect(self.on_finished)
                self.generation_thread.start()
            
            def on_progress(self, progress: int, message: str):
                self.progress_bar.setValue(progress)
                self.log(message)
            
            def on_finished(self, result: dict):
                status = result['status']
                
                if status == 'completed':
                    self.log("Generation completed!")
                else:
                    self.log(f"Generation failed: {result.get('error')}")
                
                self.gen_button.setEnabled(True)
                self.progress_bar.setValue(100 if status == 'completed' else 0)
        
        app = QApplication([])
        window = GenesisPyQtWindow()
        window.show()
        return app
    
    except ImportError:
        print("PyQt5 not available")
        print("Install: pip install PyQt5")
        return None


# ==================== Example Usage ====================

def example_http_client():
    """Example using HTTP client"""
    print("=== HTTP Client Example ===")
    
    client = GenesisHTTPClient("http://localhost:5000")
    client.create_session("example")
    
    # Submit task
    task_id = client.submit_task('generate', {
        'prompt': 'a beautiful sunset',
        'steps': 20,
        'cfg_scale': 7.0
    })
    
    print(f"Task ID: {task_id}")
    
    # Wait for completion
    result = client.wait_for_task(task_id)
    print(f"Status: {result['status']}")
    print(f"Result: {result.get('result')}")


def example_websocket_client():
    """Example using WebSocket client"""
    print("=== WebSocket Client Example ===")
    
    try:
        client = GenesisWebSocketClient("http://localhost:5000")
        client.connect()
        
        # Generate
        client.generate_async('a beautiful landscape', steps=20, cfg_scale=7.0)
        
        # Keep connection alive
        import time
        time.sleep(30)
        
        client.disconnect()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Genesis Client Examples")
    print("="*60)
    print("1. HTTP Client (Simple)")
    print("2. WebSocket Client (Real-time)")
    print("3. Tkinter GUI")
    print("4. PyQt GUI")
    print("="*60)
    
    choice = input("Choose example (1-4): ")
    
    if choice == "1":
        example_http_client()
    elif choice == "2":
        example_websocket_client()
    elif choice == "3":
        root = create_tkinter_client()
        if root:
            root.mainloop()
    elif choice == "4":
        app = create_pyqt_client()
        if app:
            import sys
            sys.exit(app.exec_())
