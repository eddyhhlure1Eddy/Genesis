"""
Genesis Advanced Server
High-performance Flask server with WebSocket support
Supports: Web UI, Tkinter, PyQt, and any other clients
Author: eddy
"""

import os
import json
import uuid
import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit, join_room, leave_room
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Error: Flask/SocketIO not installed")
    print("Install: pip install flask flask-cors flask-socketio python-socketio")

from ..core.engine import GenesisEngine
from ..core.config import GenesisConfig
from ..core.pipeline import Pipeline


class TaskStatus:
    """Task status constants"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Task representation"""
    
    def __init__(self, task_id: str, task_type: str, params: Dict[str, Any]):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'params': self.params,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class SessionManager:
    """Manage client sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_session(self, session_id: str, client_type: str = "unknown") -> Dict[str, Any]:
        """Create new session"""
        with self.lock:
            session = {
                'session_id': session_id,
                'client_type': client_type,
                'created_at': datetime.now(),
                'last_active': datetime.now(),
                'tasks': []
            }
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def update_activity(self, session_id: str):
        """Update last activity time"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_active'] = datetime.now()
    
    def add_task_to_session(self, session_id: str, task_id: str):
        """Add task to session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['tasks'].append(task_id)
    
    def remove_session(self, session_id: str):
        """Remove session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]


class TaskQueue:
    """Task queue manager"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.queue = Queue()
        self.lock = threading.Lock()
    
    def add_task(self, task: Task):
        """Add task to queue"""
        with self.lock:
            self.tasks[task.task_id] = task
            self.queue.put(task.task_id)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_next_task(self, timeout: float = 1.0) -> Optional[Task]:
        """Get next task from queue"""
        try:
            task_id = self.queue.get(timeout=timeout)
            with self.lock:
                return self.tasks.get(task_id)
        except Empty:
            return None
    
    def update_task_status(self, task_id: str, status: str, **kwargs):
        """Update task status"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                
                if status == TaskStatus.RUNNING and task.started_at is None:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
                
                for key, value in kwargs.items():
                    setattr(task, key, value)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        with self.lock:
            return [task.to_dict() for task in self.tasks.values()]


class GenesisAdvancedServer:
    """
    Advanced Flask server with WebSocket support
    
    Features:
    - RESTful API for HTTP clients
    - WebSocket for real-time communication
    - Task queue with progress tracking
    - Session management
    - Multi-client support (Web, Tkinter, PyQt, etc.)
    - Event broadcasting
    """
    
    def __init__(
        self,
        engine: Optional[GenesisEngine] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False
    ):
        """
        Initialize advanced server
        
        Args:
            engine: Genesis engine instance
            host: Server host
            port: Server port
            debug: Debug mode
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask/SocketIO required")
        
        self.engine = engine or GenesisEngine()
        self.host = host
        self.port = port
        self.debug = debug
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'genesis-secret-key'
        CORS(self.app)
        
        # WebSocket support
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=debug,
            engineio_logger=debug
        )
        
        # Task management
        self.task_queue = TaskQueue()
        self.session_manager = SessionManager()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Worker thread
        self.worker_thread = None
        self.worker_running = False
        
        # Setup routes
        self._setup_http_routes()
        self._setup_websocket_events()
        
    def _safe_get_param(self, data: Dict[str, Any], key: str, default: Any, param_type: type = None):
        """
        Safely get parameter with type conversion and default value
        High tolerance for different input types
        """
        try:
            value = data.get(key, default)
            
            if value is None:
                return default
            
            # Type conversion with tolerance
            if param_type:
                if param_type == int:
                    return int(float(value)) if value != "" else default
                elif param_type == float:
                    return float(value) if value != "" else default
                elif param_type == bool:
                    if isinstance(value, str):
                        return value.lower() in ['true', '1', 'yes', 'on']
                    return bool(value)
                elif param_type == str:
                    return str(value)
            
            return value
        except Exception as e:
            self.app.logger.warning(f"Failed to convert {key}={value}: {e}, using default: {default}")
            return default
    
    def _validate_and_normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize parameters with high tolerance
        Auto-fix common issues
        """
        normalized = {}
        
        # Prompt (required but flexible)
        normalized['prompt'] = str(params.get('prompt', '')).strip()
        
        # Optional parameters with defaults
        normalized['negative_prompt'] = str(params.get('negative_prompt', '')).strip()
        
        # Numeric parameters with range validation
        width = self._safe_get_param(params, 'width', 512, int)
        normalized['width'] = max(64, min(4096, width))  # Clamp to valid range
        
        height = self._safe_get_param(params, 'height', 512, int)
        normalized['height'] = max(64, min(4096, height))
        
        steps = self._safe_get_param(params, 'steps', 20, int)
        normalized['steps'] = max(1, min(200, steps))
        
        cfg_scale = self._safe_get_param(params, 'cfg_scale', 7.0, float)
        normalized['cfg_scale'] = max(1.0, min(30.0, cfg_scale))
        
        # Seed handling (accept -1, None, "random", etc.)
        seed = params.get('seed')
        if seed in [None, -1, '', 'random', 'Random']:
            normalized['seed'] = None
        else:
            try:
                normalized['seed'] = int(seed)
            except:
                normalized['seed'] = None
        
        # Sampler and scheduler (case-insensitive, with fallback)
        sampler = str(params.get('sampler', 'euler')).lower().strip()
        normalized['sampler'] = sampler if sampler else 'euler'
        
        scheduler = str(params.get('scheduler', 'normal')).lower().strip()
        normalized['scheduler'] = scheduler if scheduler else 'normal'
        
        # Batch size
        batch_size = self._safe_get_param(params, 'batch_size', 1, int)
        normalized['batch_size'] = max(1, min(10, batch_size))
        
        # Denoising strength
        denoise = self._safe_get_param(params, 'denoise', 1.0, float)
        normalized['denoise'] = max(0.0, min(1.0, denoise))
        
        return normalized
    
    def _setup_http_routes(self):
        """Setup HTTP routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check"""
            return jsonify({
                'status': 'healthy',
                'initialized': self.engine._initialized,
                'tasks_pending': self.task_queue.queue.qsize(),
                'sessions_active': len(self.session_manager.sessions)
            })
        
        @self.app.route('/api/session/create', methods=['POST'])
        def create_session():
            """Create new session"""
            data = request.get_json() or {}
            session_id = str(uuid.uuid4())
            client_type = data.get('client_type', 'unknown')
            
            session = self.session_manager.create_session(session_id, client_type)
            
            return jsonify({
                'success': True,
                'session': session,
                'session_id': session_id
            })
        
        @self.app.route('/api/task/submit', methods=['POST'])
        def submit_task():
            """Submit new task with high tolerance"""
            try:
                data = request.get_json() or {}
                
                task_type = data.get('task_type', 'generate')
                params = data.get('params', {})
                session_id = data.get('session_id')
                
                # Validate and normalize parameters
                if task_type == 'generate':
                    params = self._validate_and_normalize_params(params)
                
                # Create task
                task_id = str(uuid.uuid4())
                task = Task(task_id, task_type, params)
                
                # Add to queue
                self.task_queue.add_task(task)
                
                # Add to session if provided
                if session_id:
                    self.session_manager.add_task_to_session(session_id, task_id)
                
                return jsonify({
                    'success': True,
                    'task_id': task_id,
                    'task': task.to_dict()
                })
            
            except Exception as e:
                self.app.logger.error(f"Error submitting task: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'hint': 'Check your parameters and try again'
                }), 400
        
        @self.app.route('/api/task/<task_id>', methods=['GET'])
        def get_task(task_id):
            """Get task status"""
            task = self.task_queue.get_task(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            return jsonify({
                'success': True,
                'task': task.to_dict()
            })
        
        @self.app.route('/api/task/<task_id>/cancel', methods=['POST'])
        def cancel_task(task_id):
            """Cancel task"""
            task = self.task_queue.get_task(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return jsonify({
                    'success': False,
                    'error': 'Task already finished'
                }), 400
            
            self.task_queue.update_task_status(
                task_id,
                TaskStatus.CANCELLED
            )
            
            return jsonify({
                'success': True,
                'task': task.to_dict()
            })
        
        @self.app.route('/api/tasks', methods=['GET'])
        def list_tasks():
            """List all tasks"""
            tasks = self.task_queue.get_all_tasks()
            return jsonify({
                'success': True,
                'tasks': tasks,
                'count': len(tasks)
            })
        
        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            """List available models"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                models = self.engine.get_available_models()
                return jsonify({
                    'success': True,
                    'models': models
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/device', methods=['GET'])
        def device_info():
            """Get device information"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                info = self.engine.get_device_info()
                return jsonify({
                    'success': True,
                    'device': info
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/workflow/execute', methods=['POST'])
        def execute_workflow():
            """Execute ComfyUI workflow"""
            try:
                from ..core.workflow_converter import ComfyUIWorkflowConverter
                
                data = request.get_json() or {}
                workflow = data.get('workflow')
                
                if not workflow:
                    return jsonify({
                        'success': False,
                        'error': 'No workflow provided'
                    }), 400
                
                converter = ComfyUIWorkflowConverter()
                
                if not converter.parse_workflow(workflow):
                    return jsonify({
                        'success': False,
                        'error': 'Failed to parse workflow'
                    }), 400
                
                result = converter.execute_workflow()
                
                return jsonify(result)
                
            except Exception as e:
                self.app.logger.error(f"Workflow execution error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/workflow/parse', methods=['POST'])
        def parse_workflow():
            """Parse ComfyUI workflow"""
            try:
                from ..core.workflow_converter import ComfyUIWorkflowConverter
                
                data = request.get_json() or {}
                workflow = data.get('workflow')
                
                if not workflow:
                    return jsonify({
                        'success': False,
                        'error': 'No workflow provided'
                    }), 400
                
                converter = ComfyUIWorkflowConverter()
                
                if converter.parse_workflow(workflow):
                    info = converter.get_workflow_info()
                    genesis_workflow = converter.convert_to_genesis(workflow)
                    
                    return jsonify({
                        'success': True,
                        'info': info,
                        'genesis_workflow': genesis_workflow.get('workflow')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to parse workflow'
                    }), 400
                    
            except Exception as e:
                self.app.logger.error(f"Workflow parse error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        web_dir = Path(__file__).parent.parent / 'web'
        
        @self.app.route('/')
        def serve_index():
            """Serve web UI index"""
            if web_dir.exists():
                return send_from_directory(web_dir, 'index.html')
            return jsonify({
                'name': 'Genesis Advanced Server',
                'version': '0.1.0',
                'author': 'eddy',
                'features': [
                    'RESTful API',
                    'WebSocket support',
                    'Task queue',
                    'Progress tracking',
                    'Multi-client support',
                    'Session management'
                ],
                'endpoints': {
                    'http': self._get_http_endpoints(),
                    'websocket': self._get_websocket_events()
                }
            })
        
        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """Serve static files"""
            if web_dir.exists():
                return send_from_directory(web_dir, filename)
            return jsonify({'error': 'File not found'}), 404
    
    def _setup_websocket_events(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Client connected"""
            print(f"Client connected: {request.sid}")
            emit('connected', {
                'message': 'Connected to Genesis Server',
                'sid': request.sid
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Client disconnected"""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('join')
        def handle_join(data):
            """Join room"""
            room = data.get('room', 'default')
            join_room(room)
            emit('joined', {
                'room': room,
                'message': f'Joined room: {room}'
            })
        
        @self.socketio.on('leave')
        def handle_leave(data):
            """Leave room"""
            room = data.get('room', 'default')
            leave_room(room)
            emit('left', {
                'room': room,
                'message': f'Left room: {room}'
            })
        
        @self.socketio.on('submit_task')
        def handle_submit_task(data):
            """Submit task via WebSocket"""
            task_type = data.get('task_type', 'generate')
            params = data.get('params', {})
            
            # Create task
            task_id = str(uuid.uuid4())
            task = Task(task_id, task_type, params)
            
            # Add to queue
            self.task_queue.add_task(task)
            
            emit('task_submitted', {
                'task_id': task_id,
                'task': task.to_dict()
            })
        
        @self.socketio.on('get_task_status')
        def handle_get_task_status(data):
            """Get task status via WebSocket"""
            task_id = data.get('task_id')
            task = self.task_queue.get_task(task_id)
            
            if task:
                emit('task_status', {
                    'task': task.to_dict()
                })
            else:
                emit('error', {
                    'message': 'Task not found'
                })
    
    def _get_http_endpoints(self) -> List[str]:
        """Get list of HTTP endpoints"""
        return [
            'GET  /',
            'GET  /health',
            'POST /api/session/create',
            'POST /api/task/submit',
            'GET  /api/task/<task_id>',
            'POST /api/task/<task_id>/cancel',
            'GET  /api/tasks',
            'GET  /api/models',
            'GET  /api/device',
            'POST /api/workflow/execute',
            'POST /api/workflow/parse'
        ]
    
    def _get_websocket_events(self) -> List[str]:
        """Get list of WebSocket events"""
        return [
            'connect',
            'disconnect',
            'join',
            'leave',
            'submit_task',
            'get_task_status'
        ]
    
    def emit_event(self, event: str, data: Dict[str, Any], room: Optional[str] = None):
        """Emit event to clients"""
        if room:
            self.socketio.emit(event, data, room=room)
        else:
            self.socketio.emit(event, data, broadcast=True)
    
    def emit_progress(self, task_id: str, progress: int, message: str = ""):
        """Emit progress update"""
        self.emit_event('progress', {
            'task_id': task_id,
            'progress': progress,
            'message': message
        })
    
    def emit_task_complete(self, task_id: str, result: Any):
        """Emit task completion"""
        self.emit_event('task_complete', {
            'task_id': task_id,
            'result': result
        })
    
    def emit_task_error(self, task_id: str, error: str):
        """Emit task error"""
        self.emit_event('task_error', {
            'task_id': task_id,
            'error': error
        })
    
    def _worker_loop(self):
        """Worker loop to process tasks"""
        print("Worker thread started")
        
        while self.worker_running:
            task = self.task_queue.get_next_task(timeout=1.0)
            
            if task is None:
                continue
            
            try:
                # Update status
                self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.RUNNING
                )
                self.emit_progress(task.task_id, 0, "Starting task")
                
                # Initialize engine if needed
                if not self.engine._initialized:
                    self.engine.initialize()
                    self.emit_progress(task.task_id, 10, "Engine initialized")
                
                # Execute task
                if task.task_type == 'generate':
                    result = self._execute_generate_task(task)
                elif task.task_type == 'pipeline':
                    result = self._execute_pipeline_task(task)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                # Update status
                self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.COMPLETED,
                    result=result,
                    progress=100
                )
                self.emit_task_complete(task.task_id, result)
                
            except Exception as e:
                # Update status
                self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error=str(e)
                )
                self.emit_task_error(task.task_id, str(e))
        
        print("Worker thread stopped")
    
    def _execute_generate_task(self, task: Task) -> Dict[str, Any]:
        """Execute generation task"""
        params = task.params
        
        # Emit progress updates
        self.emit_progress(task.task_id, 20, "Preparing generation")
        
        # Generate
        result = self.engine.generate(**params)
        
        self.emit_progress(task.task_id, 90, "Finalizing")
        
        return result
    
    def _execute_pipeline_task(self, task: Task) -> Dict[str, Any]:
        """Execute pipeline task"""
        params = task.params
        
        # Create pipeline
        pipeline = Pipeline.from_dict(params)
        
        self.emit_progress(task.task_id, 20, "Executing pipeline")
        
        # Execute
        result = self.engine.executor.execute_pipeline(pipeline)
        
        return result
    
    def start_worker(self):
        """Start worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            self.worker_thread.start()
    
    def stop_worker(self):
        """Stop worker thread"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def run(self):
        """Start server"""
        print("="*60)
        print(f"Genesis Advanced Server")
        print(f"Listening on: http://{self.host}:{self.port}")
        print(f"WebSocket enabled")
        print("="*60)
        
        # Start worker
        self.start_worker()
        
        try:
            # Run server
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
        finally:
            self.stop_worker()


def create_advanced_server(
    config: Optional[GenesisConfig] = None,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False
) -> GenesisAdvancedServer:
    """
    Create advanced server
    
    Args:
        config: Engine configuration
        host: Server host
        port: Server port
        debug: Debug mode
        
    Returns:
        GenesisAdvancedServer instance
    """
    engine = GenesisEngine(config or GenesisConfig())
    server = GenesisAdvancedServer(engine, host, port, debug)
    return server


if __name__ == "__main__":
    server = create_advanced_server(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    server.run()
