#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis API Server - 真实图像生成版本
使用 Stable Diffusion 生成真实图像
"""

import sys
import os
from pathlib import Path
import base64
from io import BytesIO

# 禁用所有可选的注意力优化库以避免兼容性问题
os.environ['DIFFUSERS_DISABLE_FLASH_ATTENTION'] = '1'
os.environ['DISABLE_SAGE_ATTENTION'] = '1'
os.environ['ATTN_BACKEND'] = 'pytorch'  # 强制使用 PyTorch 原生实现

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("=" * 70)
    print("错误: Flask 未安装")
    print("=" * 70)
    print()
    print("请安装必要的依赖:")
    print("  pip install flask flask-cors")
    print()
    sys.exit(1)

try:
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    DIFFUSERS_AVAILABLE = True
    print("✓ Diffusers 已安装")
except ImportError as e:
    print("=" * 70)
    print("警告: Diffusers 未安装或导入失败，将使用模拟模式")
    print("=" * 70)
    print()
    print(f"错误信息: {e}")
    print()
    print("如需真实生成图像，请安装:")
    print("  pip install torch torchvision")
    print("  pip install diffusers transformers accelerate")
    print()
    print("如果遇到 flash_attn 错误，请卸载它:")
    print("  pip uninstall flash-attn -y")
    print()
    DIFFUSERS_AVAILABLE = False

import uuid
import time
import threading
from queue import Queue, Empty
from datetime import datetime

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 任务存储
tasks = {}
task_queue = Queue()

# 任务状态常量
class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# 任务类
class Task:
    def __init__(self, task_id, task_type, params):
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
    
    def to_dict(self):
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

# 全局生成器
class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if DIFFUSERS_AVAILABLE else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32 if DIFFUSERS_AVAILABLE else None
        self.model_loaded = False
        
    def load_model(self, model_id="runwayml/stable-diffusion-v1-5"):
        """加载模型"""
        if not DIFFUSERS_AVAILABLE:
            print("⚠️  Diffusers 未安装，跳过模型加载")
            return False
            
        if self.model_loaded:
            print("✓ 模型已加载")
            return True
            
        try:
            print(f"📥 加载模型: {model_id}")
            print(f"   设备: {self.device}")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # 使用 Euler 调度器
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # 启用优化
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✓ 启用 xformers 优化")
                except:
                    print("ℹ xformers 未安装，使用默认注意力")
            
            self.model_loaded = True
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(self, params, progress_callback=None):
        """生成图像"""
        if not DIFFUSERS_AVAILABLE:
            # 返回模拟数据
            return self._generate_mock(params)
        
        if not self.model_loaded:
            if not self.load_model():
                raise Exception("模型加载失败")
        
        try:
            prompt = params.get('prompt', '')
            negative_prompt = params.get('negative_prompt', '')
            width = params.get('width', 512)
            height = params.get('height', 512)
            steps = params.get('steps', 20)
            cfg_scale = params.get('cfg_scale', 7.0)
            seed = params.get('seed')
            
            # 设置随机种子
            if seed is None or seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            
            print(f"🎨 开始生成图像...")
            print(f"   提示词: {prompt[:50]}...")
            print(f"   尺寸: {width}x{height}")
            print(f"   步数: {steps}")
            
            # 生成图像
            def callback(step, timestep, latents):
                if progress_callback:
                    progress = int((step / steps) * 80) + 10  # 10-90%
                    progress_callback(progress)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                callback=callback,
                callback_steps=1
            )
            
            image = result.images[0]
            
            # 转换为 base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_data_url = f"data:image/png;base64,{img_str}"
            
            print("✅ 图像生成成功")
            
            return {
                'success': True,
                'image': img_data_url,
                'seed': seed,
                'prompt': prompt,
                'width': width,
                'height': height,
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_mock(self, params):
        """生成模拟数据"""
        import random
        time.sleep(2)  # 模拟生成时间
        
        # 1x1 透明 PNG
        mock_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        return {
            'success': True,
            'image': mock_image,
            'seed': params.get('seed') or random.randint(0, 2147483647),
            'prompt': params.get('prompt', ''),
            'width': params.get('width', 512),
            'height': params.get('height', 512),
            'note': '⚠️ 这是模拟数据，请安装 diffusers 以生成真实图像'
        }

# 创建全局生成器
generator = ImageGenerator()

# 工作线程
worker_running = False
worker_thread = None

def worker_loop():
    """处理任务队列"""
    global worker_running
    print("🔄 Worker thread started")
    
    while worker_running:
        try:
            task_id = task_queue.get(timeout=1.0)
            task = tasks.get(task_id)
            
            if not task:
                continue
            
            # 更新状态为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.progress = 5
            
            print(f"📝 Processing task: {task_id}")
            
            try:
                # 进度回调
                def progress_callback(progress):
                    task.progress = progress
                
                # 执行生成
                result = generator.generate(task.params, progress_callback)
                
                # 完成
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 100
                task.completed_at = datetime.now()
                
                print(f"✅ Task completed: {task_id}")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                print(f"❌ Task failed: {task_id} - {e}")
                
        except Empty:
            continue
        except Exception as e:
            print(f"❌ Worker error: {e}")
    
    print("🛑 Worker thread stopped")

# API 路由
@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'diffusers_available': DIFFUSERS_AVAILABLE,
        'model_loaded': generator.model_loaded if DIFFUSERS_AVAILABLE else False,
        'device': generator.device if DIFFUSERS_AVAILABLE else 'N/A',
        'tasks_pending': task_queue.qsize(),
        'tasks_total': len(tasks)
    })

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """创建会话"""
    data = request.get_json() or {}
    session_id = str(uuid.uuid4())
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'session': {
            'session_id': session_id,
            'client_type': data.get('client_type', 'web'),
            'created_at': datetime.now().isoformat()
        }
    })

@app.route('/api/task/submit', methods=['POST'])
def submit_task():
    """提交任务"""
    try:
        data = request.get_json() or {}
        
        task_type = data.get('task_type', 'generate')
        params = data.get('params', {})
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task = Task(task_id, task_type, params)
        
        # 保存任务
        tasks[task_id] = task
        task_queue.put(task_id)
        
        print(f"📥 Task submitted: {task_id}")
        print(f"   Prompt: {params.get('prompt', 'N/A')[:50]}...")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'task': task.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """获取任务状态"""
    task = tasks.get(task_id)
    
    if not task:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404
    
    return jsonify({
        'success': True,
        'task': task.to_dict()
    })

@app.route('/api/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """取消任务"""
    task = tasks.get(task_id)
    
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
    
    task.status = TaskStatus.FAILED
    task.error = 'Cancelled by user'
    task.completed_at = datetime.now()
    
    return jsonify({
        'success': True,
        'task': task.to_dict()
    })

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    return jsonify({
        'success': True,
        'tasks': [task.to_dict() for task in tasks.values()],
        'count': len(tasks)
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    return jsonify({
        'success': True,
        'models': {
            'checkpoints': ['runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1'],
            'loras': [],
            'vae': []
        }
    })

@app.route('/api/device', methods=['GET'])
def device_info():
    """获取设备信息"""
    if not DIFFUSERS_AVAILABLE:
        return jsonify({
            'success': True,
            'device': {
                'device': 'N/A',
                'note': 'Diffusers not installed'
            }
        })
    
    info = {'device': generator.device}
    
    if generator.device == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['memory_total'] = props.total_memory
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
    
    return jsonify({
        'success': True,
        'device': info
    })

@app.route('/')
def index():
    """API 首页"""
    return jsonify({
        'name': 'Genesis API Server (Real Generation)',
        'version': '1.0.0',
        'status': 'running',
        'diffusers_available': DIFFUSERS_AVAILABLE,
        'model_loaded': generator.model_loaded if DIFFUSERS_AVAILABLE else False,
        'endpoints': {
            'GET  /health': 'Health check',
            'POST /api/session/create': 'Create session',
            'POST /api/task/submit': 'Submit task',
            'GET  /api/task/<id>': 'Get task status',
            'POST /api/task/<id>/cancel': 'Cancel task',
            'GET  /api/tasks': 'List all tasks',
            'GET  /api/models': 'List models',
            'GET  /api/device': 'Device info'
        }
    })

def start_worker():
    """启动工作线程"""
    global worker_running, worker_thread
    
    worker_running = True
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

def stop_worker():
    """停止工作线程"""
    global worker_running
    
    worker_running = False
    if worker_thread:
        worker_thread.join(timeout=5.0)

def main():
    """主函数"""
    print("=" * 70)
    print("Genesis API Server - Real Image Generation")
    print("=" * 70)
    print()
    
    if DIFFUSERS_AVAILABLE:
        print("✅ Diffusers 可用")
        print(f"   设备: {generator.device}")
        if generator.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print()
        print("📥 首次生成时会自动下载模型（约 4GB）")
        print("   模型会缓存到: ~/.cache/huggingface/")
    else:
        print("⚠️  Diffusers 未安装，使用模拟模式")
        print()
        print("如需真实生成，请安装:")
        print("  pip install torch torchvision")
        print("  pip install diffusers transformers accelerate")
    
    print()
    print("🚀 Starting server...")
    print()
    print("📡 Server URL: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("📚 API docs: http://localhost:5000/")
    print()
    print("=" * 70)
    print()
    
    # 启动工作线程
    start_worker()
    
    try:
        # 启动 Flask 服务器
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        stop_worker()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
