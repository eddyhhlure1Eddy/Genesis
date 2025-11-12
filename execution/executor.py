"""
Genesis Executor
Executor - Responsible for executing generation tasks
Author: eddy
"""

import logging
from typing import Dict, Any, Optional, Callable
import time
from pathlib import Path


class Executor:
    """
    Executor
    
    Responsible for executing image generation and other tasks
    """
    
    def __init__(self, config, device):
        """
        Initialize executor

        Args:
            config: Genesis configuration
            device: Computing device
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger('Genesis.Executor')

        self.is_executing = False
        self.current_task = None
        self.sd_pipeline = None
        self.progress_callback = None
        
    def set_progress_callback(self, callback: Optional[Callable[[int, int], None]]):
        """
        Set progress callback

        Args:
            callback: Callback function(current_step, total_steps)
        """
        self.progress_callback = callback

    def initialize_pipeline(self, model_path: Optional[str] = None):
        """
        Initialize Stable Diffusion pipeline

        Args:
            model_path: Optional model path, uses pretrained if None
        """
        if self.sd_pipeline is not None:
            self.logger.warning("Pipeline already initialized")
            return

        from ..models.sd_pipeline import StableDiffusionPipeline

        self.logger.info("Initializing SD Pipeline...")
        self.sd_pipeline = StableDiffusionPipeline(self.device)

        if model_path:
            self.logger.info(f"Loading model from: {model_path}")
        else:
            self.logger.info("Loading pretrained model from HuggingFace...")
            self.sd_pipeline.load_from_pretrained()

        self.logger.info("✓ Pipeline initialized")

    def execute_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute image generation

        Args:
            params: Generation parameters

        Returns:
            Generation result
        """
        if self.is_executing:
            raise RuntimeError("Another task is currently executing")

        self.is_executing = True
        start_time = time.time()

        try:
            if self.sd_pipeline is None:
                self.initialize_pipeline()

            self.logger.info("Starting generation...")

            prompt = params.get('prompt', '')
            negative_prompt = params.get('negative_prompt', '')
            width = params.get('width', 512)
            height = params.get('height', 512)
            steps = params.get('steps', 20)
            cfg_scale = params.get('cfg_scale', 7.0)
            seed = params.get('seed')

            self.logger.info(f"Prompt: {prompt[:100]}...")
            self.logger.info(f"Steps: {steps}, CFG: {cfg_scale}, Size: {width}x{height}")

            def progress_wrapper(step, total):
                if self.progress_callback:
                    self.progress_callback(step, total)
                self.logger.debug(f"Progress: {step}/{total}")

            image = self.sd_pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                callback=progress_wrapper
            )

            output_dir = Path(self.config.output_dir) if hasattr(self.config, 'output_dir') else Path('outputs')
            output_dir.mkdir(exist_ok=True)

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"genesis_{timestamp}.png"
            output_path = output_dir / filename

            image.save(str(output_path))
            self.logger.info(f"✓ Image saved: {output_path}")

            result = {
                'success': True,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'execution_time': time.time() - start_time,
                'image_path': str(output_path),
                'image': image,
                'status': 'completed'
            }

            self.logger.info(f"✓ Generation completed in {result['execution_time']:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'status': 'failed'
            }
        finally:
            self.is_executing = False
    
    def execute_pipeline(self, pipeline) -> Dict[str, Any]:
        """
        Execute Pipeline
        
        Args:
            pipeline: Pipeline object
            
        Returns:
            Execution result
        """
        if self.is_executing:
            raise RuntimeError("Another task is currently executing")
        
        self.is_executing = True
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing pipeline: {pipeline.name}")
            
            # Validate Pipeline
            errors = pipeline.validate()
            if errors:
                raise ValueError(f"Pipeline validation failed: {errors}")
            
            # TODO: Actual Pipeline execution logic
            
            result = {
                'success': True,
                'pipeline_name': pipeline.name,
                'execution_time': time.time() - start_time,
                'status': 'completed'
            }
            
            self.logger.info(f"✓ Pipeline executed in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'status': 'failed'
            }
        finally:
            self.is_executing = False
    
    def cancel(self):
        """Cancel current execution"""
        if self.is_executing:
            self.logger.warning("Cancelling current execution...")
            # TODO: Implement cancel logic
            self.is_executing = False
