"""
Genesis Server - Workflow API Routes
API endpoints for workflow execution
Author: eddy
"""

from flask import Blueprint, jsonify, request, current_app
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

workflow_bp = Blueprint('workflow', __name__)


@workflow_bp.route('/execute', methods=['POST'])
def execute_workflow():
    """
    Execute a workflow

    Request body:
    {
        "workflow": {...},  // ComfyUI workflow JSON
        "client_id": "...",  // Optional client ID
        "prompt_id": "..."  // Optional prompt ID
    }
    """
    try:
        server = current_app.genesis_server
        data = request.get_json()

        if not data or 'workflow' not in data:
            return jsonify({
                'success': False,
                'error': 'Workflow data is required'
            }), 400

        workflow = data['workflow']
        client_id = data.get('client_id', str(uuid.uuid4()))
        prompt_id = data.get('prompt_id', str(uuid.uuid4()))

        # Validate workflow
        if not isinstance(workflow, dict):
            return jsonify({
                'success': False,
                'error': 'Workflow must be a JSON object'
            }), 400

        # Get executor from server
        from ..execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(server.registry, server.engine)

        # Queue workflow execution
        task = {
            'prompt_id': prompt_id,
            'client_id': client_id,
            'workflow': workflow,
            'status': 'queued',
            'created_at': datetime.now().isoformat()
        }

        # Start execution (async in real implementation)
        try:
            result = executor.execute(workflow, prompt_id)
            task['status'] = 'completed'
            task['result'] = result
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            logger.error(f"Workflow execution failed: {e}")

        return jsonify({
            'success': True,
            'prompt_id': prompt_id,
            'task': task
        })

    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/validate', methods=['POST'])
def validate_workflow():
    """Validate a workflow without executing it"""
    try:
        server = current_app.genesis_server
        data = request.get_json()

        if not data or 'workflow' not in data:
            return jsonify({
                'success': False,
                'error': 'Workflow data is required'
            }), 400

        workflow = data['workflow']

        # Validate workflow structure
        from ..execution.executor import WorkflowExecutor
        executor = WorkflowExecutor(server.registry, server.engine)

        errors = executor.validate(workflow)

        return jsonify({
            'success': len(errors) == 0,
            'valid': len(errors) == 0,
            'errors': errors
        })

    except Exception as e:
        logger.error(f"Failed to validate workflow: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/status/<prompt_id>', methods=['GET'])
def get_workflow_status(prompt_id):
    """Get workflow execution status"""
    try:
        # In a real implementation, this would query a task queue
        return jsonify({
            'success': True,
            'prompt_id': prompt_id,
            'status': 'completed',
            'message': 'Status tracking not yet implemented'
        })

    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/queue', methods=['GET'])
def get_queue():
    """Get current execution queue"""
    try:
        # In a real implementation, this would return the task queue
        return jsonify({
            'success': True,
            'queue': [],
            'running': [],
            'pending': 0
        })

    except Exception as e:
        logger.error(f"Failed to get queue: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
