"""
Genesis WebSocket Event Handlers
Real-time communication with clients
Author: eddy
"""

import logging
from flask_socketio import emit, join_room, leave_room
from flask import request

logger = logging.getLogger(__name__)


def register_handlers(socketio, server):
    """
    Register WebSocket event handlers

    Args:
        socketio: SocketIO instance
        server: GenesisServer instance
    """

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        client_id = request.sid
        logger.info(f"Client connected: {client_id}")

        emit('connected', {
            'client_id': client_id,
            'message': 'Connected to Genesis Server',
            'nodes_loaded': len(server.registry)
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        client_id = request.sid
        logger.info(f"Client disconnected: {client_id}")

    @socketio.on('get_nodes')
    def handle_get_nodes(data):
        """
        Get list of nodes

        Args:
            data: {category: optional}
        """
        try:
            category = data.get('category') if data else None
            registry = server.registry

            if category:
                nodes_by_cat = registry.list_by_category(category)
                node_names = nodes_by_cat.get(category, [])
            else:
                node_names = registry.list_all()

            # Build response
            nodes = []
            for name in node_names:
                node_info = registry.get_info(name)
                if node_info:
                    nodes.append({
                        'name': name,
                        'display_name': node_info.get('display_name', name),
                        'category': node_info.get('category', 'misc'),
                    })

            emit('nodes_list', {
                'success': True,
                'nodes': nodes,
                'total': len(nodes)
            })

        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            emit('error', {'error': str(e)})

    @socketio.on('get_node_info')
    def handle_get_node_info(data):
        """
        Get detailed node information

        Args:
            data: {node_name: string}
        """
        try:
            node_name = data.get('node_name')
            if not node_name:
                emit('error', {'error': 'node_name is required'})
                return

            registry = server.registry
            node_info = registry.get_info(node_name)

            if not node_info:
                emit('error', {'error': f'Node not found: {node_name}'})
                return

            emit('node_info', {
                'success': True,
                'node': node_info
            })

        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            emit('error', {'error': str(e)})

    @socketio.on('execute_workflow')
    def handle_execute_workflow(data):
        """
        Execute a workflow

        Args:
            data: {
                workflow: dict,
                prompt_id: optional string
            }
        """
        try:
            workflow = data.get('workflow')
            prompt_id = data.get('prompt_id', 'default')

            if not workflow:
                emit('error', {'error': 'workflow is required'})
                return

            # Join room for this prompt
            join_room(prompt_id)

            # Emit status
            emit('workflow_status', {
                'prompt_id': prompt_id,
                'status': 'started'
            }, room=prompt_id)

            # Execute workflow
            from ..execution.executor import WorkflowExecutor
            executor = WorkflowExecutor(server.registry, server.engine)

            # This should be async in production
            result = executor.execute(workflow, prompt_id)

            # Emit result
            if result['success']:
                emit('workflow_completed', {
                    'prompt_id': prompt_id,
                    'result': result
                }, room=prompt_id)
            else:
                emit('workflow_failed', {
                    'prompt_id': prompt_id,
                    'error': result.get('error')
                }, room=prompt_id)

        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            emit('workflow_failed', {
                'prompt_id': data.get('prompt_id', 'default'),
                'error': str(e)
            })

    @socketio.on('join_room')
    def handle_join_room(data):
        """Join a room for receiving updates"""
        room = data.get('room')
        if room:
            join_room(room)
            emit('joined_room', {'room': room})

    @socketio.on('leave_room')
    def handle_leave_room(data):
        """Leave a room"""
        room = data.get('room')
        if room:
            leave_room(room)
            emit('left_room', {'room': room})

    @socketio.on('ping')
    def handle_ping():
        """Handle ping request"""
        emit('pong', {'message': 'pong'})

    logger.info("WebSocket handlers registered")
