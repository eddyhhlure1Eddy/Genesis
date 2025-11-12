"""
Genesis Server - Nodes API Routes
API endpoints for node information
Author: eddy
"""

from flask import Blueprint, jsonify, request, current_app
import logging

logger = logging.getLogger(__name__)

nodes_bp = Blueprint('nodes', __name__)


@nodes_bp.route('/', methods=['GET'])
def list_nodes():
    """
    Get list of all registered nodes

    Query parameters:
    - category: Filter by category (optional)
    - search: Search query (optional)
    """
    try:
        server = current_app.genesis_server
        registry = server.registry

        # Get query parameters
        category = request.args.get('category')
        search = request.args.get('search')

        if search:
            # Search nodes
            node_names = registry.search(search)
        elif category:
            # Filter by category
            nodes_by_cat = registry.list_by_category(category)
            node_names = nodes_by_cat.get(category, [])
        else:
            # Get all nodes
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
                    'description': node_info.get('description', ''),
                })

        return jsonify({
            'success': True,
            'nodes': nodes,
            'total': len(nodes)
        })

    except Exception as e:
        logger.error(f"Failed to list nodes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@nodes_bp.route('/<node_name>', methods=['GET'])
def get_node_info(node_name):
    """Get detailed information for a specific node"""
    try:
        server = current_app.genesis_server
        registry = server.registry

        node_info = registry.get_info(node_name)

        if not node_info:
            return jsonify({
                'success': False,
                'error': f'Node not found: {node_name}'
            }), 404

        return jsonify({
            'success': True,
            'node': node_info
        })

    except Exception as e:
        logger.error(f"Failed to get node info for {node_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@nodes_bp.route('/categories', methods=['GET'])
def list_categories():
    """Get list of all node categories"""
    try:
        server = current_app.genesis_server
        registry = server.registry

        categories = registry.get_categories()
        nodes_by_category = registry.list_by_category()

        # Build response with counts
        category_info = []
        for cat in categories:
            category_info.append({
                'name': cat,
                'node_count': len(nodes_by_category.get(cat, []))
            })

        return jsonify({
            'success': True,
            'categories': category_info,
            'total': len(categories)
        })

    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@nodes_bp.route('/search', methods=['POST'])
def search_nodes():
    """Search nodes"""
    try:
        server = current_app.genesis_server
        registry = server.registry

        data = request.get_json()
        query = data.get('query', '')
        search_in = data.get('search_in', ['name', 'category', 'description'])

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400

        node_names = registry.search(query, search_in=search_in)

        # Build response
        nodes = []
        for name in node_names:
            node_info = registry.get_info(name)
            if node_info:
                nodes.append({
                    'name': name,
                    'display_name': node_info.get('display_name', name),
                    'category': node_info.get('category', 'misc'),
                    'description': node_info.get('description', ''),
                })

        return jsonify({
            'success': True,
            'query': query,
            'nodes': nodes,
            'total': len(nodes)
        })

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@nodes_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get node registry statistics"""
    try:
        server = current_app.genesis_server
        registry = server.registry

        stats = registry.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        })

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
