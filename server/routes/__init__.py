"""
Genesis Server Routes
API route registration
Author: eddy
"""

from flask import Flask


def register_routes(app: Flask):
    """Register all API routes"""
    from .nodes import nodes_bp
    from .workflow import workflow_bp
    from .system import system_bp

    app.register_blueprint(nodes_bp, url_prefix='/api/nodes')
    app.register_blueprint(workflow_bp, url_prefix='/api/workflow')
    app.register_blueprint(system_bp, url_prefix='/api/system')
