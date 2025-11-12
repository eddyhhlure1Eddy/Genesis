"""
ComfyUI Workflow Converter
Parse and execute ComfyUI workflow JSON
Author: eddy
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from .interfaces import COMFYUI_NODE_REGISTRY
from .custom_node_loader import custom_node_loader


class WorkflowNode:
    """Workflow node representation"""
    
    def __init__(self, node_id: str, class_type: str, inputs: Dict[str, Any]):
        self.node_id = node_id
        self.class_type = class_type
        self.inputs = inputs
        self.outputs = {}
        self.executed = False
    
    def __repr__(self):
        return f"WorkflowNode(id={self.node_id}, type={self.class_type})"


class ComfyUIWorkflowConverter:
    """
    Convert and execute ComfyUI workflow
    
    Supports:
    - Parse ComfyUI JSON workflow
    - Dynamic node creation
    - Connection resolution
    - Execution order calculation
    - Result extraction
    """
    
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.execution_order: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """
        Load workflow from JSON file
        
        Args:
            workflow_path: Path to workflow JSON
            
        Returns:
            Workflow dictionary
        """
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        return workflow
    
    def parse_workflow(self, workflow: Dict[str, Any]) -> bool:
        """
        Parse ComfyUI workflow format
        
        Args:
            workflow: Workflow dictionary
            
        Returns:
            True if parsed successfully
        """
        try:
            self.nodes.clear()
            self.execution_order.clear()
            
            if isinstance(workflow, dict):
                for node_id, node_data in workflow.items():
                    if isinstance(node_data, dict) and 'class_type' in node_data:
                        class_type = node_data['class_type']
                        inputs = node_data.get('inputs', {})
                        
                        node = WorkflowNode(node_id, class_type, inputs)
                        self.nodes[node_id] = node
                        
                        self.logger.info(f"Parsed node: {node_id} ({class_type})")
            
            self._calculate_execution_order()
            
            self.logger.info(f"Workflow parsed: {len(self.nodes)} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to parse workflow: {e}")
            return False
    
    def _calculate_execution_order(self):
        """Calculate node execution order (topological sort)"""
        
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected at node {node_id}")
            
            if node_id in visited:
                return
            
            temp_visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for input_name, input_value in node.inputs.items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        dep_node_id = str(input_value[0])
                        if dep_node_id in self.nodes:
                            visit(dep_node_id)
            
            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
        
        for node_id in self.nodes.keys():
            if node_id not in visited:
                visit(node_id)
        
        self.execution_order = order
        self.logger.info(f"Execution order: {' -> '.join(order)}")
    
    def _resolve_input(self, input_value: Any) -> Any:
        """
        Resolve input value (handle node connections)
        
        Args:
            input_value: Input value (can be node reference)
            
        Returns:
            Resolved value
        """
        if isinstance(input_value, list) and len(input_value) == 2:
            source_node_id = str(input_value[0])
            output_index = input_value[1]
            
            if source_node_id in self.nodes:
                source_node = self.nodes[source_node_id]
                
                if source_node.executed:
                    outputs = source_node.outputs
                    
                    if isinstance(output_index, int):
                        if isinstance(outputs, (list, tuple)):
                            if output_index < len(outputs):
                                return outputs[output_index]
                        elif isinstance(outputs, dict):
                            keys = list(outputs.keys())
                            if output_index < len(keys):
                                return outputs[keys[output_index]]
                    elif isinstance(output_index, str):
                        if isinstance(outputs, dict) and output_index in outputs:
                            return outputs[output_index]
                    
                    if outputs:
                        if isinstance(outputs, tuple) and len(outputs) == 1:
                            return outputs[0]
                        return outputs
        
        return input_value
    
    def execute_workflow(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute workflow
        
        Args:
            context: Execution context (engine, models, etc.)
            
        Returns:
            Execution results
        """
        context = context or {}
        results = {}
        
        try:
            for node_id in self.execution_order:
                node = self.nodes[node_id]
                
                self.logger.info(f"Executing node: {node_id} ({node.class_type})")
                
                resolved_inputs = {}
                for input_name, input_value in node.inputs.items():
                    resolved_inputs[input_name] = self._resolve_input(input_value)
                
                # Try Genesis native nodes first, then custom nodes
                node_class = COMFYUI_NODE_REGISTRY.get(node.class_type)
                if not node_class:
                    node_class = custom_node_loader.get_node_class(node.class_type)
                
                if node_class:
                    node_instance = node_class()
                    
                    try:
                        output = node_instance.execute(**resolved_inputs)
                        node.outputs = output
                        node.executed = True
                        
                        results[node_id] = {
                            'success': True,
                            'outputs': output,
                            'class_type': node.class_type
                        }
                        
                        self.logger.info(f"Node {node_id} executed successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Node {node_id} execution failed: {e}")
                        results[node_id] = {
                            'success': False,
                            'error': str(e),
                            'class_type': node.class_type
                        }
                else:
                    self.logger.warning(f"Node class {node.class_type} not registered, skipping")
                    node.executed = True
                    results[node_id] = {
                        'success': False,
                        'error': f"Node class {node.class_type} not found",
                        'class_type': node.class_type
                    }
            
            return {
                'success': True,
                'results': results,
                'execution_order': self.execution_order
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': results
            }
    
    def get_output_nodes(self) -> List[WorkflowNode]:
        """Get output nodes (nodes with no dependents)"""
        output_nodes = []
        
        dependent_nodes = set()
        for node in self.nodes.values():
            for input_value in node.inputs.values():
                if isinstance(input_value, list) and len(input_value) == 2:
                    dependent_nodes.add(str(input_value[0]))
        
        for node_id, node in self.nodes.items():
            if node_id not in dependent_nodes:
                output_nodes.append(node)
        
        return output_nodes
    
    def convert_to_genesis(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ComfyUI workflow to Genesis format
        
        Args:
            workflow: ComfyUI workflow
            
        Returns:
            Genesis workflow format
        """
        if not self.parse_workflow(workflow):
            return {'success': False, 'error': 'Failed to parse workflow'}
        
        genesis_workflow = {
            'nodes': [],
            'connections': [],
            'metadata': {
                'format': 'genesis',
                'source': 'comfyui',
                'node_count': len(self.nodes)
            }
        }
        
        for node_id, node in self.nodes.items():
            genesis_node = {
                'id': node_id,
                'type': node.class_type,
                'inputs': {},
                'outputs': {}
            }
            
            for input_name, input_value in node.inputs.items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    connection = {
                        'from_node': str(input_value[0]),
                        'from_output': input_value[1],
                        'to_node': node_id,
                        'to_input': input_name
                    }
                    genesis_workflow['connections'].append(connection)
                else:
                    genesis_node['inputs'][input_name] = input_value
            
            genesis_workflow['nodes'].append(genesis_node)
        
        return {
            'success': True,
            'workflow': genesis_workflow
        }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow information"""
        return {
            'node_count': len(self.nodes),
            'nodes': {
                node_id: {
                    'class_type': node.class_type,
                    'input_count': len(node.inputs),
                    'executed': node.executed
                }
                for node_id, node in self.nodes.items()
            },
            'execution_order': self.execution_order
        }


def load_and_execute_workflow(
    workflow_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load and execute workflow
    
    Args:
        workflow_path: Path to workflow JSON
        context: Execution context
        
    Returns:
        Execution results
    """
    converter = ComfyUIWorkflowConverter()
    
    workflow = converter.load_workflow(workflow_path)
    
    if not converter.parse_workflow(workflow):
        return {'success': False, 'error': 'Failed to parse workflow'}
    
    results = converter.execute_workflow(context)
    
    return results
