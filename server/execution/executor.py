"""
Genesis Workflow Executor
Execute ComfyUI-compatible workflows
Author: eddy
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Workflow Executor - Execute ComfyUI-compatible workflows

    Features:
    - Parse ComfyUI workflow JSON
    - Validate workflow structure
    - Execute nodes in correct order (topological sort)
    - Cache node outputs
    - Progress tracking
    """

    def __init__(self, registry, engine):
        """
        Initialize executor

        Args:
            registry: Node registry
            engine: Genesis engine
        """
        self.registry = registry
        self.engine = engine
        self.logger = logging.getLogger(f"{__name__}.WorkflowExecutor")

    def execute(self, workflow: Dict[str, Any], prompt_id: str = None) -> Dict[str, Any]:
        """
        Execute a workflow

        Args:
            workflow: ComfyUI workflow JSON
            prompt_id: Prompt ID for tracking

        Returns:
            Execution results
        """
        self.logger.info(f"Executing workflow (prompt_id={prompt_id})")

        try:
            # Validate workflow
            errors = self.validate(workflow)
            if errors:
                raise ValueError(f"Workflow validation failed: {errors}")

            # Parse workflow
            nodes, connections = self._parse_workflow(workflow)

            # Determine execution order
            execution_order = self._topological_sort(nodes, connections)

            # Execute nodes
            outputs = {}
            for node_id in execution_order:
                self.logger.info(f"Executing node: {node_id}")
                node_output = self._execute_node(node_id, nodes, connections, outputs)
                outputs[node_id] = node_output

            self.logger.info(f"Workflow execution completed (prompt_id={prompt_id})")

            return {
                'success': True,
                'prompt_id': prompt_id,
                'outputs': outputs
            }

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'prompt_id': prompt_id,
                'error': str(e)
            }

    def validate(self, workflow: Dict[str, Any]) -> List[str]:
        """
        Validate workflow structure

        Args:
            workflow: ComfyUI workflow JSON

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not isinstance(workflow, dict):
            errors.append("Workflow must be a dictionary")
            return errors

        # Check if workflow has nodes
        if not workflow:
            errors.append("Workflow is empty")
            return errors

        # Validate each node
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                errors.append(f"Node {node_id} data must be a dictionary")
                continue

            # Check for required fields
            if 'class_type' not in node_data:
                errors.append(f"Node {node_id} missing 'class_type' field")
                continue

            class_type = node_data['class_type']

            # Check if node class exists
            if not self.registry.has(class_type):
                errors.append(f"Node {node_id}: Unknown node type '{class_type}'")
                continue

            # Validate inputs
            if 'inputs' in node_data:
                inputs = node_data['inputs']
                if not isinstance(inputs, dict):
                    errors.append(f"Node {node_id}: inputs must be a dictionary")

        return errors

    def _parse_workflow(self, workflow: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Parse workflow into nodes and connections

        Args:
            workflow: ComfyUI workflow JSON

        Returns:
            Tuple of (nodes_dict, connections_dict)
        """
        nodes = {}
        connections = defaultdict(list)

        for node_id, node_data in workflow.items():
            class_type = node_data['class_type']
            inputs = node_data.get('inputs', {})

            nodes[node_id] = {
                'id': node_id,
                'class_type': class_type,
                'inputs': inputs,
                'raw_data': node_data
            }

            # Parse connections
            for input_name, input_value in inputs.items():
                # Check if input is a connection (list format: [source_node_id, source_output_index])
                if isinstance(input_value, list) and len(input_value) >= 2:
                    source_node_id = str(input_value[0])
                    source_output_index = input_value[1]

                    connections[node_id].append({
                        'source_node': source_node_id,
                        'source_output': source_output_index,
                        'target_input': input_name
                    })

        return nodes, connections

    def _topological_sort(self, nodes: Dict, connections: Dict) -> List[str]:
        """
        Perform topological sort to determine execution order

        Args:
            nodes: Nodes dictionary
            connections: Connections dictionary

        Returns:
            List of node IDs in execution order
        """
        # Build dependency graph
        in_degree = {node_id: 0 for node_id in nodes}
        adjacency = defaultdict(list)

        for target_node, conns in connections.items():
            for conn in conns:
                source_node = conn['source_node']
                adjacency[source_node].append(target_node)
                in_degree[target_node] += 1

        # Kahn's algorithm
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        execution_order = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(execution_order) != len(nodes):
            raise ValueError("Workflow contains cycles or disconnected nodes")

        return execution_order

    def _execute_node(
        self,
        node_id: str,
        nodes: Dict,
        connections: Dict,
        outputs: Dict
    ) -> Any:
        """
        Execute a single node

        Args:
            node_id: Node ID
            nodes: All nodes
            connections: All connections
            outputs: Outputs from previously executed nodes

        Returns:
            Node output
        """
        node = nodes[node_id]
        class_type = node['class_type']
        inputs = node['inputs'].copy()

        # Get node class
        node_class = self.registry.get_class(class_type)
        if not node_class:
            raise ValueError(f"Node class not found: {class_type}")

        # Resolve input connections
        node_connections = connections.get(node_id, [])
        for conn in node_connections:
            source_node = conn['source_node']
            source_output = conn['source_output']
            target_input = conn['target_input']

            # Get output from source node
            if source_node not in outputs:
                raise ValueError(f"Source node {source_node} not executed yet")

            source_result = outputs[source_node]

            # Handle tuple outputs (multiple return values)
            if isinstance(source_result, tuple):
                if source_output >= len(source_result):
                    raise ValueError(f"Source output index {source_output} out of range")
                inputs[target_input] = source_result[source_output]
            else:
                inputs[target_input] = source_result

        # Instantiate and execute node
        try:
            node_instance = node_class()

            # Get function name
            function_name = getattr(node_class, 'FUNCTION', 'execute')

            # Call node function
            if hasattr(node_instance, function_name):
                node_function = getattr(node_instance, function_name)
                result = node_function(**inputs)
            else:
                raise ValueError(f"Node {class_type} has no function '{function_name}'")

            self.logger.debug(f"Node {node_id} ({class_type}) executed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Node {node_id} ({class_type}) execution failed: {e}")
            raise RuntimeError(f"Node {node_id} execution failed: {e}")

    def get_node_outputs(self, node_id: str, outputs: Dict) -> Any:
        """Get outputs from a specific node"""
        return outputs.get(node_id)
