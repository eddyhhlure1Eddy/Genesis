#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ComfyUI Workflow Runner for Genesis
Author: eddy

Run and integrate ComfyUI workflow JSON files stored under workflows/comfyui.
This tool loads compatible custom nodes from custom_nodes/Comfyui and
executes the workflow using the built-in ComfyUIWorkflowConverter.

Usage examples:
  py apps\run_comfy_workflow.py --list
  py apps\run_comfy_workflow.py --name example.json
  py apps\run_comfy_workflow.py --path C:\\path\\to\\workflow.json
"""

import sys
import json
import argparse
from pathlib import Path
import logging

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows" / "comfyui"
CUSTOM_NODES_COMFY_DIR = PROJECT_ROOT / "custom_nodes" / "Comfyui"

# Ensure the parent of project root is on sys.path so that
# 'original_Genesis' can be imported as a package name
sys.path.insert(0, str(PROJECT_ROOT.parent))

# Imports from core via package-qualified name
from original_Genesis.core.workflow_converter import ComfyUIWorkflowConverter
from original_Genesis.core.custom_node_loader import custom_node_loader


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def list_workflows() -> list[str]:
    if not WORKFLOWS_DIR.exists():
        return []
    return [p.name for p in sorted(WORKFLOWS_DIR.glob("*.json"))]


def load_workflow_path(name: str | None, path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Workflow file not found: {p}")
        return p
    if not name:
        raise ValueError("Either --name or --path must be provided")
    p = WORKFLOWS_DIR / name
    if not p.exists():
        # Allow missing .json extension
        if not name.lower().endswith('.json'):
            p = WORKFLOWS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Workflow not found in {WORKFLOWS_DIR}: {name}")
    return p


def ensure_custom_nodes_loaded(logger: logging.Logger) -> None:
    """Scan and load compatible custom nodes from custom_nodes/Comfyui."""
    if CUSTOM_NODES_COMFY_DIR.exists():
        logger.info(f"Scanning custom nodes in: {CUSTOM_NODES_COMFY_DIR}")
        custom_node_loader.scan_and_load_custom_nodes(str(CUSTOM_NODES_COMFY_DIR))
        loaded = custom_node_loader.get_all_nodes()
        logger.info(f"Loaded {len(loaded)} custom nodes")
    else:
        logger.warning(
            f"Custom nodes directory not found: {CUSTOM_NODES_COMFY_DIR}. "
            f"Workflows that reference external nodes may fail to execute."
        )


def summarize_results(result: dict) -> str:
    if not result.get('success'):
        return f"Execution failed: {result.get('error', 'unknown error')}"
    lines = []
    lines.append("Execution succeeded")
    lines.append(f"Processed nodes: {len(result.get('results', {}))}")
    order = result.get('execution_order')
    if order:
        lines.append("Execution order: " + " -> ".join(order))
    failed = [nid for nid, r in result.get('results', {}).items() if not r.get('success')]
    if failed:
        lines.append(f"Failed nodes: {', '.join(failed)}")
    return "\n".join(lines)


def run_workflow(workflow_file: Path, log_level: str = "INFO") -> int:
    setup_logging(log_level)
    logger = logging.getLogger("WorkflowRunner")

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Workflow file: {workflow_file}")

    ensure_custom_nodes_loaded(logger)

    # Load JSON
    with workflow_file.open('r', encoding='utf-8') as f:
        workflow = json.load(f)

    # Execute
    converter = ComfyUIWorkflowConverter()
    logger.info("Parsing workflow...")
    if not converter.parse_workflow(workflow):
        logger.error("Failed to parse workflow")
        return 1

    logger.info("Executing workflow...")
    result = converter.execute_workflow()

    print("\n" + "=" * 70)
    print(" ComfyUI Workflow Result")
    print("=" * 70)
    print(summarize_results(result))
    print("=" * 70 + "\n")

    # Non-zero if any node failed
    if not result.get('success'):
        return 2
    any_failed = any(not r.get('success') for r in result.get('results', {}).values())
    return 0 if not any_failed else 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ComfyUI workflow JSON using Genesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", help="Workflow file name under workflows/comfyui")
    parser.add_argument("--path", help="Absolute or relative path to a workflow JSON")
    parser.add_argument("--list", action="store_true", help="List available workflows")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    if args.list:
        items = list_workflows()
        print("\nAvailable workflows in workflows/comfyui:")
        if not items:
            print("  (none)")
        else:
            for name in items:
                print(f"  - {name}")
        print()
        return 0

    wf_path = load_workflow_path(args.name, args.path)
    return run_workflow(wf_path, args.log_level)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
