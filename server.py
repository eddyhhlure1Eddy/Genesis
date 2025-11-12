"""
Genesis Production Server
Official entry point for Genesis AI Engine

Author: eddy
Date: 2025-11-12
Version: 1.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set environment variables BEFORE importing torch
if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('Genesis.Server')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Genesis AI Engine - Production Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic startup
  python server.py

  # With FP8 optimization
  python server.py --enable-fp8 --fp8-mode e4m3fn

  # Specific port and host
  python server.py --host 0.0.0.0 --port 8080

  # Low VRAM mode
  python server.py --lowvram

  # Debug mode with verbose logging
  python server.py --debug --verbose
        """
    )

    # Server Configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--host', type=str, default='127.0.0.1',
                             help='Server host (default: 127.0.0.1)')
    server_group.add_argument('--port', type=int, default=8080,
                             help='Server port (default: 8080)')
    server_group.add_argument('--listen', type=str,
                             help='Listen address (host:port)')

    # Device Configuration
    device_group = parser.add_argument_group('Device Configuration')
    device_group.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'mps', 'cpu'],
                             help='Compute device (default: cuda)')
    device_group.add_argument('--device-id', type=int, default=0,
                             help='CUDA device ID (default: 0)')

    # GPU Optimization
    gpu_group = parser.add_argument_group('GPU Optimization')
    gpu_group.add_argument('--enable-fp8', action='store_true',
                          help='Enable FP8 quantization (Ada/Hopper/Blackwell)')
    gpu_group.add_argument('--fp8-mode', type=str, default='e4m3fn',
                          choices=['e4m3fn', 'e5m2', 'e8m0fnu'],
                          help='FP8 format (default: e4m3fn)')
    gpu_group.add_argument('--enable-tf32', action='store_true', default=True,
                          help='Enable TF32 (default: True)')
    gpu_group.add_argument('--disable-tf32', action='store_true',
                          help='Disable TF32')
    gpu_group.add_argument('--enable-cudnn-benchmark', action='store_true', default=True,
                          help='Enable cuDNN benchmark (default: True)')
    gpu_group.add_argument('--disable-optimizations', action='store_true',
                          help='Disable all optimizations')

    # Memory Management
    memory_group = parser.add_argument_group('Memory Management')
    memory_group.add_argument('--lowvram', action='store_true',
                             help='Low VRAM mode')
    memory_group.add_argument('--highvram', action='store_true',
                             help='Keep models in VRAM')

    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument('--model-dir', type=str,
                           help='Model directory')
    path_group.add_argument('--output-dir', type=str, default='outputs',
                           help='Output directory (default: outputs)')
    path_group.add_argument('--log-file', type=str, default='logs/genesis.log',
                           help='Log file path (default: logs/genesis.log)')

    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--verbose', action='store_true',
                          help='Verbose logging')
    log_group.add_argument('--debug', action='store_true',
                          help='Debug logging')
    log_group.add_argument('--quiet', action='store_true',
                          help='Minimal logging')
    log_group.add_argument('--no-log-file', action='store_true',
                          help='Disable log file')

    return parser.parse_args()


def detect_gpu_info(logger):
    """Detect GPU architecture and capabilities"""
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return {
                'available': False,
                'device_name': 'CPU',
                'architecture': 'cpu'
            }

        device_props = torch.cuda.get_device_properties(0)
        major, minor = device_props.major, device_props.minor

        # Architecture mapping
        arch_map = {
            (7, 0): 'Volta',
            (7, 5): 'Turing',
            (8, 0): 'Ampere',
            (8, 6): 'Ampere',
            (8, 9): 'Ada Lovelace',
            (9, 0): 'Hopper',
            (12, 0): 'Blackwell',
        }

        arch_name = arch_map.get((major, minor), f'Unknown (sm_{major}{minor})')

        info = {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'architecture': arch_name,
            'compute_capability': (major, minor),
            'total_memory_gb': device_props.total_memory / (1024**3),
            'tf32_supported': major >= 8,
            'fp8_supported': (major == 8 and minor >= 9) or major >= 9,
        }

        return info

    except Exception as e:
        logger.error(f"Failed to detect GPU: {e}")
        return {'available': False, 'error': str(e)}


def print_banner(logger):
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗                  ║
║  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝                  ║
║  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗                  ║
║  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║                  ║
║  ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║                  ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝                  ║
║                                                                           ║
║                      Production AI Generation Engine                     ║
║                                                                           ║
║                          Version 1.0 | 2025-11-12                        ║
║                              Author: eddy                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    for line in banner.split('\n'):
        logger.info(line)


def initialize_genesis(args, gpu_info, logger):
    """Initialize Genesis core"""
    from genesis import GenesisCore

    config = {
        'device': args.device,
        'enable_optimizations': not args.disable_optimizations,
        'enable_tf32': args.enable_tf32 and not args.disable_tf32 and gpu_info.get('tf32_supported', False),
        'enable_cudnn_benchmark': args.enable_cudnn_benchmark,
        'enable_jit_fusion': not args.disable_optimizations,
        'enable_fp8': args.enable_fp8 and gpu_info.get('fp8_supported', False),
        'fp8_format': args.fp8_mode if args.enable_fp8 else 'e4m3fn',
    }

    logger.info("=" * 79)
    logger.info(" Initializing Genesis Core")
    logger.info("=" * 79)

    core = GenesisCore(config)

    logger.info(f"Device: {core.device}")

    if core.optimizer:
        report = core.optimizer.get_optimization_report()
        logger.info(f"Optimizations: {report['optimization_count']} applied")

        if args.verbose:
            for opt in report['optimizations_applied']:
                logger.info(f"  • {opt}")

        if report.get('fp8_enabled'):
            logger.info(f"FP8: Enabled ({report.get('fp8_format', 'unknown')})")

    logger.info("=" * 79)
    logger.info("")

    return core


def start_api_server(args, core, logger):
    """Start Flask API server"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS

        app = Flask('genesis')
        CORS(app)

        @app.route('/')
        def index():
            return jsonify({
                'service': 'Genesis AI Engine',
                'version': '1.0',
                'status': 'running',
                'device': str(core.device),
                'endpoints': {
                    'health': '/health',
                    'status': '/api/status',
                    'generate': '/api/generate',
                }
            })

        @app.route('/health')
        def health():
            return jsonify({'status': 'healthy', 'service': 'genesis'})

        @app.route('/api/status')
        def status():
            return jsonify(core.get_status())

        @app.route('/api/generate', methods=['POST'])
        def generate():
            try:
                data = request.get_json()

                task = {
                    'type': data.get('type', 'text_to_image'),
                    'params': data.get('params', {}),
                    'id': data.get('id')
                }

                result = core.execute(task)
                return jsonify(result)

            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500

        # Parse listen address
        if args.listen:
            if ':' in args.listen:
                host, port = args.listen.rsplit(':', 1)
                port = int(port)
            else:
                host = args.listen
                port = args.port
        else:
            host = args.host
            port = args.port

        logger.info("=" * 79)
        logger.info(" Genesis API Server")
        logger.info("=" * 79)
        logger.info(f"  URL: http://{host}:{port}")
        logger.info(f"  Endpoints:")
        logger.info(f"    GET  /              Service info")
        logger.info(f"    GET  /health        Health check")
        logger.info(f"    GET  /api/status    System status")
        logger.info(f"    POST /api/generate  Generate image")
        logger.info("=" * 79)
        logger.info("")
        logger.info("Server is running. Press Ctrl+C to stop.")
        logger.info("")

        app.run(host=host, port=port, debug=args.debug, threaded=True)

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install flask flask-cors")
        return 1
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point"""
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.debug else (
        logging.WARNING if args.quiet else logging.INFO
    )
    log_file = None if args.no_log_file else args.log_file
    logger = setup_logging(log_level, log_file)

    try:
        # Print banner
        print_banner(logger)

        # Detect GPU
        logger.info("Detecting GPU capabilities...")
        gpu_info = detect_gpu_info(logger)

        if gpu_info['available']:
            logger.info(f"GPU: {gpu_info['device_name']}")
            logger.info(f"Architecture: {gpu_info['architecture']}")
            logger.info(f"Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")

            if gpu_info.get('fp8_supported') and args.enable_fp8:
                logger.info(f"FP8: Enabled ({args.fp8_mode})")
            elif gpu_info.get('fp8_supported'):
                logger.info("FP8: Available (use --enable-fp8 to enable)")

            if gpu_info.get('tf32_supported') and args.enable_tf32:
                logger.info("TF32: Enabled")
        else:
            logger.info(f"Using: {gpu_info.get('device_name', 'CPU')}")

        logger.info("")

        # Initialize Genesis
        core = initialize_genesis(args, gpu_info, logger)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info("")

        # Start server
        return start_api_server(args, core, logger)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        if 'core' in locals():
            core.cleanup()
        logger.info("Genesis shutdown complete")


if __name__ == '__main__':
    sys.exit(main())
