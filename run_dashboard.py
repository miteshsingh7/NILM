"""Run script for the NILM Telemetry Rack & Oscilloscope Console.

Usage:
    python run_dashboard.py [--port 8000] [--host 0.0.0.0]
"""

import argparse
import sys
from pathlib import Path
from aiohttp import web

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
from app.server import create_app


def main() -> None:
    default_port = int(os.environ.get("PORT", 7860))
    parser = argparse.ArgumentParser(description="Run NILM Telemetry Rack Web Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port number (default: {default_port})")
    args = parser.parse_args()

    app = create_app()

    banner = f"""
================================================================================
⚡ NILM CORE // DSP TELEMETRY RACK v4.2
================================================================================
Server initialized and streaming live REDD telemetry.
Access the web console in your browser:

    -> Local:   http://localhost:{args.port}
    -> Network: http://{args.host}:{args.port}

Endpoints:
    - Web UI:           GET  /
    - Real-Time Stream: WS   /ws/telemetry
    - Telemetry State:  GET  /api/status
    - Feeder List:      GET  /api/feeders
    - Transport Action: POST /api/control
    - Cost Attribution: GET  /api/usage-cost
    - Signatures:       GET  /api/signatures
    - Sensor Feeds:     GET  /api/sensor-feeds
    - Diagnostics:      GET  /api/diagnostics
    - Export CSV:       GET  /api/export
================================================================================
"""
    print(banner)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
