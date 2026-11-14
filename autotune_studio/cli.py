"""Command line entry point: ``autotune-studio`` / ``python -m autotune_studio``. Owned by: Chief Architect."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autotune-studio", description="AutoTune Studio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("launch", help="start the backend and open the UI")
    lp.add_argument("--port", type=int, default=None)
    lp.add_argument("--host", default="127.0.0.1")
    lp.add_argument("--workspace", default=None)
    lp.add_argument("--allow", action="append", default=[], help="extra allowed dataset root (repeatable)")
    lp.add_argument("--tunnel", choices=["colab_proxy", "cloudflared", "local"], default=None)
    lp.add_argument("--no-browser", action="store_true")

    sub.add_parser("env", help="print detected environment and hardware as JSON")

    args = ap.parse_args(argv)
    from .studio import Studio

    if args.cmd == "env":
        s = Studio()
        print(json.dumps({"environment": s.detect_environment().model_dump(mode="json"),
                          "hardware": s.detect_hardware().model_dump(mode="json")}, indent=2))
        return 0

    studio = Studio(workspace_dir=args.workspace, allowed_roots=args.allow or None)
    info = studio.launch(port=args.port, host=args.host, prefer=args.tunnel, open_browser=not args.no_browser)
    print(f"AutoTune Studio backend: {info.local_url}  (method={info.method}, "
          f"public={'verified' if info.publicly_verified else 'no'})")
    if info.public_url:
        print(f"Public URL: {info.public_url}")
    for d in info.diagnostics:
        print(f"  - {d}")
    if args.no_browser and info.open_url:
        # Printed only on explicit request for a headless start; this is the user's own terminal.
        print("Open this link (contains a one-time session token in the #fragment):")
        print(f"  {info.open_url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        studio.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
