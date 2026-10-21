"""python -m demo [--serve --port 8765] [--out runs/demo_versailles.html] [--scenario all|<id>]"""
import argparse
import sys

from .scenarios import SCENARIOS


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m demo', description='Versailles MAS propagation & recovery demo')
    parser.add_argument('--serve', action='store_true', help='run the live HTTP + SSE server instead of exporting')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--delay', type=float, default=0.9, help='seconds between streamed frames at 1x')
    parser.add_argument('--out', default='runs/demo_versailles.html')
    parser.add_argument('--scenario', default='all', choices=['all', *SCENARIOS])
    args = parser.parse_args(argv)
    from .visualizer import build_demo_html, serve
    if args.serve:
        serve(args.port, args.host, args.delay)
        return 0
    path = build_demo_html(args.out, None if args.scenario == 'all' else [args.scenario])
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
