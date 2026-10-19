"""Generic metadata discovery/inspection; synthetic execution demos only."""
import argparse
import json
import sys
from pathlib import Path
from . import Tool, Middleware, PlanError
from .discovery import discover, response_schema
from .examples import smoke, evaluate, save


def load_catalog(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise PlanError('Catalog must be a list of tool metadata objects')
    def unavailable(**kwargs):
        raise RuntimeError('Metadata-only tools cannot be invoked')
    return [Tool(row['name'], row.get('description', ''), row['input_schema'], unavailable,
                 row.get('output_schema'), row.get('effect', 'unknown')) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generic conservative tool prerequisite middleware')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('smoke')
    sub.add_argument('--out', default='runs/middleware-smoke')
    sub.add_argument('--plan', help='Use a compatible saved plan instead of the offline fixture')
    sub = commands.add_parser('evaluate')
    sub.add_argument('--out', default='runs/middleware-evaluation')
    for name in ('discover', 'inspect'):
        sub = commands.add_parser(name)
        sub.add_argument('--catalog', required=True, help='Tool metadata JSON; no enterprise endpoints are invoked')
        sub.add_argument('--validators', required=True, nargs='+', help='Host-designated read-only validation tool names')
        if name == 'inspect':
            sub.add_argument('--plan', required=True)
        else:
            sub.add_argument('--config', default='configs/dev_ollama.json')
            sub.add_argument('--model', help='Installed Ollama model override')
            sub.add_argument('--out', default='runs/middleware-discovery')
    args = parser.parse_args(argv)
    try:
        if args.command == 'smoke':
            plan = json.loads(Path(args.plan).read_text()) if args.plan else None
            result = smoke(args.out, plan)
        elif args.command == 'evaluate':
            result = evaluate(args.out)
        else:
            tools = load_catalog(args.catalog)
            if args.command == 'inspect':
                plan = json.loads(Path(args.plan).read_text())
                result = Middleware(tools, plan, validators=args.validators).coverage
            else:
                from agentrx.providers import OllamaProvider
                config = json.loads(Path(args.config).read_text()).get('client', {})
                if config.get('provider') != 'ollama':
                    raise PlanError('Discovery CLI expects a native Ollama configuration')
                if args.model:
                    config = {**config, 'model': args.model}
                    config.pop('model_env', None)
                model = OllamaProvider.from_config(config)
                model.prepare_for_run()
                plan = discover(tools, model, validators=args.validators)
                save(Path(args.out) / 'plan.json', plan)
                save(Path(args.out) / 'model-response.json', model.last_raw)
                save(Path(args.out) / 'response-schema.json', response_schema(tools, args.validators))
                result = {'plan': str(Path(args.out) / 'plan.json'), 'rules': len(plan['rules']),
                          'coverage': Middleware(tools, plan, validators=args.validators).coverage,
                          'model': model.identity, 'telemetry': model.last_telemetry}
                save(Path(args.out) / 'discovery.json', result)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'error', 'message': str(exc)}), file=sys.stderr)
        return 2
