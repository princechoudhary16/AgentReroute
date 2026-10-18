"""Local prevention commands; approval is a trusted host operation."""
import argparse
import json
import sys
from pathlib import Path
from .contracts import CompatibilityError,Denied,Context
from .gateway import rewire
from .retail import RetailSystem
from .evaluation import smoke,evaluate,screen_trajectories,save


def main(argv=None):
    parser=argparse.ArgumentParser(description='Deterministic retail tool gateway')
    subs=parser.add_subparsers(dest='command',required=True)
    for command in ('smoke','evaluate'):
        sub=subs.add_parser(command);sub.add_argument('--out',default='runs/prevention-'+command)
    sub=subs.add_parser('screen');sub.add_argument('--corpus',default='data/benchmark');sub.add_argument('--out',default='runs/prevention-screen.json')
    for command in ('compile','approve','audit','request','call'):
        sub=subs.add_parser(command)
        sub.add_argument('--state',default='runs/prevention/retail.db');sub.add_argument('--store',default='runs/prevention/grants.db')
        sub.add_argument('--policy',help='Explicit versioned retail policy JSON')
        if command=='approve':
            sub.add_argument('request_id');sub.add_argument('--decision',choices=('preview','approve','deny'),default='preview')
        if command=='audit':
            sub.add_argument('--session');sub.add_argument('--customer');sub.add_argument('--trace',action='store_true')
        if command in ('request','call'):
            sub.add_argument('tool');sub.add_argument('--arguments',required=True,help='JSON object')
            sub.add_argument('--session',required=True);sub.add_argument('--customer',required=True,help='Trusted host identity, never supplied by an agent')
            if command=='call':
                sub.add_argument('--operation-id');sub.add_argument('--token-env',default='PREVENTION_APPROVAL_TOKEN',help='Read capability from environment, not shell arguments')
    args=parser.parse_args(argv)
    try:
        if args.command=='smoke':result=smoke(args.out)
        elif args.command=='evaluate':
            full=evaluate(args.out)
            result={**full,'conditions':{name:{k:v for k,v in values.items() if k!='cases'} for name,values in full['conditions'].items()}}
        elif args.command=='screen':
            from agentrx.data import load_corpus
            cases,_=load_corpus(args.corpus)
            result=screen_trajectories([case['trajectory'] for case in cases])
            save(args.out,result)
            result={'trajectories':len(result['trajectories']),'proven_prevented_failures':0,'recovered_tasks':None,'report':args.out}
        else:
            policy=json.loads(Path(args.policy).read_text()) if args.policy else None
            gateway=rewire(RetailSystem.create(args.state),policy,store_path=args.store)
            if args.command=='compile':result=gateway.coverage
            elif args.command=='approve':
                action={'preview':gateway.admin.preview,'approve':gateway.admin.approve,'deny':gateway.admin.deny}[args.decision]
                result=action(args.request_id)
            elif args.command=='audit':
                if args.trace:
                    if not args.session or not args.customer:raise ValueError('--trace needs trusted --session and --customer')
                    ctx=Context(args.session,args.customer);gateway._store.register(ctx)
                    result=gateway.export_trace(ctx)
                else:result=gateway.audit()
            else:
                import os
                ctx=Context(args.session,args.customer);arguments=json.loads(args.arguments)
                if args.command=='request':result=gateway.request_approval(ctx,args.tool,arguments)
                else:result=gateway.call(ctx,args.tool,arguments,args.operation_id,os.environ.get(args.token_env))
        print(json.dumps(result,indent=2));return 0
    except (CompatibilityError,Denied,ValueError,OSError) as exc:
        print(json.dumps({'status':'error','message':str(exc)}),file=sys.stderr);return 2
