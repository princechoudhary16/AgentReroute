"""Command-line reproduction, audit, diagnosis, evaluation, and offline smoke."""
import argparse
import copy
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from .data import (EXPECTED, REFERENCE_REVISION, REFERENCE_URL, load_corpus,
                   normalize, normalize_gold, prepare_corpus, validate_case)
from .experiments import (evaluate_run, import_predictions, run_experiment, run_stress,
                          validate_experiment)
from .io import digest, read_json, write_json
from .pipeline import diagnose, validate_config
from .providers import CachedProvider, HTTPProvider, ReplayProvider


def apply_model_selection(config, model_name):
    if not model_name:
        return config
    cfg = copy.deepcopy(config)
    models = cfg.get('models', {})
    if isinstance(models, dict) and model_name in models:
        cfg['client'] = copy.deepcopy(models[model_name])
    elif model_name in ('ollama', 'gemini', 'openai'):
        defaults = {
            'ollama': {'provider':'ollama', 'model':'qwen2.5-coder:7b',
                       'num_ctx':16384, 'settings':{'temperature':0}},
            'gemini': {'provider':'gemini', 'model':'gemini-2.5-flash', 'api_key_env':'GEMINI_API_KEY'},
            'openai': {'provider':'openai', 'model':'gpt-4o', 'api_key_env':'OPENAI_API_KEY'},
        }
        current = cfg.get('client', {})
        # Preserve all same-provider options; do not carry an incompatible model
        # or sampling settings across a provider switch.
        cfg['client'] = (copy.deepcopy(current) if current.get('provider') == model_name
                         else copy.deepcopy(defaults[model_name]))
        cfg['client'].setdefault('model', defaults[model_name]['model'])
    else:
        cfg.setdefault('client', {})
        cfg['client']['model'] = model_name
        cfg['client'].pop('model_env', None)  # An explicit CLI model wins.
    return cfg


def resolve_run_config(config_arg, model_arg):
    if config_arg is not None:
        return read_json(config_arg)
    if model_arg == 'ollama':
        return read_json(Path('configs/dev_ollama.json'))
    if model_arg == 'gemini':
        return read_json(Path('configs/dev_gemini.json'))
    if model_arg == 'openai':
        return read_json(Path('configs/dev_openai.json'))
    return read_json(Path('configs/paper_gpt5.json'))


def fetch_reference(destination):
    destination=Path(destination).resolve()
    if destination.exists():
        try:
            revision=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True,
                                             stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError,FileNotFoundError):
            raise ValueError('Fetch destination already exists; select a new empty path') from None
        if revision != REFERENCE_REVISION:
            raise ValueError('Existing reference checkout is not the pinned revision; choose a new path')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','init',str(destination)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(destination),'fetch','--depth','1',REFERENCE_URL,REFERENCE_REVISION],check=True)
    subprocess.run(['git','-C',str(destination),'checkout','--detach','FETCH_HEAD'],check=True,stdout=subprocess.DEVNULL)
    return destination


def smoke(output, ranked=False):
    output=Path(output)
    demo=json.loads(files('agentrx').joinpath('resources/demo.json').read_text(encoding='utf-8'))
    trajectory=normalize(demo['trajectory'],'tau')
    case={'trajectory':trajectory,'gold':normalize_gold(demo['annotation']),'policy':'','tools':{}}
    validate_case(case)
    cases=[case]
    if ranked:
        demo['responses']['ranked_judge'] = json.loads(files('agentrx').joinpath('resources/track_b_demo.json').read_text(encoding='utf-8'))
    write_json(output/'fixture_replay.json',demo['responses'])
    write_json(output/'corpus/cases.json',cases)
    write_json(output/'corpus/manifest.json',{'corpus_hash':digest(cases),'available_total':1,
        'measurement_scope':'offline_fixture_corpus','domains':{
            domain:{'expected':count,'available':1 if domain=='tau' else 0,'status':'fixture_only'}
            for domain,count in EXPECTED.items()}})
    config={'name':'offline-smoke','repetitions':3,'domains':['tau'],
            'client':{'provider':'replay','replay_file':str((output/'fixture_replay.json').resolve())},
            'variants':[{'name':'baseline','evidence':False,'checklist':False},
                        {'name':'agentrx','evidence':True,'checklist':True}]}
    if ranked:
        config['name'] = 'offline-track-b-smoke'
        config['variants'] = [{'name':'ranked','protocol':'ranked','divergence_threshold':.3,'max_candidates':3}]
    return run_experiment(output/'corpus',config,output/'experiment')


def parser():
    p=argparse.ArgumentParser(description='AgentRx paper reconstruction; benchmark data and offline fixtures are kept distinct.')
    commands=p.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Import and audit authors\' released benchmark subset')
    source=prepare.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--fetch',action='store_true')
    prepare.add_argument('--reference-dir',type=Path,default=Path('data/reference'))
    prepare.add_argument('--out',type=Path,default=Path('data/benchmark'))
    audit=commands.add_parser('audit',help='Verify corpus hashes and gold references')
    audit.add_argument('--corpus',type=Path,default=Path('data/benchmark'))
    single=commands.add_parser('diagnose',help='Diagnose one raw or normalized trajectory')
    single.add_argument('--input',type=Path,required=True)
    single.add_argument('--domain',choices=list(EXPECTED),required=True)
    single.add_argument('--config',type=Path)
    single.add_argument('--model',help='Model preset or override (gemini, openai, ollama, or model in config)')
    single.add_argument('--variant',help='Variant name from config (default: first)')
    single.add_argument('--replay',type=Path,help='Explicit offline fixture responses; never benchmark measurements')
    single.add_argument('--out',type=Path,required=True)
    single.add_argument('--policy',type=Path)
    single.add_argument('--tools',type=Path)
    run=commands.add_parser('run',help='Run independent benchmark repetitions, resumable')
    run.add_argument('--corpus',type=Path,default=Path('data/benchmark'))
    run.add_argument('--config',type=Path,default=None)
    run.add_argument('--model',help='Model preset or override (gemini, openai, ollama, or model in config)')
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--preflight',action='store_true',help='Validate corpus/config/credentials without calling a model')
    evaluate=commands.add_parser('evaluate',help='Regenerate fixed-denominator reports from stored jobs')
    evaluate.add_argument('--corpus',type=Path,default=Path('data/benchmark'))
    evaluate.add_argument('--run',type=Path,required=True)
    replay=commands.add_parser('smoke',help='Offline fixture test of generation/checking/judging/repetition plumbing')
    replay.add_argument('--out',type=Path,default=Path('runs/smoke'))
    ranked=commands.add_parser('smoke-track-b',help='Offline fixture test of divergence gating and ranked recovery payload')
    ranked.add_argument('--out',type=Path,default=Path('runs/smoke-track-b'))
    external=commands.add_parser('import-predictions',help='Score external W&W or other baseline predictions')
    external.add_argument('--corpus',type=Path,default=Path('data/benchmark'))
    external.add_argument('--input',type=Path,required=True)
    external.add_argument('--source-name',required=True)
    external.add_argument('--out',type=Path,required=True)
    stress=commands.add_parser('stress',help='Unlabeled MAST-style normalized IR; no gold accuracy claims')
    stress.add_argument('--input',type=Path,required=True)
    stress.add_argument('--config',type=Path,required=True)
    stress.add_argument('--out',type=Path,required=True)
    return p


def _brief(summary, path):
    return {'measurement_kind':summary['measurement_kind'], 'output':str(path.resolve()),
            **({'handoff_summary':str((path/summary['handoff_summary']).resolve())} if 'handoff_summary' in summary else {}),
            'scope':summary.get('measurement_scope', 'unlabeled_stress'),
            'result_rows':len(summary.get('results',[]))}


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='prepare':
            source=fetch_reference(args.reference_dir) if args.fetch else args.source
            manifest=prepare_corpus(source,args.out)
            value={'available_total':manifest['available_total'],'paper_total':170,
                   'domains':manifest['domains'],'exclusions':manifest['exclusions'],
                   'corpus_hash':manifest['corpus_hash'],'source_pinned':manifest['source_pinned']}
        elif args.command=='audit':
            cases,manifest=load_corpus(args.corpus)
            value={'validated_pairs':len(cases),'corpus_hash':manifest['corpus_hash'],'domains':manifest['domains']}
        elif args.command=='smoke':
            value=_brief(smoke(args.out),args.out/'experiment')
        elif args.command=='smoke-track-b':
            value=_brief(smoke(args.out,ranked=True),args.out/'experiment')
        elif args.command=='run':
            config = resolve_run_config(args.config, args.model)
            if args.model:
                config = apply_model_selection(config, args.model)
            if args.preflight:
                validated=validate_experiment(config)
                cases,manifest=load_corpus(args.corpus)
                provider=HTTPProvider.from_config(validated['client'])
                value={'preflight':'ok','network_calls':0,'provider':provider.identity,
                       'available_pairs':len(cases),'domains':manifest['domains']}
            else:
                summary=run_experiment(args.corpus,config,args.out)
                value=_brief(summary,args.out)
                print(json.dumps(value,indent=2))
                if any(rep['failed_jobs'] or rep['invalid_predictions'] or rep['partial_checker_jobs']
                       for row in summary['results'] for rep in row['repetitions']):
                    return 1
                return 0
        elif args.command=='evaluate':
            value=_brief(evaluate_run(args.corpus,args.run),args.run)
        elif args.command=='import-predictions':
            value=import_predictions(args.corpus,args.input,args.out,args.source_name)
            value={'output':str(args.out.resolve()),'denominator':value['metrics']['denominator']}
        elif args.command=='stress':
            value=_brief(run_stress(args.input,read_json(args.config),args.out),args.out)
        elif args.command=='diagnose':
            raw=read_json(args.input)
            if isinstance(raw,list) and len(raw)==1 and isinstance(raw[0],dict) and 'steps' in raw[0]:
                raw=raw[0]
            trajectory=normalize(raw,args.domain,args.input.stem if not isinstance(raw,dict) else None)
            config_dict = read_json(args.config) if args.config else (resolve_run_config(None, args.model) if args.model else {})
            if args.model:
                config_dict = apply_model_selection(config_dict, args.model)
            config=validate_experiment(config_dict)
            variants=config['variants']
            variant=next((v for v in variants if v['name']==args.variant),None) if args.variant else variants[0]
            if variant is None:
                raise ValueError('Requested variant absent from config')
            provider=ReplayProvider(args.replay) if args.replay else HTTPProvider.from_config(config['client'])
            policy=args.policy.read_text(encoding='utf-8') if args.policy else ''
            tools=read_json(args.tools) if args.tools else {}
            cached=CachedProvider(provider,args.out/'requests',{'single':digest(trajectory),'variant':variant})
            result=diagnose(trajectory,policy,tools,variant,cached,args.out)
            value={'status':result['status'],'prediction':result['prediction'],
                   **({'recovery_handoff':result['recovery_handoff']} if 'recovery_handoff' in result else {}),
                   'measurement_kind':result['measurement_kind'],'error':result.get('error'),
                   'violation_count':len(result['violations']),'output':str(args.out.resolve())}
            print(json.dumps(value,indent=2))
            return 0 if result['status']=='ok' else 1
        print(json.dumps(value,indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
