"""Independent repetitions, resumable jobs, and paper-style reports."""
import csv
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from .data import EXPECTED, load_corpus, normalize
from .evaluation import score, aggregate, METRIC_NAMES
from .io import digest, read_json, write_json
from .pipeline import diagnose, validate_config
from .prompts import prompt_identity
from .providers import CachedProvider, HTTPProvider


CLIENT_KEYS = {'provider','model','base_url','api_key_env','deployment','api_version','settings',
               'timeout','retries','max_input_chars','replay_file','base_url_env','model_env',
               'num_ctx','think','keep_alive'}


def experiment_variants(config):
    if 'variants' in config:
        variants = config['variants']
    elif config.get('matrix') == 'paper':
        variants = [{'name':'baseline','evidence':False,'checklist':False,'protocol':'joint'},
                    {'name':'step_then_category','evidence':False,'checklist':False,'protocol':'step_then_category'},
                    {'name':'checklist','evidence':False,'checklist':True,'protocol':'joint'}]
        for generation in ('stepwise','oneshot'):
            variants.extend([
                {'name':f'violations_{generation}','evidence':True,'checklist':False,'generation':generation,'protocol':'joint'},
                {'name':f'step_then_category_violations_{generation}','evidence':True,'checklist':False,'generation':generation,'protocol':'step_then_category'},
                {'name':f'checklist_violations_{generation}','evidence':True,'checklist':True,'generation':generation,'protocol':'joint'}])
    else:
        variants = [{'name':'agentrx','evidence':True,'checklist':True,'generation':'stepwise','protocol':'joint'}]
    if not isinstance(variants,list) or not variants:
        raise ValueError('variants must be a non-empty list')
    result, names = [], set()
    for variant in variants:
        if set(variant) - {'name','evidence','checklist','generation','protocol','python_timeout'}:
            raise ValueError('Unsupported variant field')
        name = variant.get('name','')
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name) or name in names:
            raise ValueError('Variant names must be unique safe path components')
        names.add(name)
        result.append(validate_config(variant))
    return result


def validate_experiment(config):
    if set(config) - {'name','repetitions','matrix','variants','client','domains','limit_per_domain','models'}:
        raise ValueError('Unsupported experiment field; credentials must be environment variables')
    if config.get('matrix', 'paper') != 'paper':
        raise ValueError('Only the paper matrix is predefined; use variants for custom studies')
    repetitions = config.get('repetitions',3)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    client = config.get('client',{})
    if not isinstance(client,dict) or set(client)-CLIENT_KEYS:
        raise ValueError('Unsupported client field; do not put secret values in configuration')
    models = config.get('models')
    if models is not None:
        if not isinstance(models, dict):
            raise ValueError('models must be a dictionary of model configurations')
        for m_name, m_client in models.items():
            if not isinstance(m_client, dict) or set(m_client)-CLIENT_KEYS:
                raise ValueError(f'Unsupported fields in models.{m_name}; credentials must be environment variables')
    domains = config.get('domains', list(EXPECTED))
    if not isinstance(domains,list) or not domains or len(set(domains))!=len(domains) or set(domains)-set(EXPECTED):
        raise ValueError('Unknown or duplicated domains')
    limit = config.get('limit_per_domain')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('limit_per_domain must be positive')
    return {**config, 'repetitions':repetitions, 'client':client, 'domains':domains,
            'variants':experiment_variants(config)}


def implementation_hash():
    root = Path(__file__).parent
    paths = sorted(list(root.glob('*.py')) + list((root/'resources').glob('*.json')))
    return digest({str(p.relative_to(root)):p.read_text(encoding='utf-8') for p in paths})


def _select(cases, config):
    selected = []
    for domain in config['domains']:
        group = [c for c in cases if c['trajectory']['domain']==domain]
        limit = config.get('limit_per_domain')
        selected.extend(group[:limit] if limit else group)
    return selected


def job_path(output, variant, repetition, trajectory):
    return Path(output)/'jobs'/variant/f'rep-{repetition}'/trajectory['domain']/digest(trajectory['trajectory_id'])


def run_experiment(corpus_dir, config, output, provider_factory=None):
    config = validate_experiment(config)
    cases, manifest = load_corpus(corpus_dir)
    selected = _select(cases,config)
    if not selected:
        raise ValueError('No available cases in selected domains')
    factory = provider_factory or HTTPProvider.from_config
    provider = factory(config['client'])  # Credentials fail before any experiment is started.
    prepare = getattr(provider, 'prepare_for_run', None)
    if prepare is not None:
        prepare()
    identity = {'config':config,'corpus_hash':manifest['corpus_hash'], 'provider':provider.identity,
                'prompt_hash':prompt_identity(),'implementation_hash':implementation_hash()}
    identity_hash = digest(identity)
    output = Path(output)
    metadata_path = output/'experiment.json'
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata['identity_hash'] != identity_hash:
            raise ValueError('Experiment identity changed; choose a new output directory')
    else:
        metadata = {'identity_hash':identity_hash,'identity':identity,
                    'measurement_kind':provider.measurement_kind,'created_at':datetime.now(timezone.utc).isoformat(),
                    'python':platform.python_version(),'api_client':'stdlib urllib',
                    'selected_case_keys':[[c['trajectory']['domain'],c['trajectory']['trajectory_id']] for c in selected]}
        write_json(metadata_path,metadata)
    for variant in config['variants']:
        for repetition in range(config['repetitions']):
            cached = CachedProvider(provider, output/'requests',
                                    {'experiment':identity_hash,'variant':variant['name'],'repetition':repetition})
            for case in selected:
                trajectory = case['trajectory']
                path = job_path(output,variant['name'],repetition,trajectory)
                result_path = path/'result.json'
                if result_path.exists():
                    old = read_json(result_path)
                    if old.get('experiment_identity') != identity_hash:
                        raise ValueError('Job identity mismatch')
                    if old.get('status') == 'ok':
                        continue
                print(f"{variant['name']} rep={repetition+1}/{config['repetitions']} "
                      f"{trajectory['domain']}/{trajectory['trajectory_id']}",file=sys.stderr,flush=True)
                result = diagnose(trajectory,case['policy'],case['tools'],variant,cached,path)
                result['experiment_identity'] = identity_hash
                write_json(result_path,result)
    return evaluate_run(corpus_dir,output)


def evaluate_run(corpus_dir, output):
    cases,manifest = load_corpus(corpus_dir)
    output = Path(output)
    metadata = read_json(output/'experiment.json')
    if metadata['identity']['corpus_hash'] != manifest['corpus_hash']:
        raise ValueError('Evaluation corpus does not match experiment identity')
    config = metadata['identity']['config']
    selected = _select(cases,config)
    if [[c['trajectory']['domain'],c['trajectory']['trajectory_id']] for c in selected] != metadata['selected_case_keys']:
        raise ValueError('Selected corpus changed')
    results = []
    for variant in config['variants']:
        for domain in config['domains']:
            group = [c for c in selected if c['trajectory']['domain']==domain]
            if not group:
                continue
            repetitions = []
            for repetition in range(config['repetitions']):
                predictions = []
                for case in group:
                    path = job_path(output,variant['name'],repetition,case['trajectory'])/'result.json'
                    if path.exists():
                        result = read_json(path)
                        if result.get('experiment_identity') != metadata['identity_hash']:
                            raise ValueError('Stored result identity mismatch')
                        predictions.append(result)
                measured = score(group,predictions)
                measured['repetition'] = repetition
                repetitions.append(measured)
            results.append({'variant':variant['name'],'domain':domain,'denominator':len(group),
                            'repetitions':repetitions,'aggregate':aggregate(repetitions)})
    summary = {'measurement_kind':metadata['measurement_kind'],'measurement_scope':
               'pilot_subset' if config.get('limit_per_domain') else manifest.get('measurement_scope','available_release_subset'),
               'corpus_hash':manifest['corpus_hash'],'experiment_identity':metadata['identity_hash'],
               'domains':manifest['domains'],'results':results,
               'std_convention':'sample standard deviation across independent repetitions',
               'replication_claim':'No exact numeric replication claimed; original corpus membership/model settings require verification.'}
    write_json(output/'summary.json',summary)
    _reports(output,summary)
    return summary


def _reports(output, summary):
    # Table 4 focuses on ablations; Table 7 adds tolerance and relaxed categories.
    columns4 = ['step_accuracy','critical_category_accuracy','average_step_distance']
    columns7 = list(METRIC_NAMES)
    warning = ('OFFLINE FIXTURE TEST — NOT BENCHMARK RESULTS' if summary['measurement_kind']=='offline_fixture'
               else 'MEASURED RECONSTRUCTION — '+summary['measurement_scope'])
    for name,columns in [('table4',columns4),('table7',columns7)]:
        lines = [f'# {name.upper()}', '', warning, '',
                 'Corpus: '+summary['corpus_hash'], '',
                 'Mean ± sample SD across repetitions; accuracy shown as percentages.', '',
                 '| Domain | Variant | N | '+' | '.join(columns)+' |',
                 '| --- | --- | ---: | '+' | '.join(['---:']*len(columns))+' |']
        csv_rows = []
        for result in summary['results']:
            cells = []
            row = {'measurement_kind':summary['measurement_kind'],'scope':summary['measurement_scope'],
                   'domain':result['domain'],'variant':result['variant'],'denominator':result['denominator']}
            for column in columns:
                stat = result['aggregate'][column]
                mean,std = stat['mean'],stat['std']
                scale = 1 if column=='average_step_distance' else 100
                cells.append('—' if mean is None else f'{mean*scale:.2f} ± {std*scale:.2f}')
                row[column+'_mean'] = mean; row[column+'_std'] = std; row[column+'_n'] = stat['n']
            lines.append('| '+result['domain']+' | '+result['variant']+' | '+str(result['denominator'])+' | '+' | '.join(cells)+' |')
            csv_rows.append(row)
        lines.extend(['','Coverage, abstentions, checker errors, and per-trajectory details: see summary.json.','',
                      'Unavailable paper domains:'])
        for domain,coverage in summary['domains'].items():
            if coverage['available'] < coverage['expected']:
                lines.append(f"- {domain}: {coverage['available']}/{coverage['expected']} available ({coverage['status']}).")
        (output/f'{name}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if csv_rows:
            with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    flat, matrices = [], []
    for result in summary['results']:
        for repetition in result['repetitions']:
            matrices.append({'domain':result['domain'],'variant':result['variant'],'repetition':repetition['repetition'],
                             'confusion_matrix':repetition['confusion_matrix']})
            for row in repetition['per_trajectory']:
                p=row['prediction'] or {}
                flat.append({'domain':row['domain'],'trajectory_id':row['trajectory_id'],'variant':result['variant'],
                    'repetition':repetition['repetition'],'state':row['state'],'gold_step':row['gold']['step_index'],
                    'predicted_step':p.get('step_index'),'gold_category':row['gold']['category'],
                    'predicted_category':p.get('category'),'gold_agent':row['gold']['agent'],'predicted_agent':p.get('agent')})
    write_json(output/'confusion_matrices.json',matrices)
    if flat:
        with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


def import_predictions(corpus_dir, predictions_path, output, source_name):
    cases,manifest=load_corpus(corpus_dir)
    rows=read_json(predictions_path)
    if not isinstance(rows,list):
        raise ValueError('External predictions must be a JSON list')
    measured=score(cases,rows)
    result={'source':source_name,'corpus_hash':manifest['corpus_hash'],'measurement_kind':'external_predictions',
            'metrics':measured,'note':'Caller must document original baseline prompts/model/configuration.'}
    write_json(output,result)
    return result


def run_stress(input_path, config, output, provider_factory=None):
    """MAST-style unlabeled normalized IR: distributions only, never gold accuracy."""
    config=validate_experiment(config)
    rows=read_json(input_path)
    if not isinstance(rows,list) or not rows:
        raise ValueError('Stress input must be a nonempty list of normalized trajectory IR objects')
    trajectories=[normalize(r,r.get('domain','relwork')) for r in rows]
    keys=[(t['domain'],t['trajectory_id']) for t in trajectories]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate stress trajectory IDs')
    provider=(provider_factory or HTTPProvider.from_config)(config['client'])
    prepare=getattr(provider,'prepare_for_run',None)
    if prepare is not None:
        prepare()
    corpus_hash=digest(trajectories)
    identity=digest({'config':config,'corpus':corpus_hash,'provider':provider.identity,
                     'prompts':prompt_identity(),'implementation':implementation_hash()})
    output=Path(output)
    if (output/'stress.json').exists() and read_json(output/'stress.json')['identity']!=identity:
        raise ValueError('Stress experiment identity changed; choose a new output directory')
    write_json(output/'stress.json',{'identity':identity,'corpus_hash':corpus_hash,'config':config})
    results=[]
    for variant in config['variants']:
        for rep in range(config['repetitions']):
            cached=CachedProvider(provider,output/'requests',{'stress':identity,'variant':variant['name'],'rep':rep})
            for trajectory in trajectories:
                path=job_path(output,variant['name'],rep,trajectory)
                result=diagnose(trajectory,'',{},variant,cached,path)
                results.append({'variant':variant['name'],'repetition':rep,**result})
    counts={}
    for r in results:
        label=(r.get('prediction') or {}).get('category','<error>')
        counts[label]=counts.get(label,0)+1
    summary={'identity':identity,'measurement_kind':provider.measurement_kind,'input_count':len(trajectories),
             'paper_mast_count':1022,'gold_accuracy':None,'category_counts':counts,
             'note':'Unlabeled stress test; normalized IR required; counts pooled over configured variants/repetitions.',
             'results':results}
    write_json(output/'stress_summary.json',summary)
    return summary
