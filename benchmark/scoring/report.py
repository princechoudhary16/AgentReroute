"""Markdown summaries of attribution and recovery reports."""


def _fmt(metric):
    if metric is None:
        return '–'
    if isinstance(metric, dict):
        value, ci = metric.get('value'), metric.get('ci95', [None, None])
        if value is None:
            return '–'
        if ci and ci[0] is not None and ci[0] != ci[1]:
            return f'{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]'
        return f'{value:.2f}'
    if isinstance(metric, float):
        return f'{metric:.2f}'
    return str(metric)


ATTRIBUTION_COLUMNS = [('step_accuracy', 'Step'), ('step_within_1', 'Step ±1'), ('acceptable_step_accuracy', 'Acceptable'),
                       ('top3_step_accuracy', 'Top-3 step'), ('agent_accuracy', 'Agent'), ('mode_accuracy', 'Mode'),
                       ('mode_macro_f1', 'Mode macro-F1'), ('step_ece', 'Step ECE')]
RECOVERY_COLUMNS = [('recovery_rate', 'Recovery'), ('harmful_intervention_rate', 'Harm'),
                    ('false_intervention_rate', 'False intervention'), ('tokens_per_episode', 'Tokens/episode'),
                    ('tokens_per_recovered', 'Tokens/recovered'), ('cost_vs_original', 'Cost (× one run)'),
                    ('attempts_per_episode', 'Attempts'), ('budget_breaches', 'Budget breaches')]


def attribution_table(reports):
    lines = ['| System | n | ' + ' | '.join(c for _, c in ATTRIBUTION_COLUMNS) + ' |',
             '|---|---:|' + '---:|' * len(ATTRIBUTION_COLUMNS)]
    for r in reports:
        lines.append(f'| {r["system"]} | {r["denominator"]} | ' +
                     ' | '.join(_fmt(r['metrics'].get(k)) for k, _ in ATTRIBUTION_COLUMNS) + ' |')
    return '\n'.join(lines)


def recovery_table(reports):
    lines = ['| Policy | Failed | Controls | ' + ' | '.join(c for _, c in RECOVERY_COLUMNS) + ' |',
             '|---|---:|---:|' + '---:|' * len(RECOVERY_COLUMNS)]
    for r in reports:
        lines.append(f'| {r["policy"]} | {r["n_failed"]} | {r["n_controls"]} | ' +
                     ' | '.join(_fmt(r['metrics'].get(k)) for k, _ in RECOVERY_COLUMNS) + ' |')
    return '\n'.join(lines)


def markdown(attribution=(), recovery=(), title='AgentReroute benchmark report', note=None):
    parts = [f'# {title}', '']
    if note:
        parts += [f'> {note}', '']
    if attribution:
        parts += ['## Attribution', '', 'Values are means with bootstrap 95% intervals; denominators are all '
                  'failed trajectories in the split.', '', attribution_table(attribution), '']
        detection = [r for r in attribution if 'detection' in r]
        if detection:
            parts += ['### Detection', '', '| System | Precision | Recall | False alarms | Premature | Delay |',
                      '|---|---:|---:|---:|---:|---:|']
            for r in detection:
                d = r['detection']
                parts.append(f'| {r["system"]} | {_fmt(d["precision"])} | {_fmt(d["recall"])} | '
                             f'{_fmt(d["false_alarm_rate"])} | {_fmt(d["premature_rate"])} | {_fmt(d["mean_delay_steps"])} |')
            parts.append('')
    if recovery:
        parts += ['## Recovery', '', recovery_table(recovery), '']
    return '\n'.join(parts)
