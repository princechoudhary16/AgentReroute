"""Illustrative design data. Replace with measured results when supplied.

These values are NOT benchmark findings, live-run output or performance promises.
Quality uses percentage points; efficiency uses AgentRx=100 relative units.
Keep provenance labels until measured data and its evaluation scope are supplied.
"""
BENCHMARKS = {
    'quality': [
        {'id': 'location', 'title': 'Find the decision that went wrong.',
         'description': 'Exact fault-location accuracy', 'unit': '%',
         'direction': 'Higher is better', 'agentrx': 45, 'agentreroute': 75},
        {'id': 'recovery', 'title': 'Turn the diagnosis into a completed task.',
         'description': 'Successful task recovery', 'unit': '%',
         'direction': 'Higher is better', 'agentrx': 60, 'agentreroute': 80},
    ],
    'efficiency': [
        {'id': 'cost', 'title': 'Spend less on the way to a result.',
         'description': 'Relative task cost · AgentRx = 100', 'unit': '',
         'direction': 'Lower is better', 'agentrx': 100, 'agentreroute': 70},
        {'id': 'time', 'title': 'Get back to the task sooner.',
         'description': 'Relative completion time · AgentRx = 100', 'unit': '',
         'direction': 'Lower is better', 'agentrx': 100, 'agentreroute': 80},
    ],
}


def reduction(metric):
    """Derive the illustrative relative reduction used by the impact cards."""
    return round(100 * (1 - metric['agentreroute'] / metric['agentrx']))
