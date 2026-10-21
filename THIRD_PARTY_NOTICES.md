# Attribution

This is an independent reconstruction of AgentRx, not Microsoft's original implementation.

The taxonomy descriptions and checklist in `agentrx/resources/taxonomy.json` are
adapted from Microsoft AgentRx `agentrx/judge/judge.py`, commit
`7a18c79708e7671be15124460f4f7296107c2a55`:
https://github.com/microsoft/AgentRx . Copyright (c) Microsoft Corporation.
The complete Microsoft MIT permission notice is included in
`LICENSE-Microsoft-AgentRx.txt`. The repository's original MIT license is
preserved in `LICENSE`.

`agentrx/resources/tools.json` is adapted from the same revision's
`agentrx/invariants/domain_registry.py`, under the same MIT license and copyright.

The prepare command imports original trajectories, annotations, retail policy,
and tool descriptions from the supplied authors' release. These files remain
under their source licenses; the Hugging Face benchmark declares CC BY 4.0:
https://huggingface.co/datasets/microsoft/AgentRx . Source hashes and revision
are retained in each corpus manifest. Imported data is not committed here.

Reference: Barke, Goyal, Khare, Singh, Nath, and Bansal (2026),
*AgentRx: Diagnosing AI Agent Failures from Execution Trajectories*,
arXiv:2602.02475v2. https://arxiv.org/abs/2602.02475
