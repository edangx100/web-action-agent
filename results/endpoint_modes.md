# Endpoint modes

The same task and the same code, reaching a model three ways: through an aggregator, through the model maker's own service, and through a server run by the user. Each arm is one sweep with an empty skill library, and the arms are meant to differ only in `--profile`; the checks below test that against what each sweep recorded. Tokens and time are always shown. Dollars are shown only where every call had a price: a self-hosted server has no per-token price, and free trial access is not one, so those arms read unknown by design.

## Results

| Arm | Service | Planner model | Screenshot model | Tasks | Passed | Avg steps | Avg tokens | Avg time | Avg cost (agent) | Code | Started |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [local](modes-local.md) | `http://127.0.0.1:8000/v1` | `nvidia/nemotron-3-nano-30b-a3b` | none | cpf-employee-share | 1 of 1 | 1.0 | 2,607 | 3.9 s | unknown (1 call unpriced) | `6decfca45dc7` | 2026-10-20T06:02:59.529748+00:00 |
| [nvidia](modes-nvidia.md) | `https://integrate.api.nvidia.com/v1` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | cpf-employee-share | 1 of 1 | 1.0 | 2,685 | 11.0 s | unknown (1 call unpriced) | `6decfca45dc7` | 2026-10-20T06:03:50.992158+00:00 |
| [openrouter](modes-openrouter.md) | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-nano-30b-a3b` | `nvidia/nemotron-3.5-super-vl` | cpf-employee-share | 1 of 1 | 2.0 | 6,411 | 31.3 s | $0.0007 | `6decfca45dc7` | 2026-10-20T06:03:05.786814+00:00 |

## Did only the endpoint change?

| Check | Result |
|---|---|
| Same code in every arm | yes: every trial of every arm ran commit `6decfca45dc7`, with nothing uncommitted |
| Same tasks | yes: sg-gov-v1[cpf-employee-share] |
| Same judge | yes: openrouter-judge |
| Empty skill library in every arm | yes: cold |
| A different service in every arm | yes: 3 services |

A code version is the git commit the sweep ran, with `-dirty` added when the agent's code, settings, tasks or dependencies had changes not yet committed. One commit, clean, in every arm is what shows that only the configuration changed: keys live in `.env`, which is not part of the code.
