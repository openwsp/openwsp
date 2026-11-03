# Codex transport capture

These synthetic requests and JSONL events were captured from the installed Codex
CLI 0.163.0 against a loopback-only Responses mock. The only key supplied was the
fixed, fake fixture key. No existing login or real model provider was used.

Regenerate in a new directory with:

```sh
python scripts/check_codex_transport.py --output-dir /tmp/new-codex-wire-check --resume
```

The fixture uses the pinned Astra catalog in `src/openwsp/editors/data/`, its
three audited tool-exposure overrides, and the exact neutral instruction line.
It first starts a persistent thread, then resumes that exact UUID. Both requests
contain no advertised tools or injected environment instructions. Model metadata
and these captures are evidence for CLI transport behavior, not a claim that a
subscription model response is scientifically equivalent to the Responses API.

Each mocked response reports 10 input tokens and 2 output tokens. The two CLI
`turn.completed` events report cumulative totals of 10/2 and 20/4 respectively.
The captured requests have no `max_output_tokens` or prompt-cache retention field.
The provider supplies deterministic `MOCK_OK` text; it performs no inference.
