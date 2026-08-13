# Security Policy

## Reporting a vulnerability

Please report security issues privately to **<contact@tokirecur.com>**.
Do not open a public issue for something security-related.

Include, if you can:

* The affected version (`toki --version`)
* The steps to reproduce
* What you observed and what you expected

## Response

Reports are acknowledged within 48 hours. You'll get a status update as the
issue is triaged. Depending on severity, fixes ship in the next patch release,
and the issue may be disclosed after the fix lands.

## Scope

The TokiRecur daemon, CLI, API, and installers. If you're unsure whether
something counts, report it anyway.

## Non-goals

Bugs in llama-server itself (upstream project) should go to
https://github.com/ggml-org/llama.cpp. CVEs published for upstream dependencies
apply here only as far as our packaging is affected.
