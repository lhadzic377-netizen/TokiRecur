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

The TokiRecur daemon, CLI, API, web UI, and installers. If you're unsure
whether something counts, report it anyway.

The web UI has no login of its own: it proxies to the daemon with the
daemon's token, so anyone who can reach the UI can use the daemon through
it. That is a property of the shipped design, not a bug report — but a
deployment that exposes it to a network it should not reach is worth
telling us about.

## Non-goals

Bugs in llama-server itself (upstream project) should go to
https://github.com/ggml-org/llama.cpp. CVEs published for upstream dependencies
apply here only as far as our packaging is affected.
