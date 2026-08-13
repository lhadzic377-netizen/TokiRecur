# Changelog

All notable changes to TokiRecur are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.33] - 2026-08-13

### Changed
- Hardware profiling drives backend detection and CPU tuning (`toki doctor`,
  `--profile`/`--no-profile`, 30-day profile TTL).
- Installers no longer download a model at install time; model selection is
  a separate, explicit step.
