# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-14

### Added

- Initial version: recipe schema, resumable project/manifest state, coherence
  gate (ported from `minicpm-quant/coherence_check.py`), three backends
  (`llmcompressor`, `modelopt`, `gguf`), CLI (`recipe init` / `run` / `status`
  / `gate`). 18/18 tests passing (recipe parsing, resumable state including
  crash-recovery and stale-recipe rejection, gate degenerate-detection logic
  against the zaya1 pad-collapse and MiniCPM repetition-loop failure shapes).
  Not yet run against a real model / real GPU — see `ROADMAP.md` "Next".
