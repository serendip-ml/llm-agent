# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added

- Agent framework with trait-based architecture (LLM, Storage, Rating, Learn, Directive traits)
- Jokester-P agent: joke generation with novelty checking and quality rating
- LLM trait with multi-backend support via llm-infer
- Storage trait with PostgreSQL backend and schema migrations
- Rating trait for automated LLM-based content evaluation
- Learn trait for training data collection (SFT/DPO)
- Training infrastructure with SFT and DPO support via llm-kelt
- CLI tools for agent management, training, and statistics
- HTTP server for running agents as services
- JSON cleaner for robust LLM output parsing

### Changed

- Renamed package from llm-agent to llm-gent
- Migrated from llm-learn to llm-kelt for training
- Refactored training infrastructure to core modules

[Unreleased]: https://github.com/serendip-ml/llm-gent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/serendip-ml/llm-gent/releases/tag/v0.1.0
