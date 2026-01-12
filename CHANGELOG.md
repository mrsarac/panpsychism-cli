# Changelog

All notable changes to Project Panpsychism are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-08

### Added
- Caching layer with LRU eviction and TTL expiration
- Error recovery system with circuit breaker pattern
- Graceful shutdown handling for all services
- CI/CD workflows (lint, test, build, release)
- Comprehensive documentation suite
  - README.md with quick start guide
  - ARCHITECTURE.md with system design
  - API.md with complete reference
  - CONTRIBUTING.md with development guidelines
- Performance benchmarks and optimization
- Production-ready logging and monitoring hooks

### Changed
- Refined all 10 agent prompts for production use
- Optimized search algorithms for sub-100ms response times
- Improved error messages with actionable suggestions

### Fixed
- Memory leak in long-running indexer sessions
- Race condition in concurrent search requests
- Edge cases in privacy tier validation

## [0.3.0] - 2026-01-07

### Added
- Intelligence layer with 10 specialized AI agents
  - Indexer Agent: Prompt cataloging and metadata extraction
  - Search Agent: Semantic and keyword search orchestration
  - Validator Agent: Quality and consistency checking
  - Corrector Agent: Auto-fix suggestions and improvements
  - Analyzer Agent: Usage patterns and insights
  - Optimizer Agent: Performance tuning recommendations
  - Security Agent: Privacy and safety auditing
  - Translator Agent: Multi-language support
  - Versioner Agent: Change tracking and diff generation
  - Orchestrator Agent: Multi-agent coordination
- 255 comprehensive test cases across all modules
- Agent communication protocol (request/response/stream)
- Confidence scoring for all agent outputs
- Fallback chains for agent failures

### Changed
- Upgraded search to use hybrid semantic + keyword approach
- Enhanced validator with context-aware rules
- Improved corrector with LLM-powered suggestions

## [0.2.0] - 2026-01-05

### Added
- Core module: Indexer
  - File system traversal with glob patterns
  - Frontmatter parsing (YAML/TOML)
  - Content hashing for change detection
  - Incremental indexing support
- Core module: Search
  - Full-text search with ranking
  - Tag and category filtering
  - Date range queries
  - Pagination and sorting
- Core module: Validator
  - Schema validation for prompt structure
  - Required field checking
  - Format consistency rules
  - Custom validation rule support
- Core module: Corrector
  - Typo detection and suggestions
  - Format normalization
  - Missing field population
  - Batch correction mode

### Changed
- Refactored CLI to use modular command structure
- Improved configuration hot-reloading

## [0.1.0] - 2026-01-03

### Added
- CLI skeleton with subcommand architecture
  - `panpsychism index` - Index prompt files
  - `panpsychism search` - Search indexed prompts
  - `panpsychism validate` - Validate prompt structure
  - `panpsychism correct` - Auto-correct issues
- Configuration system
  - YAML/JSON config file support
  - Environment variable overrides
  - Default sensible settings
- Privacy tier system
  - Tier 0: Public (shareable)
  - Tier 1: Internal (team only)
  - Tier 2: Confidential (restricted)
  - Tier 3: Secret (encrypted)
- Basic project structure
  - TypeScript with strict mode
  - ESLint + Prettier configuration
  - Jest test framework setup
  - Package.json with scripts

### Security
- Implemented privacy tier enforcement
- Added file permission checks
- Sanitized user inputs in CLI

---

[1.0.0]: https://github.com/mrsarac/prompt-library/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/mrsarac/prompt-library/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mrsarac/prompt-library/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mrsarac/prompt-library/releases/tag/v0.1.0
