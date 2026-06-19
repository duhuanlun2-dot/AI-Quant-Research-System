# Roadmap Issues

This roadmap mirrors the first issues that should be opened for the project. Each item is written so it can be copied directly into a GitHub issue.

## 1. PIT Vendor Support For Historical S&P 500 Universe

### Background

The system currently supports point-in-time universe ingestion and strict PIT checks, but users must provide a correctly shaped CSV manually. To reduce survivorship-bias risk, the project should support real vendor workflows more directly.

### Goal

Add first-class support for importing, validating, and documenting historical S&P 500 constituent data from legal PIT universe providers.

### Scope

- Define a vendor-neutral PIT universe ingestion contract.
- Add adapter/import docs for common providers such as Sharadar, Norgate, CRSP/Compustat-style exports, or user-provided vendor CSVs.
- Improve validation messages for entry/exit dates, removed members, active member counts, and ticker changes.
- Add sample sanitized fixture data that demonstrates removed constituents without redistributing proprietary data.
- Ensure `--require-pit-universe` fails loudly when vendor data is incomplete.

### Acceptance Criteria

- A documented PIT import workflow exists in README or docs.
- `check-universe` reports vendor coverage clearly.
- `build-factor-table --require-pit-universe` uses only point-in-time membership rows.
- Tests or fixture checks cover at least one removed-member scenario.
- No proprietary vendor data is committed.

### Risks / Notes

- Historical index membership is licensed data; this feature must not scrape or redistribute restricted datasets.
- Ticker changes and corporate actions need careful identifier handling.

## 2. Paper Trading Execution Layer

### Background

The system can generate signals, optimized weights, orders, fills, and current-position files. The next step toward live realism is a paper-trading loop that records intended orders and simulated broker/account state across runs.

### Goal

Add a paper-trading mode that converts the latest target weights into paper orders, tracks positions, and produces reviewable execution reports without placing real trades.

### Scope

- Add a `paper-trade` CLI command and matching GUI workflow.
- Persist paper orders, fills, positions, cash, and account equity in DuckDB.
- Support next-open / VWAP / close execution assumptions.
- Add order preview before execution.
- Add exportable paper blotter and current holdings reports.
- Add clear safeguards that this is not live brokerage execution.

### Acceptance Criteria

- Users can run paper trading from latest model/portfolio output.
- Paper orders are reproducible and stored in DuckDB.
- Current paper positions and cash can be inspected after each run.
- GUI shows paper-trading status and export files.
- No real broker API is called.

### Risks / Notes

- Paper fills can still be unrealistic; docs must describe assumptions.
- A later live broker integration should be isolated behind a separate adapter.

## 3. CI Tests And Packaging Checks

### Background

The project is now open source and needs automated checks to prevent packaging, CLI, and import regressions.

### Goal

Add GitHub Actions CI that validates packaging metadata, imports, CLI help, and a minimal demo pipeline.

### Scope

- Add GitHub Actions workflow for Python 3.10, 3.11, and 3.12.
- Run `python -m py_compile` or equivalent import checks.
- Run `pip install -e .`.
- Run `aiqrs --help` or `python -m ai_quant_research_system.cli --help`.
- Run a minimal fixture-based pipeline using example data.
- Ensure generated local files are ignored by git.

### Acceptance Criteria

- CI runs on pull requests and pushes to `main`.
- Package installs cleanly from `pyproject.toml`.
- CLI starts without import errors.
- Minimal fixture pipeline completes without network access.
- CI artifacts do not include secrets, local DBs, or model outputs.

### Risks / Notes

- Network-dependent collectors should not run in CI.
- Tests should use deterministic fixture data only.
