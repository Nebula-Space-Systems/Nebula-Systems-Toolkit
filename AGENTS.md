# AGENTS.md

## Purpose
Project-specific instructions for AI coding agents working in this repository.

## Environment
- OS and shell: Windows + PowerShell
- Python interpreter: `.venv\Scripts\python.exe`
- Prefer `rg` for fast file/text search.

## Repository Scope
- Main package code is in `nebula/`.
- Tests are in `tests/`.
- `nebula_old/` is legacy reference code; do not modify it unless explicitly requested.

## Coding Expectations
- Keep public behavior backward compatible unless a breaking change is explicitly requested.
- Keep dependencies minimal; avoid adding heavy dependencies unless justified by clear value.
- Keep changes focused to the user request; avoid unrelated refactors.
- Use clear, maintainable code over clever shortcuts.

## Test Expectations
- Add or update tests for behavior changes in `tests/`.
- Prefer targeted test runs before broad suites.
- Do not add plotting or interactive UI requirements to tests.

## Packaging Expectations
- `pyproject.toml` is the packaging source of truth.
- When dependencies change, update `pyproject.toml` and summarize why.

## Safety and Git Rules
- Never run destructive commands (for example `git reset --hard`) unless explicitly requested.
- Do not revert or overwrite unrelated local changes made by the user.
