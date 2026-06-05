# 🛡️ Engineering Runbook — Best Practices & Anti-Patterns

> **Audience:** Solo developer or small team working on a data-heavy ML/RAG pipeline.
> **Scope:** Covers Git, data safety, pipeline execution, secrets, code quality, debugging, and deployment.
> **Rule of thumb:** If a mistake would cost you more than 30 minutes to undo, there should be a rule here preventing it.

---

## Table of Contents

1. [Git & Version Control](#1-git--version-control)
2. [Data Safety & Artifact Management](#2-data-safety--artifact-management)
3. [Pipeline Execution Protocol](#3-pipeline-execution-protocol)
4. [Secrets & API Key Management](#4-secrets--api-key-management)
5. [Code Quality & Hygiene](#5-code-quality--hygiene)
6. [Dependency Management](#6-dependency-management)
7. [Debugging & Troubleshooting](#7-debugging--troubleshooting)
8. [Testing & Validation](#8-testing--validation)
9. [Documentation Discipline](#9-documentation-discipline)
10. [Deployment & Environment](#10-deployment--environment)
11. [Anti-Pattern Hall of Shame](#11-anti-pattern-hall-of-shame)

---

## 1. Git & Version Control

### ✅ DO

| Practice | Why |
|---|---|
| **Commit early, commit often.** Small, atomic commits (one logical change per commit). | You can `git revert` a single bad commit instead of untangling a 500-line monster. |
| **Write meaningful commit messages.** Format: `type: short summary` (e.g., `fix: deduplicate chunks in embedding.py`). | Six months from now, `git log` is your only history. `"fixed stuff"` tells you nothing. |
| **Branch before experimenting.** Create a feature branch (`git checkout -b v2/phase-0.5`) before any risky change. | If the experiment fails, `git checkout master` and you're back to safety in 2 seconds. |
| **Use `git status` before every commit.** Read the file list. Make sure nothing unexpected is staged. | Prevents accidentally committing `.env`, `chroma_db/`, or 500MB of venv files. |
| **Use `git diff --staged` before every commit.** Read your own changes one last time. | Catches leftover `print("DEBUG")`, hardcoded API keys, or accidental deletions. |
| **Tag stable milestones.** `git tag v1.0-baseline` after a working version. | Gives you an instant rollback point. `git checkout v1.0-baseline` restores everything. |
| **Pull before you push** (if collaborating). `git pull --rebase origin main`. | Avoids merge conflicts and keeps history linear. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Committing secrets** (`.env`, API keys, passwords). | Even if you delete them in the next commit, they live in Git history **forever**. Attackers scrape GitHub for this. You'd need `git filter-branch` or BFG Repo Cleaner to purge them — painful. |
| **Committing generated/binary files** (`chroma_db/`, `__pycache__/`, `.pyc`). | Bloats the repo, causes merge conflicts, and the files are reproducible from code anyway. |
| **Force-pushing to `main`/`master`** (`git push --force`). | Rewrites shared history. If anyone else pulled, their local repo is now broken. |
| **Working directly on `main` for experiments.** | One bad experiment and your stable codebase is corrupted. Branches are free — use them. |
| **Giant "WIP" commits** with 20 unrelated changes. | Impossible to revert one specific change. Impossible to review. Impossible to debug with `git bisect`. |
| **Ignoring `.gitignore` warnings.** If Git says a file is untracked and you don't recognize it, investigate before adding. | You might accidentally commit 2GB of vector embeddings or your virtual environment. |

---

## 2. Data Safety & Artifact Management

### ✅ DO

| Practice | Why |
|---|---|
| **Never delete raw source data.** The original PDF, `fulltext.txt`, `tables.json` are sacred. | These are your ground truth. Everything downstream (chunks, embeddings) can be regenerated from them. |
| **Version your processed artifacts.** Before re-chunking, copy: `chunks/ → chunks_v1_backup/` (or just rely on Git if they're tracked). | If the new chunking strategy is worse, you can instantly compare or rollback. |
| **Use `--reset` flags, not manual `rm -rf`.** Build reset logic into your scripts (like we did in `embedding.py`). | Script-controlled resets are logged, reproducible, and less error-prone than manual deletion. |
| **Log what was generated.** Print counts: `"✅ Generated 52 chunks"`, `"⏭️ Skipped 3 noisy blocks"`. | If something silently produces 0 chunks, you catch it immediately instead of debugging a broken retriever 3 hours later. |
| **Separate raw → processed → embedded directories.** `data/parsed_output/` vs `data/chroma_db/`. | Clear data lineage. You know exactly what stage each file belongs to. |
| **Add processed artifacts to `.gitignore` if they're large or reproducible.** ChromaDB, venv, and cached embeddings don't belong in Git. | Keeps the repo lean. Anyone can regenerate them by running the pipeline. |
| **Document your data lineage.** Write down: PDF → `ingestion.py` → `fulltext.txt` + `tables.json` → `preprocessing.py` → `clean_text.txt` + `processed_tables.md` → `chunking.py` → `chunks/`. | When something breaks at step 4, you know exactly which input file to inspect. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Deleting `clean_text.txt` or `processed_tables.md` before re-running preprocessing.** | If the re-run fails halfway, you've lost both the old output AND the new output. Always generate to a new file or let the script overwrite atomically. |
| **Manually editing generated files** (e.g., hand-fixing a chunk in `all_chunks.txt`). | The next time you run the pipeline, your manual edits are silently overwritten. Fix the source code, not the output. |
| **Assuming "it worked last time" means it's safe to delete the backup.** | Murphy's Law. Keep backups until you've verified the new output end-to-end (retrieval + generation). |
| **Storing large binary data in Git** (PDFs, ChromaDB, images). | Git stores every version forever. A 50MB PDF committed 10 times = 500MB repo. Use `.gitignore` and a shared drive or cloud storage. |

---

## 3. Pipeline Execution Protocol

### ✅ DO — The "Pre-Flight Checklist"

Before running any pipeline script that modifies data:

```
□ 1. git status          → Is my working tree clean? Have I committed my latest code changes?
□ 2. git stash           → If I have uncommitted changes I want to keep, stash them first.
□ 3. Verify inputs exist → Does clean_text.txt / processed_tables.md actually exist and look right?
□ 4. Dry-run if possible → Run with a --dry-run or --preview flag to see what WOULD happen.
□ 5. Check .env          → Is the correct API key loaded? Am I pointing at the right model?
□ 6. Run the script      → Execute.
□ 7. Verify outputs      → Open the output file. Check counts. Spot-check content.
□ 8. git add + commit    → Lock in the new state.
```

### ✅ DO — Execution Order Matters

| Rule | Example |
|---|---|
| **Always run the pipeline in order.** Ingestion → Preprocessing → Chunking → Embedding. | Running `chunking.py` on stale `clean_text.txt` (from before your preprocessing changes) gives you old chunks with new code. Confusing and wrong. |
| **Re-run ALL downstream steps after changing an upstream step.** | If you change `preprocessing.py`, you must re-run: preprocessing → chunking → embedding. Skipping chunking means your embeddings are built on old chunks. |
| **Use CLI flags to control destructive operations.** (`--reset`, `--force`, `--dry-run`) | Prevents accidental data loss. A script should never silently delete a database without explicit intent. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Running embedding before chunking is verified.** | You embed garbage → your retriever returns garbage → you debug for hours → the bug was in chunking all along. |
| **Running scripts from the wrong directory.** | Relative paths like `../data/` resolve differently. Always `cd` to project root or use `SCRIPT_DIR`-based absolute paths (like we already do). |
| **Running two pipeline steps simultaneously** (in separate terminals). | Race condition. Both scripts might try to write to the same output file. One wins, one loses, you get corrupted data. |
| **Ignoring warnings or "0 chunks generated" messages.** | A silent failure is the worst kind. If the script says 0, STOP. Don't proceed to the next step hoping it'll fix itself. |

---

## 4. Secrets & API Key Management

### ✅ DO

| Practice | Why |
|---|---|
| **Use `.env` files for all secrets.** Load with `python-dotenv`. | Keeps secrets out of source code. Easy to swap between dev/staging/prod keys. |
| **Add `.env` to `.gitignore` BEFORE your first commit.** (We already do this ✅) | If `.env` is committed even once, the secret is in Git history forever. |
| **Use a `.env.example` file** with dummy values (`OPENAI_API_KEY=sk-your-key-here`). | Tells collaborators which env vars are needed without exposing real values. |
| **Rotate keys immediately if exposed.** Go to the provider dashboard, revoke, and generate a new one. | Exposed keys get exploited within minutes by automated scrapers. |
| **Use different API keys for dev vs production.** | A runaway loop in dev doesn't burn through your production budget. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Hardcoding API keys in source code.** `api_key = "sk-abc123..."` | Instant security breach if the repo is ever made public (or even shared with one person). |
| **Printing API keys in logs.** `print(f"Using key: {api_key}")` | Logs get stored, shared, and indexed. Your key is now everywhere. |
| **Sharing `.env` files over Slack/email/Discord.** | Those messages are stored on third-party servers indefinitely. Use a password manager or encrypted channel. |
| **Using your production OpenAI key for testing.** | One infinite loop in your script and you wake up to a $500 bill. |

---

## 5. Code Quality & Hygiene

### ✅ DO

| Practice | Why |
|---|---|
| **Remove all debug `print()` statements before committing.** Or use `logging.debug()` which can be toggled off. | Debug prints clutter output, confuse other developers, and occasionally leak sensitive data. |
| **Use descriptive variable names.** `text_chunks` not `tc`. `processed_rows` not `pr`. | Code is read 10x more than it's written. `tc` means nothing 2 weeks later. |
| **Keep functions small and single-purpose.** One function = one job. | Easier to test, debug, and reuse. If `clean_text()` does 5 things, split it into 5 functions. |
| **Use constants for magic numbers.** `ALPHA_RATIO_THRESHOLD = 0.4` not `if ratio < 0.4`. | When you need to tune the threshold, you change one line at the top, not hunt through 200 lines of code. |
| **Handle errors explicitly.** `try/except` with specific exception types. | A bare `except:` swallows ALL errors silently, including typos and logic bugs. You'll debug for hours. |
| **Use type hints for function signatures.** `def chunk_text(text: str) -> list[Document]:` | Acts as built-in documentation. IDEs catch type mismatches before you even run the code. |
| **Use `if __name__ == "__main__":` guards.** (We already do this ✅) | Prevents the script from auto-executing when imported as a module by another script. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Leaving commented-out code blocks.** `# old_method()` with 30 lines of dead code. | That's what Git history is for. Commented code rots, confuses readers, and never gets cleaned up. |
| **Copy-pasting code between files.** | When you fix a bug in one copy, the other copy still has the bug. Extract shared logic into a utility function. |
| **Using bare `except:` or `except Exception:`.** | Hides the real error. At minimum, log the exception: `except Exception as e: print(f"Error: {e}")`. |
| **Mixing tabs and spaces.** | Python will throw `IndentationError` or, worse, silently misinterpret your code blocks. Use 4 spaces everywhere. Configure your editor. |
| **Using mutable default arguments.** `def func(items=[]):` | The list is shared across ALL calls. Appending to it in one call mutates it for the next call. Use `items=None` and `items = items or []`. |

---

## 6. Dependency Management

### ✅ DO

| Practice | Why |
|---|---|
| **Pin exact versions in `requirements.txt`.** `langchain==0.2.16` not `langchain`. | Without pinning, `pip install` grabs the latest version, which might have breaking API changes. Your code works today, breaks tomorrow. |
| **Use a virtual environment.** `python -m venv ragvenv` → `ragvenv\Scripts\Activate.ps1`. | Isolates your project's dependencies from system Python. Prevents version conflicts between projects. |
| **Update `requirements.txt` after installing new packages.** `pip freeze > requirements.txt` (or manually add the specific package). | If you install `rank_bm25` but don't add it to requirements, the next person (or you on a new machine) gets `ModuleNotFoundError`. |
| **Test after upgrading any dependency.** Upgrade one package → run pipeline → verify output. | A minor version bump in `sentence-transformers` broke our embedding pipeline (documented in DEV_LOG). Always verify. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Installing packages globally** (`pip install X` without activating venv). | Pollutes system Python, causes version conflicts, and makes the project non-reproducible. |
| **Running `pip install --upgrade` on everything at once.** | Multiple breaking changes hit simultaneously. You can't tell which upgrade broke your code. Upgrade one at a time. |
| **Committing the virtual environment** (`ragvenv/` in Git). | 500MB+ of binary files. Non-portable across OS. Just commit `requirements.txt` and let others `pip install -r`. |
| **Ignoring deprecation warnings.** `DeprecationWarning: X will be removed in v3.0` | When v3.0 drops and your CI auto-upgrades, your entire pipeline breaks overnight. Fix warnings proactively. |

---

## 7. Debugging & Troubleshooting

### ✅ DO

| Practice | Why |
|---|---|
| **Read the full error traceback, bottom to top.** The actual error is always the LAST line. | Beginners read the first line (which is just the call chain) and get confused. The fix is in the last 2 lines. |
| **Reproduce the bug with the smallest possible input.** | If chunking fails on 50 chunks, isolate which ONE chunk causes it. Debug that one chunk. |
| **Use `print()` strategically at pipeline boundaries.** Input → `print(len(text))` → Process → `print(len(chunks))` → Output. | Instantly narrows down WHERE data is being lost or corrupted. |
| **Check your inputs before blaming your code.** Open `clean_text.txt` and actually read it. | 80% of "code bugs" are actually "bad input data" bugs. |
| **Google the exact error message** (in quotes). | Someone has hit the same error before. Stack Overflow, GitHub Issues, and forums have the fix 90% of the time. |
| **Use `git diff` to see what changed.** "It was working yesterday" → `git diff HEAD~1` shows exactly what you changed. | The bug is almost always in your most recent changes. |
| **Rubber duck debugging.** Explain the problem out loud, line by line. | The act of verbalizing forces you to think through assumptions you'd otherwise skip. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Changing multiple things at once to "fix" a bug.** | If it works, you don't know which change fixed it. If it doesn't, you've introduced new variables. Change ONE thing, test, repeat. |
| **Debugging by random trial and error.** ("Let me just try adding `.strip()` everywhere...") | Wastes hours. Stop, read the error, form a hypothesis, test it. |
| **Silencing errors to make them "go away."** Adding `try: ... except: pass` around broken code. | The error is still happening. You've just hidden it. It will resurface later in a much harder-to-debug way. |
| **Spending more than 30 minutes on the same bug without taking a break.** | Diminishing returns. Walk away for 10 minutes. Your subconscious will often solve it. |

---

## 8. Testing & Validation

### ✅ DO

| Practice | Why |
|---|---|
| **Spot-check outputs after every pipeline run.** Open `all_chunks.txt`, read 5 random chunks. Do they make sense? | Automated pipelines can produce plausible-looking but subtly wrong output. Human eyes catch semantic errors that code can't. |
| **Keep a golden test set.** (We have 20 RAGAS Q&A pairs ✅) | Gives you an objective, repeatable benchmark. "Did V2 actually improve over V1?" is answered by numbers, not feelings. |
| **Test edge cases explicitly.** Empty input, single-word input, input with only numbers, input with Unicode. | Edge cases are where most bugs hide. A function that works on "normal" input can crash on empty strings. |
| **Compare before and after.** When changing chunking strategy, diff the old vs new `all_chunks.txt`. | Ensures you didn't accidentally lose content or introduce duplicates. |
| **Validate the full pipeline end-to-end**, not just individual scripts. Ask a real question and check the answer. | A unit test on `clean_text()` passing doesn't mean the final answer is correct. Integration matters. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Assuming the code is correct because it "runs without errors."** | A script that silently produces 0 chunks, or embeds the wrong text, runs without errors. Verify the OUTPUT, not just the exit code. |
| **Only testing the happy path.** ("It works with this one query!") | That one query might be the easiest case. Test the queries that FAILED in V1 — those are the ones that matter. |
| **Skipping evaluation after changes.** "I improved the chunking, it must be better now." | MUST re-run RAGAS (or equivalent) evaluation. Intuition is not data. Sometimes "better" chunking makes retrieval worse. |

---

## 9. Documentation Discipline

### ✅ DO

| Practice | Why |
|---|---|
| **Update `DEV_LOG.md` after every significant decision.** Not every commit — just architectural choices, trade-offs, and bugs. | In 6 months, you won't remember WHY you chose `gpt-4o-mini` over `gpt-4o`. The dev log tells you. |
| **Write docstrings for non-obvious functions.** | `def forward_fill_page_12(rows):` — without the docstring, nobody knows why this exists or what "forward fill" means in this context. |
| **Document your data format.** "Each chunk in ChromaDB has metadata: `{source, chunk_index, page, section}`." | When debugging retrieval, you need to know what metadata is available to filter on. |
| **Keep README.md updated** with setup instructions and architecture overview. | The README is the front door. If it's outdated, nobody (including future you) can get the project running. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Writing documentation after the project is "done."** | You will forget 80% of the decisions and context. Document as you go. |
| **Documenting HOW the code works** (the code already shows that). Document **WHY**. | `# Split text by newlines` adds zero value. `# Split by double-newline because single newlines appear inside paragraphs` is useful. |
| **Letting docs drift from reality.** If the code says `top_k=5` but the README says `top_k=3`, one of them is lying. | Misleading docs are worse than no docs. They actively send people down the wrong path. |

---

## 10. Deployment & Environment

### ✅ DO

| Practice | Why |
|---|---|
| **Always verify your virtual environment is active** before running scripts. Check `(ragvenv)` in your terminal prompt. | Running with system Python uses different package versions. Your code might work, might crash, might produce subtly different results. |
| **Use environment variables for configuration**, not hardcoded values. Model name, API base URL, top_k — all should come from `.env` or CLI args. | Lets you switch between dev/staging/prod without editing code. |
| **Test on a clean machine** (or fresh venv) before sharing. `python -m venv fresh_test` → `pip install -r requirements.txt` → run pipeline. | "Works on my machine" is not a valid deployment strategy. If a dependency is missing from requirements.txt, you'll find out here. |

### ❌ DON'T

| Anti-Pattern | Consequence |
|---|---|
| **Running production scripts with `python` instead of `python -m`** when using modules. | Subtle import path differences. `python src/chunking.py` vs `python -m src.chunking` resolve imports differently. |
| **Hardcoding absolute paths.** `C:\Users\sagar\FinanceRAG\data\...` | Breaks instantly on any other machine, any other OS, or even if you rename a folder. Use `os.path.join(SCRIPT_DIR, ...)`. |
| **Leaving the dev server running and forgetting about it.** (FastAPI on port 8000) | Port conflicts, stale code being served, confusion about which version is running. Kill servers when done. |

---

## 11. Anti-Pattern Hall of Shame

> The greatest lessons come from the worst mistakes. These are real-world disasters.

| # | Anti-Pattern | Real Consequence |
|---|---|---|
| 1 | **Committing `.env` with API keys to a public GitHub repo.** | Bots scrape GitHub in real-time. Keys are exploited within minutes. Bills of $10,000+ have been reported. |
| 2 | **Running `rm -rf data/` instead of `rm -rf data/chroma_db/`.** | Deleted raw PDFs, ground truth data, and all processed artifacts. Hours of re-parsing, if the original files even exist anymore. |
| 3 | **`git push --force` on `main` in a team project.** | Overwrote 3 teammates' merged work. Required manual reconstruction from local clones. |
| 4 | **Upgrading all pip packages at once before a demo.** | `sentence-transformers` broke mid-upgrade. Embedding pipeline crashed. Demo failed. |
| 5 | **Not re-running embedding after changing chunking.** | New chunks existed on disk, but ChromaDB still had the old embeddings. Retriever returned stale, wrong results for 2 weeks before anyone noticed. |
| 6 | **Using `except: pass` to silence a JSON parsing error.** | 40% of tables were silently dropped. The pipeline reported "success." Took days to discover why recall was so low. |
| 7 | **Editing `all_chunks.txt` by hand instead of fixing `chunking.py`.** | Changes were lost on the next pipeline run. The same bug reappeared. Wasted a full afternoon. |
| 8 | **Testing RAG with only easy queries.** ("What is calibration?") | Reported 95% accuracy. In production, users asked conversational questions → actual accuracy was 30%. |

---

## Quick Reference — The 10 Commandments

```
 1.  Thou shalt NEVER commit secrets.
 2.  Thou shalt NEVER delete raw source data.
 3.  Thou shalt ALWAYS branch before experimenting.
 4.  Thou shalt ALWAYS re-run downstream steps after upstream changes.
 5.  Thou shalt ALWAYS verify outputs, not just exit codes.
 6.  Thou shalt ALWAYS use a virtual environment.
 7.  Thou shalt ALWAYS pin dependency versions.
 8.  Thou shalt NEVER silence errors with bare except.
 9.  Thou shalt document WHY, not HOW.
10.  Thou shalt change ONE thing at a time when debugging.
```

---

*Last updated: 2026-06-03 | Tailored for the MedTech RAG project*
