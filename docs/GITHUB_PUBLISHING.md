# GitHub Publishing Checklist

Use this checklist before pushing `real2sim_scene_foundry` to GitHub.

## 1. Verify The Repo

Run from the repository root:

```bash
source .venv/bin/activate
python -m pytest tests -q
rsf --help
```

Expected: tests pass and the CLI prints command help.

## 2. Review What Will Be Committed

```bash
git status --short
git status --short --ignored
```

Files that should be committed:

- `src/real2sim_scene_foundry/**/*.py`
- `tests/**/*.py`
- `scripts/*.sh` and `scripts/README.md`
- `README.md`
- `.gitignore`
- `.env.example`
- `pyproject.toml`
- `AGENTS.md`, `AI_START_HERE.md`, `PLAN.md`
- `docs/**/*.md`
- `.github/workflows/tests.yml`

Files that should not be committed:

- `runs/`
- `outputs/` or `output/`
- `.venv/`, `.venv_3dgs/`
- `.qwen_env.local`, `.env`, `.env.*` except `.env.example`
- `__patch_incoming__/`
- model checkpoints and weights
- generated videos, meshes, point clouds, USD files, screenshots, and rendered images

## 3. Run A Secret Scan

This is a lightweight scan over tracked files plus untracked files that are not ignored. It is not a substitute for GitHub secret scanning:

```bash
git ls-files -co --exclude-standard -z |
  xargs -0 grep -InE \
    'api[_-]?key|secret|token|password|passwd|BEGIN PRIVATE KEY|sk-|AKIA|xoxb-' ||
  true
```

Review every hit. Documentation and placeholder variable names are fine; real credentials are not.

## 4. Decide Public vs Private

This project currently contains environment-specific server paths, SSH aliases, and service registry notes in the agent handoff docs. That is acceptable for a private repo. For a public repo, sanitize:

- server hostnames and SSH aliases
- internal filesystem paths
- cloud worker addresses
- service URLs that should not be public
- run names tied to private datasets

## 5. Commit

Recommended first commit after review:

```bash
git add .gitignore .env.example README.md pyproject.toml AGENTS.md AI_START_HERE.md PLAN.md docs scripts src tests .github
git status --short
git commit -m "chore: prepare real2sim scene foundry for GitHub"
```

If `git status --short` shows generated artifacts, stop and update `.gitignore` before committing.

## 6. Push

Create the GitHub repository first, then:

```bash
git remote add origin git@github.com:<your-org-or-user>/real2sim_scene_foundry.git
git push -u origin m2-m4-qwen-sam3d-align
```

If this should become the default branch, rename or merge it after the first push:

```bash
git branch -M main
git push -u origin main
```
