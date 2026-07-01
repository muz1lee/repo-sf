# GitHub 发布检查清单

把 `real2sim_scene_foundry` 推到 GitHub 前，按这个清单检查。

## 1. 验证仓库

在仓库根目录运行：

```bash
source .venv/bin/activate
python -m pytest tests -q
rsf --help
```

期望结果：测试通过，CLI 能打印 help。

## 2. 检查将要提交的文件

```bash
git status --short
git status --short --ignored
```

应该提交的文件：

- `src/real2sim_scene_foundry/**/*.py`
- `tests/**/*.py`
- `scripts/*.sh` 和 `scripts/README.md`
- `README.md`
- `.gitignore`
- `.env.example`
- `pyproject.toml`
- `AGENTS.md`、`AI_START_HERE.md`、`PLAN.md`
- `docs/**/*.md`
- `.github/workflows/tests.yml`

不应该提交的文件：

- `runs/`
- `outputs/` 或 `output/`
- `.venv/`、`.venv_3dgs/`
- `.qwen_env.local`、`.env`、`.env.*`，但 `.env.example` 除外
- `__patch_incoming__/`
- 模型 checkpoint 和权重
- 生成的视频、mesh、点云、USD 文件、截图和渲染图

## 3. 做一次轻量 secret scan

这个命令只扫描 tracked 文件和未被 ignore 的 untracked 文件。它不能替代 GitHub secret scanning。

```bash
git ls-files -co --exclude-standard -z |
  xargs -0 grep -InE \
    'api[_-]?key|secret|token|password|passwd|BEGIN PRIVATE KEY|sk-|AKIA|xoxb-' ||
  true
```

逐条检查命中结果。文档里的占位变量名和测试里的假 key 可以接受；真实凭证不能提交。

## 4. 决定 private 还是 public

本项目包含服务器路径、SSH alias、服务注册表和 run 记录。这些信息适合 private repo。若要 public，先脱敏：

- 服务器 hostname 和 SSH alias
- 内部文件系统路径
- cloud worker 地址
- 不应公开的服务 URL
- 绑定私有数据集的 run 名称

## 5. 提交

推荐命令：

```bash
git add .gitignore .env.example README.md pyproject.toml AGENTS.md AI_START_HERE.md PLAN.md docs scripts src tests .github
git status --short
git commit -m "chore: prepare real2sim scene foundry for GitHub"
```

如果 `git status --short` 里出现生成产物，先停下来更新 `.gitignore`。

## 6. 推送

创建 GitHub repo 后：

```bash
git remote add origin git@github.com:<your-org-or-user>/real2sim_scene_foundry.git
git push -u origin m2-m4-qwen-sam3d-align
```

如果要把当前分支设为默认分支，可以 rename 或 merge：

```bash
git branch -M main
git push -u origin main
```
