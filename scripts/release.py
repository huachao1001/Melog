#!/usr/bin/env python3
"""发布脚本：按提交历史自动升版本号，提交并打包 wheel。

版本升级规则（扫描自上次版本号变更以来的提交信息）：
- 大版本 +1：feat!/fix! 等带 "!" 的破坏性变更，或正文含 BREAKING CHANGE
- 小版本 +1：feat: 前缀（新功能）
- 补丁 +1：其余（fix:/docs:/chore: 等）

用法::

    python scripts/release.py                # 自动判断升级级别并打包
    python scripts/release.py --level minor  # 手动指定级别（major/minor/patch）
    python scripts/release.py --no-commit    # 只改版本号并打包，不提交
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_PATTERN = 'version = "'


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def current_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        sys.exit("pyproject.toml 中找不到 version 字段")
    return m.group(1)


def commits_since_last_bump() -> list[str]:
    """自上次改动 pyproject.toml version 行以来的提交（不含版本提交本身）。"""
    last = sh("git", "log", "--format=%H", f"-S{VERSION_PATTERN}", "--", "pyproject.toml")
    rng = f"{last}..HEAD" if last else "HEAD"
    log = sh("git", "log", "--format=%B%x00", rng)
    return [c for c in log.split("\0") if c.strip()]


def bump_level(commits: list[str]) -> str:
    for c in commits:
        if re.search(r"(?im)^(BREAKING CHANGE|\w+(\([^)]*\))?!)[:：]", c):
            return "major"
    for c in commits:
        if re.match(r"^feat(\([^)]*\))?[:：]", c):
            return "minor"
    return "patch"


def bump(version: str, level: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def set_version(new: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    PYPROJECT.write_text(
        re.sub(r'^(version\s*=\s*)"[^"]+"', rf'\1"{new}"', text, count=1, flags=re.M),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", choices=["major", "minor", "patch"], default="auto")
    ap.add_argument("--no-commit", action="store_true", help="只改版本号并打包，不做版本提交")
    args = ap.parse_args()

    old = current_version()
    commits = commits_since_last_bump()
    level = bump_level(commits) if args.level == "auto" else args.level
    new = bump(old, level)
    set_version(new)

    if not args.no_commit:
        sh("git", "add", "pyproject.toml")
        sh("git", "commit", "-m", f"chore: 版本升至 v{new}")
        sh("git", "tag", f"v{new}")

    sh(sys.executable, "-m", "build", "--wheel")
    wheel = ROOT / "dist" / f"melog-{new}-py3-none-any.whl"
    print(f"v{old} -> v{new}（{level}，{len(commits)} 条提交）\n{wheel}")


if __name__ == "__main__":
    main()
