"""检查 markdown 文件的代码块围栏是否配对。
"""
from __future__ import annotations

import sys
from pathlib import Path


def check_file(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    fences = [i + 1 for i, l in enumerate(lines) if l.strip().startswith("```")]
    n = len(fences)
    ok = (n % 2 == 0)
    tag = "OK" if ok else "未闭合"
    print(path.name + " : " + str(n) + " 个围栏 -> " + tag)
    if not ok:
        print("  围栏行号: " + str(fences))
        last = fences[-1]
        print("  第 " + str(last) + " 行未闭合")
        print("  该行内容: " + lines[last - 1][:50])
        print("  之后 12 行预览:")
        for j in range(last, min(last + 12, len(lines))):
            print("    " + str(j + 1) + ": " + lines[j][:60])
        print("  修复: 在该代码块末尾补 ```")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python check_md_fences.py file.md")
        return 1
    bad = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print("跳过 不存在: " + str(p))
            continue
        if check_file(p):
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
