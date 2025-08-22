import re
import sys
from pathlib import Path


def find_missing_paths(index_html: Path, docs_dir: Path) -> list[str]:
    html = index_html.read_text(encoding="utf-8", errors="ignore")
    hrefs = re.findall(r'href="([^"]+)"', html)
    internal = [h for h in hrefs if not h.startswith("http") and not h.startswith("#")]
    missing = []
    for rel in internal:
        p = (docs_dir / rel).resolve()
        if not p.exists():
            missing.append(rel)
    return missing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs = repo_root / "docs"
    index_html = docs / "index.html"

    required = [
        index_html,
        docs / "assets" / "styles.css",
        docs / "assets" / "logo.svg",
        docs / "assets" / "favicon.svg",
        docs / "showcase",
    ]
    missing_required = [str(p) for p in required if not p.exists()]
    if missing_required:
        print("MISSING_REQUIRED:" + "|".join(missing_required))
        return 2

    missing_links = find_missing_paths(index_html, docs)
    if missing_links:
        print("BROKEN_LINKS:" + "|".join(missing_links))
        return 3

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())


