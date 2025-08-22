#!/usr/bin/env python3
"""
Reproducible Notebook Runner

Executes notebooks in notebooks/ headlessly and writes executed copies to
outputs/notebooks/, with optional HTML export if nbconvert is available.

Usage examples:
  python tools/run_notebooks.py                       # run all notebooks
  python tools/run_notebooks.py --pattern "02_*.ipynb" # run subset
  python tools/run_notebooks.py --html                # also export HTML

Notes:
  - Requires nbformat and nbclient. If missing, install:
      python -m pip install nbformat nbclient
  - HTML export requires nbconvert:
      python -m pip install nbconvert jupyterlab_pygments
  - Matplotlib is forced to headless (Agg) for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Force headless matplotlib for notebook execution
os.environ.setdefault("MPLBACKEND", "Agg")


def _require_packages() -> Tuple[object, object]:
    try:
        import nbformat  # type: ignore
        from nbclient import NotebookClient  # type: ignore
    except Exception as e:
        print("[ERROR] Missing required packages for notebook execution.")
        print("Install them with: python -m pip install nbformat nbclient")
        raise
    return nbformat, NotebookClient


def _maybe_get_html_exporter():
    try:
        from nbconvert import HTMLExporter  # type: ignore
        return HTMLExporter
    except Exception:
        return None


@dataclass
class RunnerConfig:
    pattern: str
    output_dir: Path
    html: bool
    timeout: int
    kernel: str
    stop_on_error: bool


def discover_notebooks(pattern: str) -> List[Path]:
    base = Path("notebooks")
    if any(ch in pattern for ch in "*?[]"):
        paths = sorted(base.glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path(".").glob(pattern))
    else:
        # If pattern is a filename (no glob), look under notebooks/
        paths = [base / pattern]
    notebooks = [p for p in paths if p.exists() and p.suffix == ".ipynb"]
    if not notebooks:
        # Fallback to all notebooks
        notebooks = sorted(base.glob("*.ipynb"))
    return notebooks


def execute_notebook(src: Path, dst_ipynb: Path, cfg: RunnerConfig) -> Tuple[bool, str]:
    nbformat, NotebookClient = _require_packages()

    # Read notebook
    with src.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute
    print(f"[RUN] Executing {src} -> {dst_ipynb}")
    dst_ipynb.parent.mkdir(parents=True, exist_ok=True)

    try:
        client = NotebookClient(
            nb,
            timeout=cfg.timeout,
            kernel_name=cfg.kernel,
            resources={"metadata": {"path": str(src.parent.resolve())}},
        )
        client.execute()
    except Exception as e:
        msg = f"[FAIL] {src.name}: {e}"
        print(msg)
        if cfg.stop_on_error:
            raise
        # Still write the partial notebook for debugging
        with dst_ipynb.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return False, msg

    # Write executed notebook
    with dst_ipynb.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return True, "ok"


def export_html(src_ipynb: Path, dst_html: Path) -> Tuple[bool, str]:
    HTMLExporter = _maybe_get_html_exporter()
    if HTMLExporter is None:
        return False, "nbconvert not installed; skipping HTML export"

    try:
        import nbformat  # type: ignore
        with src_ipynb.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(nb)
        dst_html.parent.mkdir(parents=True, exist_ok=True)
        dst_html.write_text(body, encoding="utf-8")
        return True, "ok"
    except Exception as e:
        return False, str(e)


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebooks headlessly for reproducibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pattern", default="*.ipynb", help="Glob pattern under notebooks/ or absolute glob")
    parser.add_argument("--output-dir", default="outputs/notebooks", help="Directory to write executed notebooks")
    parser.add_argument("--html", action="store_true", help="Also export HTML (requires nbconvert)")
    parser.add_argument("--timeout", type=int, default=1200, help="Cell execution timeout in seconds")
    parser.add_argument("--kernel", default="python3", help="Kernel name to use")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error instead of continuing")

    a = parser.parse_args()
    return RunnerConfig(
        pattern=a.pattern,
        output_dir=Path(a.output_dir),
        html=a.html,
        timeout=a.timeout,
        kernel=a.kernel,
        stop_on_error=a.stop_on_error,
    )


def main() -> int:
    cfg = parse_args()
    notebooks = discover_notebooks(cfg.pattern)
    if not notebooks:
        print(f"[WARN] No notebooks matched pattern '{cfg.pattern}' under notebooks/")
        return 0

    executed_dir = cfg.output_dir
    html_dir = executed_dir.with_name(executed_dir.name + "_html")

    ok_count = 0
    fail_count = 0

    print(f"[INFO] Found {len(notebooks)} notebook(s) to run")
    for nb_path in notebooks:
        dst_ipynb = executed_dir / nb_path.name
        ok, msg = execute_notebook(nb_path, dst_ipynb, cfg)
        if ok:
            ok_count += 1
            if cfg.html:
                dst_html = html_dir / (nb_path.stem + ".html")
                h_ok, h_msg = export_html(dst_ipynb, dst_html)
                if h_ok:
                    print(f"[HTML] Exported {dst_html}")
                else:
                    print(f"[HTML] Skipped/failed for {nb_path.name}: {h_msg}")
        else:
            fail_count += 1

    print("\n[SUMMARY]")
    print(f"Executed: {ok_count} succeeded, {fail_count} failed")
    print(f"Executed notebooks dir: {executed_dir.resolve()}")
    if cfg.html:
        print(f"HTML exports dir:       {html_dir.resolve()}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

