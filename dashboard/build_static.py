import argparse
import shutil
from pathlib import Path

# Copies the static dashboard (HTML/JS/CSS + data) into a docs folder for GitHub Pages hosting.


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, clean: bool = False):
    if clean and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def build(static_dir: Path, data_dir: Path, docs_dir: Path, clean: bool = False):
    if not static_dir.exists():
        raise SystemExit(f"Static directory not found: {static_dir}")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    docs_dir.mkdir(parents=True, exist_ok=True)
    copy_file(static_dir / "index.html", docs_dir / "index.html")
    copy_file(static_dir / "app.js", docs_dir / "app.js")
    copy_file(static_dir / "style.css", docs_dir / "style.css")
    copy_tree(data_dir, docs_dir / "data", clean=clean)
    print(f"Exported dashboard to {docs_dir}")


def parse_args():
    base = Path(__file__).resolve().parent
    default_static = base / "static"
    default_data = default_static / "data"
    default_docs = Path(__file__).resolve().parent.parent / "docs"

    p = argparse.ArgumentParser(description="Build docs/ folder for GitHub Pages hosting.")
    p.add_argument("--static-dir", type=Path, default=default_static, help="Path to dashboard/static")
    p.add_argument("--data-dir", type=Path, default=default_data, help="Path to dashboard/static/data")
    p.add_argument("--docs-dir", type=Path, default=default_docs, help="Destination docs folder")
    p.add_argument("--clean", action="store_true", help="Remove existing docs/data before copying")
    return p.parse_args()


def main():
    args = parse_args()
    build(static_dir=args.static_dir, data_dir=args.data_dir, docs_dir=args.docs_dir, clean=args.clean)


if __name__ == "__main__":
    main()
