"""Run once after cloning: python setup_project.py"""

from pathlib import Path

DIRECTORIES = [
    "data/raw",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/product_db",
    "models",
    "config",
    "training",
    "inference",
    "scraper",
    "app",
    "utils",
    "tests",
    "scripts",
    "logs",
]

PACKAGES = [
    "training/__init__.py",
    "inference/__init__.py",
    "scraper/__init__.py",
    "utils/__init__.py",
    "tests/__init__.py",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    print(f"Setting up project at: {base}\n")

    for rel_path in DIRECTORIES:
        target = base / rel_path
        target.mkdir(parents=True, exist_ok=True)
        gitkeep = target / ".gitkeep"
        if not any(target.iterdir()):
            gitkeep.touch()

    for rel_path in PACKAGES:
        pkg = base / rel_path
        if not pkg.exists():
            pkg.touch()

    print(f"Created {len(DIRECTORIES)} directories and {len(PACKAGES)} __init__.py files.")
    print("\nNext:")
    print("  conda create -n beverage-cnn python=3.11 -y")
    print("  conda activate beverage-cnn")
    print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
