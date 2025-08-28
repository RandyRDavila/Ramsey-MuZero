# ramsey/main.py
from pathlib import Path
import runpy

def main():
    # Reuse the existing root-level main.py as the trainer entrypoint
    root_main = Path(__file__).resolve().parents[1] / "main.py"
    runpy.run_path(str(root_main), run_name="__main__")

if __name__ == "__main__":
    main()
