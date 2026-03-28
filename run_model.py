from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    runpy.run_module("sif_model", run_name="__main__")