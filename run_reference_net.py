from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / 'run.py'), run_name='_reference_environment')
if __name__ == '__main__':
    from refnet.cli import main
    main(ROOT)
