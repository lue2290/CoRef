"""157: frozen Reference Net v1, F0 versus independent-threshold F1."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'), run_name='_reference_f1_environment')
if __name__ == '__main__':
    from current.progress_io import install
    install()
    import faulthandler
    faulthandler.enable()
    from refnet_f1.experiment import main
    main(ROOT, sys.argv[1:] or ['formal'])
