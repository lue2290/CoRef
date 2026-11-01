"""162: read-only, frozen157 top600 candidate diagnosis."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'), run_name='_f1_candidate_environment')
if __name__ == '__main__':
    from current.progress_io import install
    install()
    import faulthandler
    faulthandler.enable()
    from reference_f1_candidates import main
    main(ROOT, sys.argv[1:] or ['run'])
