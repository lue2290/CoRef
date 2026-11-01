"""163: measured current-GPU inference cost; no training."""
from pathlib import Path
import runpy, sys
ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'),run_name='_timing_environment')
if __name__=='__main__':
    from current.progress_io import install
    install()
    from reference_f1_timing import main
    main(ROOT,sys.argv[1:] or ['run'])
