"""159: all four controlled ablations and B0/full localization diagnostics."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'),run_name='_reference_ablation_environment')
if __name__=='__main__':
    from current.progress_io import install
    install()
    import faulthandler
    faulthandler.enable()
    from refnet_ablation.experiment import main
    main(ROOT,sys.argv[1:] or ['run'])
