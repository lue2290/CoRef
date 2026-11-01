"""158: retrain corrected Game Reference Net v1 parents, then evaluate F0/F1."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'), run_name='_reference_game_environment')
if __name__ == '__main__':
    from current.progress_io import install
    install()
    import faulthandler
    faulthandler.enable()
    from refnet_game.experiment import main
    main(ROOT, sys.argv[1:] or ['run'])
