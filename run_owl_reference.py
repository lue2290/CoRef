"""160: frozen OWL-ViT regional evidence, O0/O1 reference verification."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'),run_name='_owl_reference_environment')
if __name__=='__main__':
    from current.progress_io import install
    install()
    import faulthandler
    faulthandler.enable()
    from owl_reference.experiment import main
    main(ROOT,sys.argv[1:] or ['run'])
