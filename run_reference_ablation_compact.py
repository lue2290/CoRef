"""159 export adapter: unchanged experiments, compact metrics/summary/10-image ZIP."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'run.py'),run_name='_reference_ablation_compact_environment')
if __name__=='__main__':
    from current.progress_io import install
    install()
    from reference_ablation_export import main
    main(ROOT,sys.argv[1:] or ['export'])
