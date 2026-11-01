"""Single entry for the current P0/M1 research route."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for key,name in {'HF_HOME':'hf','TORCH_HOME':'torch','PIP_CACHE_DIR':'pip','TMP':'tmp','TEMP':'tmp','TMPDIR':'tmp',
                 'XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib','TORCH_EXTENSIONS_DIR':'torch_extensions',
                 'TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton'}.items():
    p=ROOT/'.cache'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor/pfrpn'),str(ROOT/'versions/joint_detector'),
              str(ROOT/'versions/joint_network_merge'),str(ROOT/'versions/project_cleanup')]
if __name__=='__main__':
    from current.cli import main
    main(ROOT)
