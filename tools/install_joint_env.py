"""Add detector dependencies to the current interpreter; never replace PyTorch."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import json
import importlib.metadata
import re
import sysconfig
import hashlib
import tarfile
import signal
import importlib

ROOT=Path(__file__).resolve().parents[1]


def show_log_tail(path, start=0, lines=60):
    """Read only the current failed phase; bound memory for long pip logs."""
    path=Path(path)
    if not path.is_file():
        print('No installation log exists yet: '+str(path),flush=True)
        return
    size=path.stat().st_size
    with path.open('rb') as f:
        f.seek(max(start,size-65536))
        tail=f.read().decode('utf8',errors='replace').splitlines()[-lines:]
    print('\n--- Installation output (last lines) ---',flush=True)
    print('\n'.join(tail) if tail else '(The command produced no output.)',flush=True)
    print('--- End installation output ---\nFull log: '+str(path),flush=True)


def run_logged(cmd, cwd, env, log_file):
    """Tee subprocess diagnostics to PyCharm and disk, including stderr."""
    process=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,encoding='utf8',
                             errors='replace',bufsize=1,start_new_session=os.name!='nt')
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line,end='',flush=True)
            log_file.write(line);log_file.flush()
    except KeyboardInterrupt:
        # Stop this command's process group, including compiler children on Linux.
        if process.poll() is None:
            try:
                if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait()
            except ProcessLookupError:pass
        raise
    finally:
        process.stdout.close()
    return process.wait()


def installed_versions():
    return {d.metadata['Name'].lower().replace('_','-'):d.version
            for d in importlib.metadata.distributions() if d.metadata['Name']}


def protected_versions(packages):
    return {name:version for name,version in packages.items()
            if name in {'torch','torchvision','torchaudio','triton','pytorch-triton'}
            or name.startswith('nvidia-')}


def assert_core_unchanged(before):
    after=protected_versions(installed_versions())
    if after!=before:
        raise RuntimeError('Protected PyTorch/CUDA packages changed unexpectedly: '+repr(dict(before=before,after=after)))


def command_version(command):
    if not command:return None
    try:
        p=subprocess.run([str(command),'--version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         text=True,encoding='utf8',errors='replace',timeout=20)
        return p.stdout.strip() if p.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):return None


def probe():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    packages=installed_versions()
    nvcc=Path(CUDA_HOME)/'bin'/('nvcc.exe' if os.name=='nt' else 'nvcc') if CUDA_HOME else None
    if nvcc is None or not nvcc.is_file():nvcc=shutil.which('nvcc')
    compiler=shutil.which('cl' if os.name=='nt' else 'g++')
    result=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                torchvision=packages.get('torchvision'),torch_cuda=torch.version.cuda,
                mmcv=packages.get('mmcv'),mmcv_lite=packages.get('mmcv-lite'),
                cuda_home=CUDA_HOME,nvcc=str(nvcc) if nvcc else None,
                nvcc_version=command_version(nvcc),compiler=compiler,
                compiler_version=command_version(compiler) if os.name!='nt' else compiler,
                cuda_available=torch.cuda.is_available(),free_project_disk_bytes=shutil.disk_usage(ROOT).free)
    if result['cuda_available']:
        result.update(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
    return result


def validate_build_tools(info):
    if not info['cuda_available']:
        raise RuntimeError('CUDA GPU is unavailable. Keep the existing interpreter; enable the rented GPU and rerun 13.')
    if not info['nvcc_version'] or not info['compiler_version']:
        raise RuntimeError('Building MMCV needs nvcc and a C++ compiler. No Python/PyTorch replacement was attempted. Send the probe output above.')
    match=re.search(r'release\s+(\d+\.\d+)',info['nvcc_version'])
    if not match or match.group(1)!=info['torch_cuda']:
        raise RuntimeError('nvcc must match the current PyTorch CUDA toolkit version for this build. Send the probe output; do not change PyTorch.')
    if info['free_project_disk_bytes']<10*1024**3:
        raise RuntimeError('Need 10GiB free on the project data disk to build MMCV.')


def configure_build_path(env):
    """PyCharm's full Python path need not imply its scripts directory is on PATH."""
    scripts=sysconfig.get_path('scripts')
    additions=[str(Path(sys.executable).resolve().parent)]
    if scripts:additions.insert(0,scripts)
    env['PATH']=os.pathsep.join(additions+[env.get('PATH','')])
    ninja=shutil.which('ninja',path=env['PATH'])
    if ninja is None:
        importlib.invalidate_caches()
        try:
            import ninja as package
            folder=getattr(package,'BIN_DIR',None)
            if folder:env['PATH']=str(folder)+os.pathsep+env['PATH']
        except ImportError:pass
        ninja=shutil.which('ninja',path=env['PATH'])
    if ninja is None:
        raise RuntimeError('Ninja is installed but its executable is unavailable; stopped before starting a slow fallback build.')
    return ninja


def build_identity(info):
    fields=['python','torch','torchvision','torch_cuda','nvcc_version','compiler_version','capability']
    identity={key:info[key] for key in fields}
    identity.update(mmcv='2.2.0',platform=sys.platform,python_executable=sys.executable)
    return identity


def build_directory(cache,info):
    key=hashlib.sha256(json.dumps(build_identity(info),sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache)/'mmcv_build'/key


def extract_source(archive,folder):
    """Extract once; never overwrite an existing build's source or object files."""
    folder=Path(folder);archive=Path(archive)
    source=folder/'mmcv-2.2.0';marker=folder/'source_ready.json'
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if marker.is_file():
        if json.loads(marker.read_text(encoding='utf8'))['archive_sha256']!=archive_sha or not (source/'setup.py').is_file():
            raise RuntimeError('Persistent MMCV source changed; preserve the directory and inspect it before rebuilding: '+str(folder))
        return source
    folder.mkdir(parents=True,exist_ok=True)
    # Preserve files if extraction was interrupted. Identical archive members can
    # be reused without changing mtimes; conflicting files are never overwritten.
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>128*1024**2:
            raise RuntimeError('Unexpectedly large MMCV source archive')
        for member in members:
            target=(folder/member.name).resolve()
            if not target.is_relative_to(source.resolve()) or not (member.isdir() or member.isfile()):
                raise RuntimeError('Unexpected MMCV archive member: '+member.name)
        for member in members:
            target=folder/member.name
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            stream=tar.extractfile(member)
            content=stream.read()
            if target.exists():
                if target.read_bytes()!=content:raise RuntimeError('Existing MMCV source differs: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    if not (source/'setup.py').is_file():raise RuntimeError('MMCV setup.py not found in source archive')
    marker.write_text(json.dumps(dict(archive_sha256=archive_sha))+'\n',encoding='utf8')
    return source


OPS_CHECK="""
import torch
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
assert torch.cuda.is_available(), 'CUDA GPU unavailable'
v=torch.randn(1,4,1,4,device='cuda',requires_grad=True)
l=torch.rand(1,2,1,1,2,2,device='cuda',requires_grad=True)
w=torch.rand(1,2,1,1,2,device='cuda',requires_grad=True)
y=MultiScaleDeformableAttnFunction.apply(v,torch.tensor([[2,2]],device='cuda'),torch.tensor([0],device='cuda'),l,w,64)
y.square().mean().backward()
torch.cuda.synchronize()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [v,l,w])
print('MMCV CUDA forward/backward passed.')
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',action='store_true',help='Print current packages/compiler information without installing anything')
    p.add_argument('--show-log',action='store_true',help='Show the previous installation error without installing anything')
    args=p.parse_args()
    log=ROOT/'reports/joint_detector/environment_install.log'
    if args.show_log:
        show_log_tail(log);return
    print('Joint installer v4 - current interpreter; Ninja and persistent build files',flush=True)
    info=probe();print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    if args.probe:return
    if (sys.version_info[:2],info['torch'].split('+')[0],(info['torchvision'] or '').split('+')[0]) not in [
            ((3,10),'2.1.0','0.16.0'),((3,12),'2.8.0','0.23.0')]:
        raise RuntimeError('Unexpected Python/PyTorch/torchvision combination. No environment change attempted. Send the probe output.')
    if info['mmcv_lite']:
        raise RuntimeError('mmcv-lite conflicts with full MMCV; no packages changed. Send the probe output before resolving it.')
    before=protected_versions(installed_versions())
    out=ROOT/'reports/joint_detector';out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PIP_PROGRESS_BAR='off',
                                    PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key,name in {'TMP':'tmp','TEMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'pip',
                     'TORCH_EXTENSIONS_DIR':'torch_extensions','CUDA_CACHE_PATH':'cuda',
                     'XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib'}.items():
        folder=cache/name;folder.mkdir(parents=True,exist_ok=True);env[key]=str(folder)
    constraints=out/'protected_current_environment.txt'
    constraints.write_text(''.join(name+'=='+version+'\n' for name,version in sorted(before.items())),encoding='utf8')
    (out/'environment_before_install.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    pip=[sys.executable,'-m','pip','install','--no-cache-dir','-c',str(constraints)]
    with log.open('a',encoding='utf8') as f:
        def phase(label,cmd):
            print('\n'+label+'; full log: '+str(log),flush=True)
            f.write('\n'+label+'\n'+json.dumps(cmd)+'\n');f.flush()
            try:returncode=run_logged(cmd,ROOT,env,f)
            except OSError as exc:raise RuntimeError('Cannot start command: '+str(exc)) from None
            assert_core_unchanged(before)
            if returncode:
                raise RuntimeError(f'{label} failed (exit {returncode}); see the actual error above. Python/PyTorch/CUDA versions were preserved.')
        usable=info['mmcv'] in {'2.1.0','2.2.0'} and run_logged([sys.executable,'-c',OPS_CHECK],ROOT,env,f)==0
        if not usable:
            # Stop before installing anything if the current toolkit cannot build CUDA ops.
            validate_build_tools(info)
        phase('Install detector Python dependencies (existing torch versions locked)',pip+['-r',str(ROOT/'requirements-joint.txt')])
        if not usable:
            phase('Install MMCV build dependencies',pip+['setuptools>=69,<81','wheel','ninja','psutil'])
            ninja=configure_build_path(env)
            phase('Verify Ninja executable: '+ninja,[ninja,'--version'])
            env.update(MMCV_WITH_OPS='1',FORCE_CUDA='1',MAX_JOBS='2',
                       TORCH_CUDA_ARCH_LIST='.'.join(map(str,info['capability'])),NINJA_STATUS='[%f/%t | %es] ')
            if info['nvcc']:env['CUDA_HOME']=str(Path(info['nvcc']).resolve().parent.parent)
            build=build_directory(cache,info);build.mkdir(parents=True,exist_ok=True)
            (build/'environment.json').write_text(json.dumps(build_identity(info),indent=2)+'\n',encoding='utf8')
            downloads=build/'downloads';downloads.mkdir(exist_ok=True)
            wheels=build/'wheels';wheels.mkdir(exist_ok=True)
            archive=downloads/'mmcv-2.2.0.tar.gz'
            if not archive.is_file():
                phase('Download MMCV source (no compilation)',[sys.executable,'-m','pip','download',
                    '--no-cache-dir','-c',str(constraints),'--no-deps','--no-build-isolation','--no-binary=mmcv',
                    '--dest',str(downloads),'mmcv==2.2.0'])
            source=extract_source(archive,build/'source')
            print('Persistent compilation directory: '+str(source/'build'),flush=True)
            print('Same environment + same source: finished object files can be reused on retry. Do not start concurrent installers.',flush=True)
            ready=sorted(wheels.glob('mmcv-2.2.0-*.whl'))
            if not ready:
                phase('Build MMCV wheel in persistent source directory',[sys.executable,'-m','pip','wheel',
                    '--no-cache-dir','-c',str(constraints),'--no-build-isolation','--no-deps',
                    '--wheel-dir',str(wheels),str(source),'-v'])
                ready=sorted(wheels.glob('mmcv-2.2.0-*.whl'))
            if len(ready)!=1:raise RuntimeError('Expected one complete MMCV wheel in '+str(wheels))
            phase('Install locally built MMCV wheel',pip+['--no-deps','--force-reinstall',str(ready[0])])
        phase('Verify MMCV CUDA forward/backward',[sys.executable,'-c',OPS_CHECK])
    print('\nDependencies ready. Keep the same PyCharm interpreter: '+sys.executable,flush=True)
    print('Next: 14 Joint check assets and GPU, then 15 tests and 16 smoke. Full-model compatibility is still pending smoke.',flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\nInterrupted by Ctrl+C/stop. Persistent build files are kept under .cache/mmcv_build; rerun 13 with the same environment.',file=sys.stderr,flush=True)
        raise SystemExit(130) from None
    except RuntimeError as exc:
        print('\nSTOP: '+str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1) from None
