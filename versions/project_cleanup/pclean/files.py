from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import zipfile


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def signed(value):
    return dict(value, identity=fingerprint(value))


def check_identity(value):
    require(value.get('identity') == fingerprint({k: v for k, v in value.items() if k != 'identity'}), 'Manifest identity differs')
    return value


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def reparse(path):
    value = Path(path).lstat()
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, 'st_file_attributes', 0) & 0x400)


def safe(root, relative, must_exist=True):
    root = Path(root).absolute()
    require(root.exists() and not reparse(root), 'Allowed root is missing or a reparse point: ' + str(root))
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and '..' not in rel.parts, 'Unsafe relative path')
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            require(not reparse(cursor), 'Reparse/symlink traversal refused: ' + str(cursor))
    resolved = cursor.resolve()
    require(root.resolve() in resolved.parents, 'Path escaped its allowed root')
    if must_exist:
        require(resolved.is_file(), 'Expected regular file: ' + str(resolved))
    return resolved


def walk(root):
    root = Path(root)
    if not root.exists() or reparse(root):
        return
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if not reparse(Path(folder) / n)]
        for name in names:
            p = Path(folder) / name
            if not reparse(p) and p.is_file():
                yield p


def file_record(root, path):
    rel = Path(path).relative_to(root).as_posix(); path = safe(root, rel)
    before = path.stat(); digest = sha(path); after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing: ' + str(path))
    return dict(path=rel, bytes=after.st_size, sha256=digest, mtime_ns=after.st_mtime_ns,
                device=after.st_dev, inode=after.st_ino, links=after.st_nlink)


def unchanged(root, row):
    path = safe(root, row['path']); st = path.stat()
    require(st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns'] and sha(path) == row['sha256'], 'Changed file; no deletion: ' + str(path))
    return path


def make_archive(path, members):
    """members: archive name -> verified regular file path; never extracts ZIPs."""
    path = Path(path); require(not path.exists(), 'Archive already exists')
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.zip.tmp'); hashes = {}
    with zipfile.ZipFile(tmp, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, source in sorted(members.items()):
            before = sha(source); z.write(source, name)
            require(sha(source) == before, 'Source changed during archival: ' + str(source))
            hashes[name] = before
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(dict(files=hashes), ensure_ascii=False, indent=2))
    tmp.replace(path); check_archive(path)
    return dict(path=str(path), sha256=sha(path), files=hashes)


def check_archive(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist(); require(len(names) == len(set(names)) and z.testzip() is None, 'Archive CRC/duplicate failure')
        hashes = json.loads(z.read('ARCHIVE_MANIFEST.json'))['files']
        require(set(names) == set(hashes) | {'ARCHIVE_MANIFEST.json'}, 'Archive membership failure')
        for name, want in hashes.items():
            require(hashlib.sha256(z.read(name)).hexdigest() == want, 'Archive SHA failure: ' + name)
    return hashes
