"""Nonblocking presentation adapter for immutable experiment source snapshots.

Only a daemon thread touches console streams. The experiment thread never writes,
flushes, joins a console worker, or holds its state lock during console I/O.
Install before starting an existing experiment, or import the aliases below in new
experiments. Numerical code, RNG state, checkpoints and experiment locks are intact.
"""
from collections import deque
import shutil
import threading
import time
from . import progress as legacy


class NonBlockingDisplay:
    def __init__(self):
        self.lock=threading.RLock()
        self.stack=[]
        self.pending=deque(maxlen=8)
        self.thread=None
        self.wake=threading.Event()
        self.idle=threading.Event();self.idle.set()
        self.width=0
        self.last_stream=None

    def _ensure_worker(self):
        if self.thread is None:
            self.thread=threading.Thread(target=self._run,name='experiment-console',daemon=True)
            self.thread.start()

    def add(self,bar):
        with self.lock:
            self.stack.append(bar);self.idle.clear();self._ensure_worker()
        self.wake.set()

    def _snapshot(self,bar):
        parents=[p.description for p in self.stack if p is not bar]
        return dict(description=bar.description,parents=parents[-1:],total=bar.total,
                    done=bar.done,unit=bar.unit,elapsed=time.monotonic()-bar.started,
                    closed=bar.closed,state=bar.state,message=bar.message,stream=bar.stream)

    def remove(self,bar):
        with self.lock:
            if bar not in self.stack:return
            if len(self.stack)==1:
                self.pending.append(self._snapshot(bar))
            self.stack.remove(bar)
        self.wake.set()

    @staticmethod
    def _format(frame):
        total=frame['total'];done=frame['done'];elapsed=frame['elapsed']
        rate=done/max(elapsed,1e-9);eta=None
        if total is None:
            body=frame['state'] if frame['closed'] else '|/-\\'[int(elapsed*4)%4]+' working'
        else:
            fraction=min(1.,done/total) if total else 1.
            filled=int(fraction*16)
            body='['+'='*filled+('>' if filled<16 else '')+'.'*max(0,15-filled)+']'
            body+=f" {fraction*100:5.1f}% {done}/{total} {frame['unit']}"
            eta=(total-done)/rate if rate else None
        suffix=f' | t {legacy.duration(elapsed)} ETA {legacy.duration(eta)}'
        if total is not None and rate:suffix+=f" {rate:.1f} {frame['unit']}/s"
        if frame['message']:suffix+=' | '+frame['message']
        label=' / '.join(frame['parents']+[frame['description']]).replace('\n',' ').replace('\r',' ')
        columns=max(80,shutil.get_terminal_size((140,24)).columns)
        label=label[:max(16,columns-len(body)-len(suffix)-3)]
        line=label+' '+body+suffix
        if frame['closed'] and frame['state']!='done':line+=' ['+frame['state']+']'
        return line

    def _run(self):
        while True:
            self.wake.wait(.25);self.wake.clear()
            # Snapshot under the lock; formatting AND all stream operations are outside.
            with self.lock:
                if self.pending:frame=self.pending.popleft()
                elif self.stack:frame=self._snapshot(self.stack[-1])
                else:self.idle.set();continue
            try:
                line=self._format(frame);stream=frame['stream']
                padding=max(0,self.width-len(line)) if stream is self.last_stream else 0
                stream.write('\r'+line+' '*padding+('\n' if frame['closed'] else ''))
                stream.flush()
                self.width=0 if frame['closed'] else len(line);self.last_stream=stream
            except (OSError,ValueError):pass
            with self.lock:
                if not self.stack and not self.pending:self.idle.set()
                more=bool(self.pending)
            if more:self.wake.set()

    def wait_idle(self,timeout=.5):
        """Bounded display drain for tests/exit only. Never called by work updates."""
        return self.idle.wait(timeout)


def install():
    display=legacy._DISPLAY
    if isinstance(display,NonBlockingDisplay):return display
    if display.stack:raise RuntimeError('Install console adapter before starting progress bars')
    legacy._DISPLAY=NonBlockingDisplay()
    return legacy._DISPLAY


Progress=legacy.Progress
track=legacy.track
stage=legacy.stage
status=legacy.status
