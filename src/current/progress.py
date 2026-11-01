"""Shared single-line experiment progress for terminals and PyCharm consoles.

No third-party dependency. Nest track()/stage() freely: only the active phase is
rendered, using carriage returns. Counts advance only after real work completes.
EXPERIMENT_PROGRESS=0 disables console output without changing iteration/logging.
"""
import os
import shutil
import sys
import threading
import time


def duration(seconds):
    if seconds is None:return '--:--'
    seconds=max(0,int(seconds));h,seconds=divmod(seconds,3600);m,s=divmod(seconds,60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


class _Display:
    def __init__(self):
        self.lock=threading.RLock();self.stack=[];self.width=0;self.stop=None;self.thread=None

    def add(self,bar):
        with self.lock:
            self.stack.append(bar)
            if len(self.stack)==1:
                stop=threading.Event();self.stop=stop
                self.thread=threading.Thread(target=self._heartbeat,args=(stop,),daemon=True)
                self.thread.start()
            self.render()

    def _heartbeat(self,stop):
        while not stop.wait(.25):
            with self.lock:self.render()

    def render(self,finished=None):
        if not self.stack and finished is None:return
        bar=finished or self.stack[-1]
        parents=[p.description for p in self.stack[:-1] if p is not bar]
        label=(' / '.join(parents[-1:]+[bar.description])).replace('\n',' ').replace('\r',' ')
        elapsed=time.monotonic()-bar.started
        rate=bar.done/max(elapsed,1e-9)
        if bar.total is None:
            spinner='|/-\\'[int(elapsed*4)%4]
            body=f'{spinner} {bar.done} {bar.unit}' if not bar.closed else bar.state
            eta=None
        else:
            fraction=min(1.,bar.done/bar.total) if bar.total else 1.
            width=16;filled=int(fraction*width)
            body='['+'='*filled+('>' if filled<width else '')+'.'*max(0,width-filled-1)+']'
            body+=f' {fraction*100:5.1f}% {bar.done}/{bar.total} {bar.unit}'
            eta=(bar.total-bar.done)/rate if rate else None
        columns=max(80,shutil.get_terminal_size((140,24)).columns)
        suffix=f' | t {duration(elapsed)} ETA {duration(eta)}'
        if rate and bar.total is not None:suffix+=f' {rate:.1f} {bar.unit}/s'
        if bar.message:suffix+=' | '+bar.message
        prefix=label[:max(16,columns-len(body)-len(suffix)-3)]
        line=f'{prefix} {body}{suffix}'
        if bar.closed and bar.state!='done':line+=' ['+bar.state+']'
        try:
            bar.stream.write('\r'+line+' '*max(0,self.width-len(line)))
            bar.stream.flush();self.width=len(line)
        except (OSError,ValueError):pass

    def remove(self,bar):
        thread=None
        with self.lock:
            if bar not in self.stack:return
            if len(self.stack)==1:
                self.render(finished=bar)
                try:bar.stream.write('\n');bar.stream.flush()
                except (OSError,ValueError):pass
                self.stack.remove(bar);self.width=0;self.stop.set();thread=self.thread
            else:
                self.stack.remove(bar);self.render()
        if thread and thread is not threading.current_thread():thread.join(timeout=.5)


_DISPLAY=_Display()


class Progress:
    def __init__(self,total=None,description='Working',unit='items',enabled=None,stream=None):
        self.total=total;self.description=description;self.unit=unit;self.done=0
        self.enabled=(os.environ.get('EXPERIMENT_PROGRESS','1')!='0') if enabled is None else enabled
        self.stream=stream or sys.stderr;self.started=None;self.closed=False;self.message='';self.state='running'

    def __enter__(self):
        self.started=time.monotonic()
        if self.enabled:_DISPLAY.add(self)
        return self

    def update(self,advance=1,*,completed=None,message=None):
        with _DISPLAY.lock:
            self.done=self.done+advance if completed is None else completed
            if message is not None:self.message=str(message)

    def close(self,state='done'):
        if self.closed:return
        self.closed=True;self.state=state
        if self.enabled:_DISPLAY.remove(self)

    def __exit__(self,exc_type,exc,tb):
        self.close('interrupted' if exc_type in (KeyboardInterrupt,GeneratorExit) else 'failed' if exc_type else 'done')
        return False


def track(items,description='Working',total=None,unit='items',enabled=None):
    if total is None:
        try:total=len(items)
        except TypeError:pass
    with Progress(total,description,unit,enabled) as bar:
        for item in items:
            yield item
            bar.update()


def stage(description,enabled=None):
    """Spinner/elapsed time while loading, hashing, or other uncounted work runs."""
    return Progress(description=description,enabled=enabled)


def status(message):
    """Update metrics on the current line; detailed metrics still belong in JSONL."""
    with _DISPLAY.lock:
        if _DISPLAY.stack:_DISPLAY.stack[-1].message=str(message)
