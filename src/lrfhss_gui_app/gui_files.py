"""GUI metadata and export tasks; call these through FileJobs, never while painting."""
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import tempfile
from concurrent.futures import CancelledError
from .journal import Journal,Lease,durable_json
from .session import FleetSession
from .config import FLEET_RTL_TCP

_SETTINGS_LOCK=threading.Lock()


def is_fleet_helper(path):
    return Path(path).resolve() == FLEET_RTL_TCP


def read_state(log_dir):
    settings_path = log_dir / 'gui_settings.json'
    repo = Path(__file__).resolve().parents[2]
    if not settings_path.exists() and log_dir.resolve() == repo / 'logs':
        settings_path = repo / 'lrfhss_gui_app' / 'logs' / 'gui_settings.json'
    try:
        saved=json.loads(settings_path.read_text(encoding='utf-8'))
        if not isinstance(saved,dict):saved={}
    except (OSError,ValueError,TypeError):saved={}
    return saved,list(Journal.unfinished(log_dir))


def save_settings(log_dir,settings):
    # Several modeless receiver windows may share one GUI settings file.
    with _SETTINGS_LOCK:
        log_dir.mkdir(parents=True,exist_ok=True)
        durable_json(log_dir/'gui_settings.json',settings)


def prepare_session(log_dir,config,public,run_name):
    if config.mode=='file' and not os.path.isfile(config.file_path):
        raise ValueError('Choose a valid IQ file first.')
    journal=Journal.create(log_dir,public,announce=False)
    summary=dict(started_at=datetime.now().astimezone().isoformat(),configuration=asdict(config))
    durable_json(journal.directory/'config.json',summary)
    # Keep both existing public formats byte-for-byte in structure. The run name
    # is GUI-only metadata beside the journal, not a decoder configuration field.
    durable_json(journal.packets.with_suffix('.gui.json'),dict(schema_version=1,run_name=run_name))
    durable_json(journal.directory/'session.json',{'packets':journal.packets.name,'public':public})
    return journal,summary


def finish_summary_if_missing(path,summary):
    if not path.exists():
        durable_json(path,dict(summary,finished_at=datetime.now().astimezone().isoformat()),indent=2)


def recovery_metadata(journal):
    session=json.loads((journal.directory/'session.json').read_text(encoding='utf-8'))
    path=journal.directory/'config.json'
    summary=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return session['public'],summary


def writer_finished(journal):
    try:
        lease=Lease(journal.directory/'writer.lock')
    except (PermissionError,BlockingIOError):
        return False
    lease.close()
    return True


def export_csv(target,paths,fallback,cancel=None):
    def check_cancelled():
        if cancel is not None and cancel.is_set():raise CancelledError()
    def records():
        if not paths:yield from fallback
        for path in paths:
            with Path(path).open('rb') as stream:
                limit=stream.seek(0,os.SEEK_END);stream.seek(0)
                while stream.tell()<limit:
                    check_cancelled()
                    line=stream.readline(limit-stream.tell())
                    if not line.endswith(b'\n'):break
                    yield json.loads(line)
    rows=FleetSession.csv_stream(records());first=next(rows,None)
    if first is None:return 0
    target=Path(target)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',newline='',encoding='utf-8',
                 prefix=target.name+'.',suffix='.partial',dir=target.parent,delete=False) as f:
            temporary=Path(f.name)
            writer=csv.DictWriter(f,fieldnames=list(first));writer.writeheader();writer.writerow(first);count=1
            for row in rows:
                check_cancelled();writer.writerow(row);count+=1
            f.flush();os.fsync(f.fileno())
        check_cancelled()
        os.replace(temporary,target);temporary=None
        return count
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
