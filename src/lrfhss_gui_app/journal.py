"""Single-writer packet publication. A sequence number is the JSONL byte offset."""
from __future__ import annotations

import json
import os
import queue
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class Lease:
    """An OS-owned byte lock disappears even if its process exits abruptly."""
    def __init__(self,path: Path):
        self.file=path.open("a+b")
        if not path.stat().st_size:
            self.file.write(b"\0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise

    def close(self):
        self.file.close()


def durable_json(path, value, *, indent=None):
    temp=path.with_suffix(".writing")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value,stream,indent=indent)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@dataclass(frozen=True)
class Journal:
    directory: Path
    packets: Path

    def finish_summary(self, last_stats):
        info=json.loads((self.directory/"session.json").read_text(encoding="utf-8"))
        if not info["public"]:
            return
        config=self.directory/"config.json"
        summary=json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        if last_stats is not None:
            summary["last_stats"]=last_stats
        summary["finished_at"]=datetime.now().astimezone().isoformat()
        durable_json(self.packets.with_suffix(".receiver.json"),summary,indent=2)

    @classmethod
    def create(cls, log_dir: Path, public: bool, *, announce=True):
        uid=uuid.uuid4().hex
        folder=log_dir/"delivery"/uid
        folder.mkdir(parents=True)
        # A session can stop and restart within the same second.
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        packets=log_dir/f"lrfhss_{stamp}.jsonl" if public else folder/"packets.jsonl"
        try:
            packets.touch(exist_ok=False)
        except FileExistsError:
            if not public:
                raise
            packets=log_dir/f"lrfhss_{stamp}_{uid}.jsonl"
            packets.touch(exist_ok=False)
        (folder/"events.jsonl").touch(exist_ok=False)
        if announce:
            durable_json(folder/"session.json", {"packets":packets.name,"public":public})
        return cls(folder,packets)

    @classmethod
    def unfinished(cls, log_dir):
        for meta in sorted((Path(log_dir)/"delivery").glob("*/session.json"),key=lambda p:p.stat().st_mtime_ns):
            if not (meta.parent/"displayed.json").exists():
                info=json.loads(meta.read_text(encoding="utf-8"))
                directory=Path(log_dir) if info["public"] else meta.parent
                yield cls(meta.parent,directory/Path(info["packets"]).name)


class Writer:
    def __init__(self,journal,hints):
        self.journal=journal
        self.hints=hints
        self.lock=Lease(journal.directory/"writer.lock")
        try:
            if any((journal.directory/name).exists() for name in ("finished.json","displayed.json")):
                raise OSError("This receiver journal is already complete")
            self.packets=journal.packets.open("ab",buffering=0)
            self.events=(journal.directory/"events.jsonl").open("ab",buffering=0)
        except BaseException:
            if hasattr(self,"packets"):
                self.packets.close()
            self.lock.close()
            raise

    def _append(self,stream,kind,value,sync=False):
        newline=os.linesep if kind=="packet" else "\n"
        raw=(json.dumps(value,separators=(",",":"))+newline).encode("utf-8")
        if stream.write(raw)!=len(raw):
            raise OSError("Short write to receiver publication journal")
        if sync:
            os.fsync(stream.fileno())
        try:
            self.hints.put_nowait((kind,stream.tell()))
        except queue.Full:
            pass  # The timer also tails the journal; a wake-up owns no data.

    def packet(self,envelope):
        self._append(self.packets,"packet",envelope,True)

    def event(self,kind,value=None):
        self._append(self.events,kind,[kind,value],kind!="status")

    def close(self):
        try:
            self.packets.close()
        finally:
            try:
                self.events.close()
            finally:
                self.lock.close()


class Tail:
    """Read complete lines only. Commit the cursor after the consumer accepts them."""
    def __init__(self,path):
        self.path=path
        self.stream=path.open("rb")
        self.offset=0

    def peek(self,limit):
        self.stream.seek(self.offset)
        records=[]
        end=self.offset
        for _ in range(limit):
            line=self.stream.readline()
            if not line.endswith(b"\n"):
                break
            records.append(json.loads(line))
            end=self.stream.tell()
        return records,end

    def remaining(self):
        return os.fstat(self.stream.fileno()).st_size-self.offset

    def close(self):
        self.stream.close()


class Delivery:
    def __init__(self,journal):
        self.journal=journal
        self.lease=Lease(journal.directory/"display.lock")
        try:
            self.packets=Tail(journal.packets)
            self.events=Tail(journal.directory/"events.jsonl")
        except BaseException:
            if hasattr(self,"packets"):
                self.packets.close()
            self.lease.close()
            raise
        self.total=0

    def drain(self,accept,on_event,limit=128):
        batch,end=self.packets.peek(limit)
        if batch:
            accept(batch)
            self.packets.offset=end
            self.total+=len(batch)
        events,end=self.events.peek(limit)
        latest=None
        for kind,value in events:
            if kind=="status" and value.get("status")=="RUNNING":
                if latest and "receiver_stages" in latest and "receiver_stages" not in value:
                    value={**value,"receiver_stages":latest["receiver_stages"]}
                latest=value
            else:
                if latest is not None:
                    on_event("status",latest)
                    latest=None
                on_event(kind,value)
        if latest is not None:
            on_event("status",latest)
        self.events.offset=end
        return len(batch)==limit or len(events)==limit

    def finish(self):
        lost=[str(t.path) for t in (self.packets,self.events) if t.remaining()]
        if lost:
            raise OSError("Incomplete trailing receiver journal record in "+", ".join(lost))
        terminal=self.journal.directory/"finished.json"
        if terminal.exists():
            errors=json.loads(terminal.read_text(encoding="utf-8")).get("publication_errors")
            if errors:
                raise OSError("Receiver could not commit a decoded packet: "+errors[0])
        durable_json(self.journal.directory/"displayed.json",{"packets":self.total,"end_offset":self.packets.offset})
        self.close()

    def close(self):
        self.packets.close()
        self.events.close()
        self.lease.close()

