"""Opt-in Batch 9.5b diagnostics. No imports/hooks in ordinary receiver use.

All clocks are perf_counter seconds from the machine's monotonic clock. Worker
CPU seconds add across processes; nested owner wall times must not be added.
Immutable snapshot bytes retain the baseline ownership model (no shared memory).
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace, astuple
from pathlib import Path
import csv
import functools
import json
import os
import queue
import sys
import threading
import time


def memory_bytes(pids):
    """Summed RSS/working set for explicit receiver PIDs; None if unavailable."""
    total = 0
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True); psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
        for pid in pids:
            handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
            if not handle: return None
            try:
                value=PMC();value.cb=ctypes.sizeof(value)
                if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb):return None
                total+=value.WorkingSetSize
            finally:kernel.CloseHandle(handle)
        return total
    try:
        page=os.sysconf('SC_PAGE_SIZE')
        for pid in pids:total+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*page
        return total
    except (OSError,ValueError,IndexError):
        try:
            import psutil
            return sum(psutil.Process(p).memory_info().rss for p in pids)
        except Exception:return None


class Trace:
    def __init__(self):
        self.event_counts=Counter();self.jobs={};self.lock=threading.Lock();self.events=[];self.blocks=[]
        # Batch 9.10 Change 13: how long the owner/transport threads waited for
        # this lock (in memory only; never touches the evidence store).
        self.lock_wait=dict(count=0,max_s=0.,over_50ms=0,total_s=0.)
        self.stages=defaultdict(lambda:Counter(calls=0,wall_s=0.,thread_cpu_s=0.,exclusive_wall_s=0.,exclusive_thread_cpu_s=0.))
        self.stack=[];self.current_stage=None;self.restore=[];self.descriptors=[];self.active_block=None
        # Batch 10.4g: the class each outstanding job got at submission. With the asynchronous
        # job store, setdefault() never reads the stored row, so a guessed default class in a
        # transport update would overwrite it (SIC jobs were stored as 'payload').
        self._job_classes={}

    @contextmanager
    def _locked(self):
        start=time.perf_counter()
        with self.lock:
            waited=time.perf_counter()-start;w=self.lock_wait
            w['count']+=1;w['total_s']+=waited;w['over_50ms']+=int(waited>.05)
            if waited>w['max_s']:w['max_s']=waited
            yield

    def transport(self, kind, jid, fields):
        if jid.startswith('warmup:'):return
        with self._locked():
            cls=self._job_classes.get(jid) or ('header' if ':h' in jid else 'payload')
            row=self.jobs.setdefault(jid,dict(job_id=jid,job_class=cls))
            if kind=='job_submit':row.update(create_submit_wall=fields['wall'],iq_bytes=fields['iq_bytes'])
            elif kind=='job_dispatch':row.update(dispatch_wall=fields['wall'],worker_pid=fields['worker_pid'])
            elif kind=='transport_sent':row.update(fields)
            elif kind=='transport_received':
                if fields['kind']=='started':row['worker_entry_wall']=fields['worker_wall']
                else:
                    row.update(fields);row['terminal']=fields['kind']
                    if cls in ('sic_rebuild','sic_tile'):
                        self._job_classes.pop(jid,None)   # no incorporation event follows for these
            elif kind=='result_incorporation' and fields['kind']!='started':
                row['incorporated_wall']=fields['wall']
                self._job_classes.pop(jid,None)

    def event(self, event):
        # Compact per-candidate/job/track events; no fragment soft arrays or per-sample output.
        exclude={'payload_result','header_feed_summary','header_validation_deferred'}
        if event['event'] not in exclude:
            self.events.append(event.copy());self.event_counts[event['event']]+=1
        if event.get('job_id') and event['event'] in ('header_candidate_cancelled','track_cancelled','header_worker_failed','payload_worker_failed'):
            with self._locked():
                row=self.jobs.setdefault(event['job_id'],dict(job_id=event['job_id']))
                row.update(termination_event=event['event'],termination_reason=event.get('reason',event.get('error')),
                    termination_wall=event.get('owner_wall'),termination_rf_sample=event.get('sample'))
        if event['event']=='header_candidate_enqueued':
            with self._locked():
                row=self.jobs.setdefault(event['job_id'],dict(job_id=event['job_id'],job_class='header'))
                row.update({k:event[k] for k in ('snapshot_start_wall','snapshot_complete_wall','snapshot_copy_complete_wall','candidate_id','physical_header','candidate_sample','inferred_packet_start') if k in event})

    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            wall=time.perf_counter();cpu=time.thread_time();frame=[0.,0.,label,wall]
            self.stack.append(frame);self.current_stage=label
            if self.active_block is not None and label in ('continuous_legacy','continuous_frontend'):
                self.active_block[label+'_start_wall']=wall
            try:return original(*args,**kwargs)
            finally:
                dw=time.perf_counter()-wall;dc=time.thread_time()-cpu
                if self.active_block is not None and label in ('continuous_legacy','continuous_frontend'):
                    self.active_block[label+'_end_wall']=wall+dw
                self.stack.pop();r=self.stages[label];r.update(calls=1,wall_s=dw,thread_cpu_s=dc,exclusive_wall_s=dw-frame[0],exclusive_thread_cpu_s=dc-frame[1])
                r['max_wall_s']=max(r['max_wall_s'],dw)
                if self.stack:self.stack[-1][0]+=dw;self.stack[-1][1]+=dc
                self.current_stage=self.stack[-1][2] if self.stack else None
        setattr(obj,name,measured);self.restore.append((obj,name,original))

    def attach(self,d,mode='full',record_descriptors=False):
        self.decoder=d
        pools=[d.payload_pool]+([d.header_pipeline.pool.pool] if d.header_pipeline else [])
        if getattr(d,'sic',None) is not None:
            pools.append(d.sic.pool)   # Batch 10.4d: SIC rebuild/tile/payload jobs get the same job evidence
        for pool in pools:
            pool.observer=self.transport
            def transform(job,pool=pool):
                frames=[f for f in self.stack if f[2] in ('header_construction','payload_construction')]
                # SIC jobs are classed by kind (sic_rebuild, sic_tile, sic_payload), never as
                # primary header/payload jobs, so primary job statistics stay primary-only.
                job_class=f"sic_{job.kind}" if pool.thread_prefix=='sic' else pool.thread_prefix
                with self._locked():
                    self._job_classes[job.job_id]=job_class
                    while len(self._job_classes)>100_000:      # bounded: jobs that never report back
                        self._job_classes.pop(next(iter(self._job_classes)))
                    row=self.jobs.setdefault(job.job_id,dict(job_id=job.job_id,job_class=job_class))
                    row['job_class']=job_class
                    row.update(creation_wall=frames[-1][3] if frames else time.perf_counter(),snapshot_ready_wall=time.perf_counter())
                if record_descriptors:
                    if pool.thread_prefix=='sic':
                        # Batch 10.4e: SIC params hold PhdrModel / PayloadJob objects; serialize explicitly
                        from lrfhss_sic_jobs import describe_job
                        fields=describe_job(job)
                    else:
                        fields={k:getattr(job,k) for k in job.__dataclass_fields__ if k not in ('iq','job_id','diagnostic_noop','frequency_selection','previous_validation')}
                    # Enough to reconstruct the exact real IQ snapshot from the SAME source.
                    self.descriptors.append(dict(kind=job_class,job_id=job.job_id,
                        submit_rf_sample=d.abs_sample,submit_wall=time.perf_counter(),iq_bytes=len(job.iq),fields=fields))
                noop=mode=='header-noop' and pool.thread_prefix=='header' or mode=='payload-noop' and pool.thread_prefix=='payload'
                return replace(job,diagnostic_noop=True) if noop else job
            pool.job_transform=transform
        for name,label in [('feed_raw','owner_feed'),('_feed_block','block_conversion_buffer'),('_trigger','continuous_legacy'),('_service','owner_service'),('poll','result_collection'),('_accept_physical','track_association'),('_submit_payload','payload_construction')]:self.wrap(d,name,label)
        for name,label in [('feed','continuous_frontend'),('_search','candidate_generation'),('_offer','candidate_classification'),('ready','candidate_scheduling')]:self.wrap(d.acquisition,name,label)
        self.wrap(d.ring,'get','snapshot_extraction')
        self.wrap(d.ring,'snapshot_bytes','direct_snapshot_extraction')
        for name in ('offer_candidate','observe_header','retire'):self.wrap(d.track_manager,name,'track_bookkeeping')
        if d.header_pipeline:
            for name,label in [('_submit','header_construction'),('collect','header_collection'),('service','header_scheduling')]:self.wrap(d.header_pipeline,name,label)
        if mode=='input-only':
            import lrfhss_live as phy
            def input_block(raw):
                z=phy.u8_iq_to_complex(raw);start=d.abs_sample;d.abs_sample+=len(z)
                d.ring.add(start,z)
                return []
            d._feed_block=input_block
        elif mode=='frontend-only':
            # Keep mixers, filters, history, decimation and noise estimator.
            # Skip spectral/correlation candidate generation and all decode jobs.
            d.acquisition._search=lambda *a,**k:None
            d.acquisition._offer=lambda *a,**k:None
            d.track_manager.offer_candidate=lambda *a,**k:None

    def state(self,source=None):
        d=self.decoder;now=time.perf_counter();hp=d.header_pipeline.pool if d.header_pipeline else None
        pools={'payload':d.payload_pool}
        if hp:pools['header']=hp.pool
        workers={};pids=[os.getpid()]
        for kind,pool in pools.items():
            workers[kind]=[]
            for s in list(pool.slots):
                try:
                    p=s['process'];pid=p.pid
                    if pid is not None:pids.append(pid)
                    workers[kind].append(dict(pid=pid,alive=p.is_alive(),exitcode=p.exitcode,job_id=s['job'],iq_bytes=s['iq_bytes'],sender_alive=s['sender'].is_alive(),reader_alive=s['reader'].is_alive()))
                except ValueError:pass
        acquisition=getattr(d,'acquisition_process',None)
        if acquisition is not None and acquisition.ready and not acquisition.closed:
            if 'pids' in acquisition.ready:
                pids.extend(acquisition.ready['pids'])
            else:
                pids.append(acquisition.ready['pid'])
        with self._locked():
            pending=[r for r in (self.jobs.pending_values() if hasattr(self.jobs,'pending_values') else self.jobs.values()) if 'incorporated_wall' not in r and 'create_submit_wall' in r]
            oldest=max((now-r['create_submit_wall'] for r in pending),default=0.)
        return dict(acquisition_process=acquisition.stats() if acquisition else None,owner_pid=os.getpid(),owner_stage=self.current_stage,processed_rf_s=d.abs_sample/1e6,
            header_waiting=len(hp.pool.waiting) if hp else 0,header_waiting_capacity=hp.capacity if hp else 0,header_outstanding=len(hp.order) if hp else 0,header_max_outstanding=hp.peak_outstanding if hp else 0,
            header_reorder=len(hp.completed) if hp else 0,payload_waiting=len(d.payload_pool.waiting),
            payload_outstanding=len(d._payload_pending),payload_waiting_capacity=d.payload_pool.capacity,payload_waiting_high_watermark=d.payload_pool.high_watermark,candidate_metadata=len(d.acquisition.pending),
            refinement_metadata=len(d._refinement_pending),oldest_outstanding_job_age_s=oldest,
            raw_blocks=source.queue.qsize() if source else 0,raw_high_watermark=source.peak if source else 0,
            worker_states=workers,rss_bytes=memory_bytes(pids),
            ring_start_sample=d.ring.chunks[0][0] if d.ring.chunks else None,ring_end_sample=d.ring.end,
            ring_iq_bytes=sum(x.nbytes for _,x in list(d.ring.chunks)),buffered_raw_bytes=len(d._raw_pending),
            receiver_finished=d._finished,still_processing_older_data=bool(self.current_stage or pending))

    def save(self,path):
        from tools.batch08_5_bench import save_csv
        rows=list(self.jobs.values())
        for r in rows:
            for name,a,b in [('owner_queue_s','create_submit_wall','dispatch_wall'),('submit_to_worker_s','create_submit_wall','worker_entry_wall'),('result_transport_s','worker_wall','result_received_wall'),('ordered_commit_wait_s','result_received_wall','incorporated_wall'),('serialization_s','serialize_start_wall','serialize_end_wall'),('pipe_send_s','serialize_end_wall','send_end_wall')]:
                if a in r and b in r:r[name]=r[b]-r[a]
        save_csv(path.with_suffix('.jobs.csv'),rows)
        save_csv(path.with_suffix('.blocks.csv'),self.blocks)
        lifecycle={'header_candidate_late','header_candidate_enqueue_rejected','header_worker_failed',
            'header_candidate_cancelled','header_refinement_skipped','header_refinement_cancelled',
            'header_refinement_rejected','track_expired','track_cancelled','payload_worker_failed'}
        save_csv(path.with_suffix('.lifecycle.csv'),[e for e in self.events if e['event'] in lifecycle])
        for suffix in ('.jobs.csv','.blocks.csv'):
            if not path.with_suffix(suffix).exists():path.with_suffix(suffix).write_text('job_id\n' if 'jobs' in suffix else 'block_id\n')
        return rows


class OwnerStallMonitor:
    """Batch 9.10 Change 13: capture WHERE the owner thread is when one unit of
    owner work (a block feed or an idle-service call) runs longer than the
    threshold. A daemon thread samples the owner's stack once per episode via
    sys._current_frames(); everything stays in bounded memory (no disk I/O).
    """
    def __init__(self, threshold_s=.3, interval_s=.05, max_episodes=20, trace=None):
        self.threshold_s=threshold_s;self.interval_s=interval_s;self.max_episodes=max_episodes;self.trace=trace
        self.owner=None;self.busy_since=None;self.label=None;self.captured=None
        self.episodes=[];self.count=0;self.max_s=0.;self.started=None
        self.stop=threading.Event();self.thread=None

    def start(self, owner_ident=None):
        self.owner=owner_ident or threading.get_ident();self.started=time.perf_counter()
        self.thread=threading.Thread(target=self._run,name='lrfhss-owner-stall-monitor',daemon=True);self.thread.start()
        return self

    def busy(self, label):
        self.label=label;self.captured=None;self.busy_since=time.perf_counter()

    def idle(self, unfinished=False):
        since=self.busy_since;self.busy_since=None
        if since is None:return
        duration=time.perf_counter()-since
        if duration>self.max_s:self.max_s=duration
        if duration>=self.threshold_s:
            self.count+=1
            episode=self.captured or dict(label=self.label,start_s=since-self.started,stack=None,stage=None)
            episode['duration_s']=duration
            if unfinished:episode['unfinished']=True   # the unit of work raised or was cut short
            if len(self.episodes)<self.max_episodes:self.episodes.append(episode)
            else:
                shortest=min(range(len(self.episodes)),key=lambda i:self.episodes[i]['duration_s'])
                if self.episodes[shortest]['duration_s']<duration:self.episodes[shortest]=episode
        self.captured=None

    def _run(self):
        while not self.stop.wait(self.interval_s):
            since=self.busy_since
            if since is None or self.captured is not None:continue
            if time.perf_counter()-since<self.threshold_s:continue
            frame=sys._current_frames().get(self.owner)
            stack=[]
            while frame is not None and len(stack)<30:
                code=frame.f_code;stack.append(f"{os.path.basename(code.co_filename)}:{frame.f_lineno} {code.co_name}")
                frame=frame.f_back
            stage=getattr(self.trace,'current_stage',None) if self.trace is not None else None
            if self.busy_since==since:
                self.captured=dict(label=self.label,start_s=since-self.started,captured_after_s=time.perf_counter()-since,
                                   stage=stage,stack=stack)

    def close(self):
        """Stop sampling. A unit of work still marked busy (it raised, or the run
        ended inside it) is recorded as an unfinished episode, never lost."""
        self.stop.set()
        if self.thread is not None:self.thread.join(1.)
        self.idle(unfinished=True)

    def summary(self):
        return dict(threshold_s=self.threshold_s,count=self.count,max_s=self.max_s,
                    episodes=sorted(self.episodes,key=lambda e:-e['duration_s']))


class FailureQueue(queue.Queue):
    """Capture the rejected offer and oldest input atomically with queue capacity."""
    def __init__(self,capacity):
        super().__init__(capacity);self.last_full=None;self.peak=0
    def put_nowait(self,item):
        with self.not_full:
            depth=self._qsize();now=time.perf_counter()
            if self.maxsize>0 and depth>=self.maxsize:
                oldest=self.queue[0][0]
                self.last_full=dict(observed_wall=now,depth=depth,
                    oldest_queued_input_age_s=now-oldest['admission_wall'],
                    oldest_block_id=oldest['block_id'])
                raise queue.Full
            item[0]['admission_wall']=now
            item[0]['queue_at_admission']=depth
            self._put(item);self.unfinished_tasks+=1
            self.peak=max(self.peak,self._qsize());self.not_empty.notify()


class PacedSource:
    """Absolute deadlines, bounded queue, one terminal gap with explicit fallout.

    Stop-at-first-gap is optional for quick isolation; normal runs still offer
    the complete source. Sample time is never compressed after a gap.
    """
    def __init__(self,path,capacity=20,fs=1_000_000,block_bytes=100000,state=None,stop_at_gap=False,records=None):
        if capacity<1 or fs<=0 or block_bytes<2 or block_bytes%2:raise ValueError('Invalid source bounds')
        self.path=Path(path);self.capacity=capacity;self.fs=fs;self.block_bytes=block_bytes
        self.queue=FailureQueue(capacity);self.stop=threading.Event();self.done=threading.Event()
        self.started=self.finished=None;self.first_failure=None;self.error=None;self.peak=0
        self.offered_bytes=self.accepted_bytes=self.dropped_bytes=0
        self.offered_blocks=self.accepted_blocks=self.dropped_blocks=self.after_gap_blocks=0
        self.records=[] if records is None else records;self.get_state=state;self.stop_at_gap=stop_at_gap
    def start(self):
        self.started=time.perf_counter()
        def produce():
            end=0
            try:
                with self.path.open('rb') as f:
                    while True:
                        created=time.perf_counter();raw=f.read(self.block_bytes)
                        if not raw:break
                        if len(raw)%2:raise ValueError('Incomplete IQ pair')
                        begin=end;end+=len(raw)//2;deadline=self.started+end/self.fs
                        if self.stop.wait(max(0,deadline-time.perf_counter())):break
                        offered=time.perf_counter();i=self.offered_blocks
                        row=dict(block_id=i,begin_sample=begin,end_sample=end,read_wall=created,offered_wall=offered,deadline_wall=deadline,deadline_lag_s=offered-deadline,admission_wall=None,queue_at_admission=self.queue.qsize(),status='offered')
                        self.offered_blocks+=1;self.offered_bytes+=len(raw)
                        if self.first_failure:
                            row['status']='discarded_after_terminal_gap';self.dropped_blocks+=1;self.dropped_bytes+=len(raw);self.after_gap_blocks+=1
                        else:
                            row['admission_wall']=time.perf_counter()
                            try:
                                self.queue.put_nowait((row,raw));row['status']='accepted'
                                self.accepted_blocks+=1;self.accepted_bytes+=len(raw);self.peak=max(self.peak,self.queue.peak)
                            except queue.Full:
                                row['admission_wall']=None;row['status']='first_queue_overflow';self.dropped_blocks+=1;self.dropped_bytes+=len(raw)
                                observed=self.queue.last_full
                                state_start=time.perf_counter()
                                state=self.get_state() if self.get_state else {}
                                state_end=time.perf_counter()
                                oldest_age=observed['oldest_queued_input_age_s']
                                self.first_failure=dict(wall_s=observed['observed_wall']-self.started,represented_rf_s=end/self.fs,
                                    block_id=i,reason='raw_queue_full_contiguous_input_gap',queue_before=self.capacity,
                                    queue_after=observed['depth'],max_queue=self.queue.peak,receiver_state_at_failure=state,
                                    state_observation_wall_range=[state_start,state_end],
                                    state_sampling_note='Queue rejection is atomic; process/buffer fields are live observations over this short interval.',
                                    previous_completed_block=getattr(self,'last_completed',None),
                                    oldest_queued_input_age_s=oldest_age,input_block_age_s=offered-created,receiver_still_processing_older_data=not state.get('receiver_finished',False))
                        self.records.append(row)
                        if self.first_failure and self.stop_at_gap:break
            except BaseException as e:self.error=repr(e)
            finally:self.finished=time.perf_counter();self.done.set()
        self.thread=threading.Thread(target=produce,name='batch095b-paced-source',daemon=True);self.thread.start();return self
    def close(self):
        self.stop.set()
        if self.started is not None:
            self.thread.join(2.)
            if self.thread.is_alive():raise RuntimeError('Source thread did not stop')


def warmup_jobs():
    """Representative real waveform, identical DSP/FEC settings to selected profile."""
    from tests.test_fragments import packet_window
    import lrfhss_live as phy
    from lrfhss_acquisition import HeaderCandidate
    from lrfhss_header_jobs import HeaderJob
    from lrfhss_payload_jobs import PayloadJob
    z,h,_=packet_window('A45');base=h.start-35000;stop=h.start+290000
    c=HeaderCandidate(1,h.start,0,phy.TRIGGER_HEADER_HZ,h.freq_hz,'sync',1.,10.,0)
    header=HeaderJob('warmup:header',z[base:stop].tobytes(),z.dtype.str,base,'physical',astuple(c),batch_header=True,grid_cache=True)
    # Use the production payload extraction footprint (starts after all headers).
    pbase=max(0,h.start+3*233472-2500);end=h.start+phy.packet_duration_samples(h.info['payload_len'])
    payload=PayloadJob('warmup:payload','warmup',1,h.start,pbase,h.start-pbase,h.freq_hz,h.cost,
        tuple(sorted(h.info.items())),z[pbase:end].tobytes(),z.dtype.str,(),(),0.,fec_backend='numpy')
    return header,payload


def warm_decoder(d,timeout=60.,stop_event=None):
    header,payload=warmup_jobs();result={}
    if d.header_pipeline:
        header=replace(header,batch_header=d.batch_header,grid_cache=d.header_grids is not None,
                       frequency_reuse=getattr(d,"phdr_frequency_reuse",False))
        result['header']=d.header_pipeline.pool.pool.warmup(header,timeout,stop_event)
    payload=replace(payload,fec_backend=d.payload_fec_backend,native_library=d.payload_native_library)
    result['payload']=d.payload_pool.warmup(payload,timeout,stop_event)
    if getattr(d,'sic',None) is not None:
        result['sic']=d.sic.warmup(timeout,stop_event)
    return result
