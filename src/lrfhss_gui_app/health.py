"""Display-only health and packet reception estimates; never receiver controls."""
from dataclasses import dataclass
import math

# Advisory display thresholds, independent of decoder admission/settings.
# Upper warning boundaries are inclusive; low ADC warns below 1.5 codes and
# is red at/below 0.5. ADC values are the existing monitor's subsampled
# I/Q RMS codes and peak rail fraction. PRR is counter-based, not offered truth.
HEALTH_THRESHOLDS = {
    'gain_amber_db': 49.0,       # near the tuner limit; display configured dB
    'adc_low_amber_codes': 1.5, 'adc_low_red_codes': .5,
    'clip_amber_percent': .01, 'clip_red_percent': .1,
    'lag_amber_s': .5, 'lag_red_s': 2.0,
    'expired_amber_per_min': 60, 'expired_red_per_min': 600,
    'prr_amber_below': .95, 'prr_red_below': .80,
    'drops_red_above': 0,
}
SENSORS = (('linear_acceleration','A','Acceleration'), ('gyroscope','G','Gyroscope'),
           ('magnetometer','M','Magnetometer'), ('quaternion','Q','Quaternion'),
           ('gps','P','GPS'), ('state_of_charge','C','State of charge'),
           ('environmental','E','Environment'))

def validity_mask(decoded):
    flags=decoded.get('valid', {})
    return sum(1 << i for i,(key,_,_) in enumerate(SENSORS) if flags.get(key))

def stale(last_seen, now, seconds=60):
    return now-last_seen >= seconds

@dataclass(frozen=True)
class Reading:
    text: str
    level: str = 'normal'
    tip: str = ''

def number(value):
    return isinstance(value,(int,float)) and math.isfinite(value)

def high(value, amber, red):
    return 'red' if value>=red else 'amber' if value>=amber else 'normal'

class ReceptionEstimate:
    """Sum independent counter epochs. Duplicate counters add no reception.

    A forward modulo-65536 delta below 32768 is progress (including wrap);
    a larger delta is a reset. Each epoch begins with one received/expected
    packet, so packets before the first observation are deliberately unknown.
    """
    def __init__(self):
        self.nodes={};self.received=0;self.expected=0

    def add(self, csid, counter):
        counter=int(counter)&0xffff
        row=self.nodes.get(csid)
        if row is None:
            row=self.nodes[csid]=dict(last=counter,received=0,expected=0,resets=0)
            delta=1
        else:
            delta=(counter-row['last'])&0xffff
            if delta==0:return
            if delta>=32768:row['resets']+=1;delta=1
        row['last']=counter;row['received']+=1;row['expected']+=delta
        self.received+=1;self.expected+=delta

    @property
    def ratio(self):
        return self.received/self.expected if self.expected else None

def readings(stats, gain, prr):
    t=HEALTH_THRESHOLDS;source=stats.get('source') or {};rr=stats.get('rr2_health') or {}
    stages=stats.get('receiver_stages') or {};record=stats.get('recording') or {}
    out={'gain':Reading(f'{gain:.1f} dB', 'amber' if gain>=t['gain_amber_db'] else 'normal', 'Configured tuner gain; not a measured RF level') if number(gain) else Reading('n/a')}
    rms=source.get('level_rms_lsb');clip=source.get('clip_fraction')
    level='normal'
    if number(rms):level='red' if rms<=t['adc_low_red_codes'] else 'amber' if rms<t['adc_low_amber_codes'] else level
    if number(clip):
        c=high(100*clip,t['clip_amber_percent'],t['clip_red_percent'])
        level=max((level,c),key=('normal','amber','red').index)
    if source.get('stalled') or (not number(clip) and source.get('level_state')=='clipping'):level='red'
    if source.get('level_state')=='low' and level=='normal':level='amber'
    warning='No IQ; check dongle. ' if source.get('stalled') else 'Clipping; lower gain. ' if source.get('level_state')=='clipping' else 'Low level; check gain. ' if source.get('level_state')=='low' else ''
    out['adc']=Reading((f'{rms:.2f} codes' if number(rms) else 'n/a')+' · '+(f'{100*clip:.3f}%' if number(clip) else 'n/a'),level,warning+'RMS ADC codes · samples at either rail (existing monitor estimate); unavailable in file mode')
    lag=(stats.get('decoder_health') or {}).get('lag_s',rr.get('lag_s'))
    out['lag']=Reading(f'{lag:.2f} s',high(lag,t['lag_amber_s'],t['lag_red_s'])) if number(lag) else Reading('n/a',tip='No input clock observation is available yet')
    # Full stage snapshots omit untouched Counter keys (zero). Partial live
    # snapshots cannot distinguish zero from an unpublished counter.
    expired=stages.get('header_candidates_expired',0 if 'acquisition_candidates' in stages else None)
    uptime=stats.get('uptime_s')
    rate=60*expired/uptime if number(expired) and number(uptime) and uptime>0 else None
    out['expired']=Reading(f'{rate:.1f} /min',high(rate,t['expired_amber_per_min'],t['expired_red_per_min']),'Run-average rate from the current counter and elapsed run time') if rate is not None else Reading('n/a',tip='No expired-header counter is available yet')
    lost=rr.get('estimated_source_drop_samples')
    out['source']=Reading(f'≥ {lost:,} samples' if lost else '0 samples','red' if lost>t['drops_red_above'] else 'normal','Estimated upstream loss; true source drops unreported by rtl_tcp') if number(lost) else Reading('n/a',tip='True source drops unreported by rtl_tcp; no estimate in this mode')
    drops=record.get('bytes_dropped',0)
    text='off' if not record else 'stopped' if record.get('stopped_reason') or record.get('error') else 'recording'
    out['recording']=Reading(f'{text} · {drops:,} B dropped','red' if drops or record.get('error') else 'amber' if record.get('stopped_reason') else 'normal',str(record.get('error') or record.get('stopped_reason') or 'Recording byte drops'))
    ratio=prr.ratio
    out['prr']=Reading(f'{100*ratio:.1f}% ({prr.received}/{prr.expected})','red' if ratio<t['prr_red_below'] else 'amber' if ratio<t['prr_amber_below'] else 'normal','Estimated received / counter span, summed over ChipSats and reset epochs; duplicates ignored') if ratio is not None else Reading('n/a',tip='No trusted packet counters received')
    return out
