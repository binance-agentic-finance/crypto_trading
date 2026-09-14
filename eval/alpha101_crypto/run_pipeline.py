"""Frozen daily Alpha101 experiment: current-cohort and historical top-ten."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from factors import compute_factors
import sys as _sys
_EVAL_ROOT = str(Path(__file__).resolve().parents[1])  # eval/ holds the shared measurement core
if _EVAL_ROOT not in _sys.path: _sys.path.insert(0, _EVAL_ROOT)
from factor_eval.engine import evaluate_factor
HERE=Path(__file__).resolve().parent; DATA=HERE/'data'; OUT=HERE/'results'
SPLITS={'dev':('2022-04-01','2024-06-30'),'val':('2024-07-01','2025-09-30'),'oot':('2025-10-01','2026-09-09')}
FIELDS=['open','high','low','close','volume','quote_volume']

def lifecycle_masks(index,columns,registry):
 # Fully elapsed bars may be inputs. Prices at a midnight before the exact
 # delivery time remain valid exit prices; later placeholder bars do not.
 info={x['symbol']:x for x in registry['symbols']}
 live_bar=pd.DataFrame(True,index=index,columns=columns)
 live_open=live_bar.copy()
 for s in columns:
  item=info[s];on=pd.to_datetime(item['onboardDate'],unit='ms',utc=True)
  off=pd.to_datetime(item['deliveryDate'],unit='ms',utc=True)
  live_bar[s]=(index>=on)&(index+pd.Timedelta(days=1)<=off)
  live_open[s]=(index>=on)&(index<off)
 return live_bar,live_open

def dump(path,obj):
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))

def load_inputs(pool):
 cfg=json.loads((DATA/'acquisition_config.json').read_text())
 end=pd.Timestamp(cfg['end_exclusive']); start=pd.Timestamp(cfg['start'])
 idx=pd.date_range(start,end-pd.Timedelta(days=1),freq='D')
 allowed={r['symbol'] for r in json.loads((DATA/'daily_manifest.json').read_text()) if r['status'] in ('ok','cached')}
 registry=json.loads((DATA/'exchange_info.json').read_text())
 info=json.loads((DATA/'universes.json').read_text())
 if pool=='current':
  cols=info['current_top10'];membership=pd.DataFrame(True,index=idx,columns=cols)
 else:
  q=pd.DataFrame({s:pd.read_parquet(DATA/'daily'/f'{s}.parquet',columns=['ts','quote_volume']).set_index('ts').quote_volume for s in sorted(allowed)}).reindex(idx)
  live_bar,_=lifecycle_masks(idx,q.columns,registry);q=q.where(live_bar)
  eligible=(q.notna().cumsum()>=60)&q.notna();adv=q.rolling(30,min_periods=30).mean()
  rows=[]
  rebs=pd.date_range(pd.Timestamp('2021-04-01',tz='UTC'),end,freq='MS')
  for t in rebs:
   prior=t-pd.Timedelta(days=1)
   snap=adv.loc[prior].where(eligible.loc[prior]).dropna().sort_values(ascending=False).head(10)
   for rank,(s,v) in enumerate(snap.items(),1):rows.append({'rebalance':t,'rank':rank,'symbol':s,'adv30_usdt':float(v)})
  hist=pd.DataFrame(rows);hist.to_csv(OUT/'membership_with_warmup.csv',index=False)
  cols=sorted(hist.symbol.unique());membership=pd.DataFrame(False,index=idx,columns=cols)
  for k,t in enumerate(rebs):
   stop=rebs[k+1] if k+1<len(rebs) else end
   names=hist.loc[hist.rebalance==t,'symbol'].tolist()
   membership.loc[(idx>=t)&(idx<stop),names]=True
 frames={s:pd.read_parquet(DATA/'daily'/f'{s}.parquet').set_index('ts') for s in cols}
 raw={f:pd.DataFrame({s:d[f] for s,d in frames.items()}).reindex(idx) for f in FIELDS}
 live_bar,live_open=lifecycle_masks(idx,cols,registry)
 raw={f:x.where(live_open if f=='open' else live_bar) for f,x in raw.items()}
 available=raw['close'].notna()&raw['quote_volume'].notna()
 eligible=membership&(available.cumsum()>=60)&available&live_bar
 return raw,eligible,{'pool':pool,'columns':cols,'current_top10':info['current_top10'],'start':str(start),'end_exclusive':str(end),'ranking_scope':'current ten with >=60 observed bars' if pool=='current' else 'only contemporaneous monthly top ten; warmup memberships from 2021-04','selection_caveat':'current registry incomplete for fully vanished historical symbols; current cohort is retrospective selection' if pool=='current' else 'current registry includes returned settling contracts but is not a complete historical securities master'}

def funding_panel(index,columns):
 out=pd.DataFrame(np.nan,index=index,columns=columns); detail=[]
 for s in columns:
  path=DATA/'funding'/f'{s}.parquet';stamp=path.with_suffix('.json')
  if not path.exists() or not stamp.exists():detail.append({'symbol':s,'status':'missing_file'});continue
  meta=json.loads(stamp.read_text());df=pd.read_parquet(path)
  if meta.get('status') not in ('ok','partial','missing'):
   detail.append(meta);continue
  if meta.get('pagination_complete') is not True:
   detail.append({**meta,'evaluation_status':'unverified_pagination'});continue
  times=pd.to_datetime(df['fundingTime'],unit='ms',utc=True)
  df=df.assign(day=times.dt.floor('D'),cash=pd.to_numeric(df.fundingRate)*pd.to_numeric(df.markPrice))
  # The completed pagination queried the whole requested date interval. Days
  # without settlement events are zero; days with an unknown event price are NA.
  grouped=df.groupby('day').cash.sum(min_count=1)
  row_valid=df.get('funding_input_valid',pd.Series(True,index=df.index)).fillna(False).astype(bool)
  unknown=set(df.loc[df.cash.isna()|~row_valid,'day'])
  # Acquisition also flags interval-transition ambiguity and missing events
  # that a max(previous,current interval) rule cannot identify reliably.
  gap_days=meta.get('event_gap_diagnostics',{}).get('gap_affected_utc_days',[])
  unknown.update(pd.to_datetime(gap_days,utc=True))
  ordered=df.assign(event_time=times).sort_values('event_time')
  intervals=pd.to_numeric(ordered.fundingIntervalHours,errors='coerce')
  gap_limit=np.maximum(intervals,intervals.shift(1))*3600
  gaps=ordered.event_time.diff().dt.total_seconds()>gap_limit+1
  for pos in np.flatnonzero(gaps.to_numpy()):
   before=ordered.event_time.iloc[pos-1]+pd.Timedelta(seconds=float(gap_limit.iloc[pos]))
   after=ordered.event_time.iloc[pos]-pd.Timedelta(seconds=1)
   if before<=after:unknown.update(pd.date_range(before.floor('D'),after.floor('D'),freq='D'))
  lo=pd.Timestamp(meta.get('queried_from',str(index[0])))
  hi=pd.Timestamp(meta.get('queried_through_exclusive',str(index[-1]+pd.Timedelta(days=1))))
  covered=(index>=lo)&(index<hi)
  out.loc[covered,s]=grouped.reindex(index[covered],fill_value=0).values
  out.loc[out.index.isin(list(unknown)),s]=np.nan
  detail.append({**meta,'interior_gap_count':int(gaps.sum()),'unknown_funding_days':len(unknown)})
 return out,detail

def baseline_signals(raw,mask,funding):
 c=raw['close'];ret=np.log(c/c.shift(1))
 return {'B_size':raw['quote_volume'].rolling(30,min_periods=30).mean().shift(1),
         'B_lowvol10':-ret.rolling(10,min_periods=10).std(),
         'B_rev5':-np.log(c/c.shift(5)),
         'B_mom20':np.log(c/c.shift(20)),
         'B_funding7':-(funding/c).rolling(7,min_periods=7).sum()}

def evaluate_all(pool,mode,null_count=0):
 tag=f'{pool}_{mode}';path=OUT/f'factors_{tag}.pkl.gz'
 obj=pd.read_pickle(path);raw,mask,design=obj['raw'],obj['mask'],obj['design'];factors=obj['factors'];meta=obj['metadata']
 funding,coverage=funding_panel(raw['open'].index,raw['open'].columns)
 dump(OUT/f'funding_coverage_{tag}.json',coverage)
 records=[];ics=[];periods=[];t0=time.time()
 signals={**factors,**baseline_signals(raw,mask,funding)}
 for i,(name,sig) in enumerate(signals.items(),1):
  result=evaluate_factor(sig,raw['open'],raw['close'],mask,funding,horizons=(1,3,5),splits=SPLITS,entry_lag=2)
  family=meta['factors'][name]['family'] if name.startswith('alpha') else 'baseline'
  for key,acc in [('metrics',records),('ic',ics),('periods',periods)]:
   acc.append(result[key].assign(factor=name,pool=pool,mode=mode,family=family))
  if i%20==0 or i==len(signals):print(f'{tag}: evaluated {i}/{len(signals)} in {time.time()-t0:.1f}s',flush=True)
 pd.concat(records,ignore_index=True).to_parquet(OUT/f'metrics_{tag}.parquet',index=False)
 pd.concat(ics,ignore_index=True).to_parquet(OUT/f'ic_{tag}.parquet',index=False)
 pd.concat(periods,ignore_index=True).to_parquet(OUT/f'periods_{tag}.parquet',index=False)
 dump(OUT/f'evaluation_config_{tag}.json',{**result['config'],'design':design,'primary_h':3,'h1_h5':'exploratory; direction separately frozen per horizon','funding_price_sources':'funding response; historical mark-kline open fills separately tagged','formula_metadata':f'metadata_{tag}.json'})
 if null_count:
  rng=np.random.default_rng(20260910);null=[]
  for k in range(null_count):
   noise=rng.standard_normal(mask.shape);x=np.zeros_like(noise)
   for t in range(1,len(x)):x[t]=.9*x[t-1]+noise[t]
   sig=pd.DataFrame(x,index=mask.index,columns=mask.columns).where(mask)
   r=evaluate_factor(sig,raw['open'],raw['close'],mask,funding,horizons=(3,),splits=SPLITS,entry_lag=2)
   null.append(r['metrics'].assign(trial=k,pool=pool,phi=.9))
   if (k+1)%25==0:print(f'{tag}: null {k+1}/{null_count}',flush=True)
  pd.concat(null,ignore_index=True).to_parquet(OUT/f'null_{pool}.parquet',index=False)
 print(f'{tag}: evaluation done',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pool',choices=['current','historical'],required=True);ap.add_argument('--mode',choices=['paper','normalized'],required=True);ap.add_argument('--stage',choices=['compute','evaluate','all'],default='all');ap.add_argument('--null-count',type=int,default=0);a=ap.parse_args();OUT.mkdir(exist_ok=True)
 tag=f'{a.pool}_{a.mode}'
 if a.stage in ('compute','all'):
  raw,mask,design=load_inputs(a.pool);t0=time.time()
  print(f'{tag}: computing 101 factors, shape {mask.shape}',flush=True)
  factors,metadata=compute_factors(raw,mask,mode=a.mode)
  pd.to_pickle({'factors':factors,'raw':raw,'mask':mask,'design':design,'metadata':metadata},OUT/f'factors_{tag}.pkl.gz')
  dump(OUT/f'metadata_{tag}.json',metadata)
  print(f'{tag}: factors complete in {time.time()-t0:.1f}s; branches {metadata["alpha088_branch_diagnostic"]}',flush=True)
 if a.stage in ('evaluate','all'):evaluate_all(a.pool,a.mode,a.null_count)
if __name__=='__main__':main()
