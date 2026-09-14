"""Download an auditable snapshot from Binance public endpoints."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, threading, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings('ignore', category=Warning, module='urllib3')
HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
START=pd.Timestamp('2021-01-01',tz='UTC')
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
STABLE={'USDC','BUSD','FDUSD','TUSD','USDP','DAI','USDE','USD1','U','USDD'}
_lock=threading.Lock(); _next_call=0.0

def api(method,url,**kw):
    global _next_call
    for attempt in range(4):
        with _lock:
            now=time.monotonic(); wait=max(0,_next_call-now)
            _next_call=max(now,_next_call)+0.29
        if wait: time.sleep(wait)
        try:
            r=requests.request(method,url,timeout=(10,35),**kw)
            if r.status_code in (418,429):
                time.sleep(min(60,max(5,int(r.headers.get('Retry-After','15')))))
                continue
            r.raise_for_status(); out=r.json()
            if isinstance(out,dict) and isinstance(out.get('code'),int) and out['code']<0:
                raise ValueError(str(out))
            return out
        except (requests.RequestException,ValueError) as e:
            if attempt==3: raise
            time.sleep(1+attempt)
    raise RuntimeError('rate-limit retries exhausted')

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));tmp.replace(path)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def daily_one(sym,end):
    path=DATA/'daily'/f'{sym}.parquet'
    if path.exists():
        frame=pd.read_parquet(path)
        return {'symbol':sym,'status':'cached','rows':len(frame),'first':str(frame.ts.min()),'last':str(frame.ts.max()),'sha256':sha(path)}
    try:
        rows=[];cur=int(START.timestamp()*1000);stop=int(end.timestamp()*1000)
        while cur<stop:
            data=api('GET','https://fapi.binance.com/fapi/v1/klines',params={'symbol':sym,'interval':'1d','startTime':cur,'endTime':stop-1,'limit':1000})
            if not isinstance(data,list): raise ValueError(str(data)[:180])
            if not data: break
            rows.extend(data)
            nxt=int(data[-1][0])+86400000
            if nxt<=cur: raise RuntimeError('nonadvancing pagination')
            cur=nxt
            if len(data)<1000: break
        if not rows: return {'symbol':sym,'status':'no_data','rows':0}
        df=pd.DataFrame(rows,columns=COLS).drop_duplicates('open_time').sort_values('open_time')
        for c in COLS: df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df[(df.open_time>=int(START.timestamp()*1000))&(df.close_time<stop)].copy()
        df['ts']=pd.to_datetime(df.open_time,unit='ms',utc=True)
        bad=(df[['open','high','low','close']]<=0).any(axis=1)|(df.high<df.low)|(df.volume<0)|(df.quote_volume<0)
        if bad.any(): raise ValueError(f'{int(bad.sum())} invalid OHLCV bars')
        df.drop(columns=['ignore']).to_parquet(path,index=False)
        return {'symbol':sym,'status':'ok','rows':len(df),'first':str(df.ts.min()),'last':str(df.ts.max()),'sha256':sha(path)}
    except Exception as e: return {'symbol':sym,'status':'error','error':str(e)[:240]}

def _funding_write_parquet(path,frame):
    """Atomic funding/cache output so interrupted downloads are resumable."""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_parquet(tmp,index=False)
    tmp.replace(path)


def _funding_mark_open_refs(sym,event_times):
    """Batch historical mark opens, never substitute a future bar's close.

    Funding timestamps may be a few milliseconds after an hourly boundary.
    Only [0,1) second offsets are accepted; other events remain missing.
    """
    times=pd.to_datetime(event_times,utc=True)
    floors=times.dt.floor('h')
    offsets=(times-floors).dt.total_seconds()
    aligned=times.notna()&offsets.ge(0)&offsets.lt(1)
    info={'unaligned_events':int((~aligned).sum()),'requests':0,
          'interval':None,'errors':[]}
    refs=pd.Series(np.nan,index=times.index,dtype=float)
    ref_times=pd.Series(pd.NaT,index=times.index,dtype='datetime64[ns, UTC]')
    if not aligned.any():return refs,ref_times,info
    wanted=floors.loc[aligned]
    hours=next(h for h in (8,4,1) if wanted.dt.hour.mod(h).eq(0).all())
    interval=f'{hours}h';info['interval']=interval
    step=hours*3600000
    lo=int(wanted.min().timestamp()*1000)
    hi=int(wanted.max().timestamp()*1000)+step
    info.update(start_ms=lo,end_exclusive_ms=hi)
    cache=DATA/'funding_mark_klines'/f'{sym}_{interval}.parquet'
    marks=pd.DataFrame(columns=['open_time','mark_open'])
    if cache.exists():
        marks=pd.read_parquet(cache)
        marks=marks[np.isfinite(marks.mark_open)&marks.mark_open.gt(0)]
    have=set(marks.open_time.astype('int64')) if len(marks) else set()
    wanted_ms=(wanted.astype('int64')//1000000).astype('int64')
    missing_ms=sorted(set(wanted_ms)-have)
    if missing_ms:
        cursor=missing_ms[0]
        stop=missing_ms[-1]+step
        try:
            while cursor<stop:
                part=api('GET','https://fapi.binance.com/fapi/v1/markPriceKlines',
                         params={'symbol':sym,'interval':interval,'startTime':cursor,
                                 'endTime':stop-1,'limit':1000})
                info['requests']+=1
                if not isinstance(part,list):raise ValueError('non-list mark kline response')
                if not part:break
                batch=pd.DataFrame({'open_time':[int(r[0]) for r in part],
                                    'mark_open':pd.to_numeric([r[1] for r in part],errors='coerce')})
                batch=batch[(batch.open_time>=cursor)&(batch.open_time<stop)]
                if batch.empty:raise ValueError('mark kline response outside requested range')
                nxt=int(batch.open_time.max())+step
                if nxt<=cursor:raise ValueError('nonadvancing mark kline pagination')
                batch=batch[np.isfinite(batch.mark_open)&batch.mark_open.gt(0)]
                marks=(pd.concat([marks,batch],ignore_index=True) if len(marks) else batch.copy())
                marks=marks.drop_duplicates('open_time',keep='last').sort_values('open_time')
                _funding_write_parquet(cache,marks)
                cursor=nxt
                # A short server page is not proof of complete historical coverage.
        except Exception as e:
            info['errors'].append(str(e)[:240])
    if len(marks):
        lookup=marks.drop_duplicates('open_time').set_index('open_time').mark_open
        refs.loc[aligned]=wanted_ms.map(lookup).to_numpy()
        ok=refs.notna()&np.isfinite(refs)&refs.gt(0)
        refs=refs.where(ok)
        ref_times.loc[ok]=floors.loc[ok]
        info['cache_path']=str(cache.relative_to(HERE))
        info['cache_sha256']=sha(cache) if cache.exists() else None
    info['resolved_events']=int(refs.notna().sum())
    return refs,ref_times,info


def _funding_gap_diagnostics(frame):
    """Describe unknown event windows; never manufacture zero funding rates."""
    elapsed=frame.ts.diff().dt.total_seconds()
    expected=frame.fundingIntervalHours*3600
    previous_interval=frame.fundingIntervalHours.shift(1)
    gap=elapsed.gt(expected+1)&np.isfinite(expected)&expected.gt(0)
    changed=frame.fundingIntervalHours.ne(previous_interval)
    transition=gap&changed&elapsed.le(pd.concat([expected,previous_interval*3600],axis=1).max(axis=1)+1)
    frame['funding_gap_before']=gap
    frame['funding_gap_class']=np.where(transition,'interval_transition_ambiguous',np.where(gap,'unexplained','none'))
    frame['funding_gap_seconds']=elapsed.where(gap)
    affected=set();unexplained_days=set();events=[]
    for i in frame.index[gap]:
        prev=frame.ts.iloc[i-1];cur=frame.ts.iloc[i]
        step=pd.Timedelta(hours=float(frame.fundingIntervalHours.iloc[i]))
        possible=pd.date_range(prev.floor('h')+step,cur.floor('h'),freq=step,inclusive='left')
        days=sorted({str(t.date()) for t in possible})
        affected.update(days)
        if not transition.iloc[i]:unexplained_days.update(days)
        events.append({'previous_time':str(prev),'current_time':str(cur),
                       'elapsed_hours':float(elapsed.iloc[i]/3600),
                       'previous_interval_hours':float(previous_interval.iloc[i]),
                       'current_interval_hours':float(frame.fundingIntervalHours.iloc[i]),
                       'classification':str(frame.funding_gap_class.iloc[i]),
                       'affected_utc_days':days})
    return {'gap_count':int(gap.sum()),'unexplained_gap_count':int((gap&~transition).sum()),
            'interval_transition_gap_count':int(transition.sum()),
            'gap_affected_utc_days':sorted(affected),
            'unexplained_gap_days_utc':sorted(unexplained_days),'gaps':events,
            'method':'adjacent actual event gap > current fundingIntervalHours + 1 second; interval switches flagged separately',
            'leading_boundary':'no gap inferred before first retained event; listing/start boundary is left-censored'}


def funding_one(sym,end):
    """Retain rates even when settlement mark prices require a flagged proxy."""
    path=DATA/'funding'/f'{sym}.parquet';stamp=path.with_suffix('.json')
    raw_path=DATA/'funding_raw'/f'{sym}.json'
    rows=[];source='bapi_funding_history';total=None;cached=False;old={}
    raw_origin='bapi_response';raw_stop=None;cache_rejected=None
    try:
        if path.exists() and stamp.exists():
            old=json.loads(stamp.read_text())
            same_end=pd.Timestamp(old.get('queried_through_exclusive'))==pd.Timestamp(end)
            actual_hash=sha(path)
            hash_ok=not old.get('sha256') or old['sha256']==actual_hash
            if not hash_ok:cache_rejected='sha256_mismatch_refetched'
            if same_end and hash_ok and old.get('status') in ('ok','missing'):
                if old.get('schema_version')==3 and old.get('status')=='ok' and raw_path.exists():
                    return {**old,'cache_hit':True}
                df=pd.read_parquet(path);cached=True;total=old.get('total_history_rows')
                raw_origin=old.get('raw_origin','legacy_cached_funding_response')
                raw_stop=old.get('pagination_stop','legacy_cache')
                if not raw_path.exists():
                    dump(raw_path,{'symbol':sym,'source':source,'raw_origin':raw_origin,
                                   'queried_through_exclusive':str(end),
                                   'records':df.to_dict('records')})
        if not cached:
            start_ms=int(START.timestamp()*1000)
            previous_oldest=None
            for page in range(1,100):
                out=api('POST','https://www.binance.com/bapi/futures/v1/public/future/common/get-funding-rate-history',
                        json={'symbol':sym,'page':page,'rows':1000})
                if not isinstance(out,dict) or not out.get('success') or not isinstance(out.get('data'),list):
                    raise ValueError('invalid funding history response')
                part=out['data'];total=int(out.get('total',0))
                if not part:
                    seen={int(r['calcTime']) for r in rows}
                    if seen and min(seen)<start_ms:raw_stop='reached_start_boundary';break
                    if total>0 and len(seen)>=total:raw_stop='reached_total';break
                    raise ValueError(f'incomplete funding pagination: empty page {page}, distinct={len(seen)}, total={total}')
                event_ms=[int(r['calcTime']) for r in part]
                if any(a<b for a,b in zip(event_ms,event_ms[1:])):
                    raise ValueError('funding history is not descending in time')
                rows.extend(part)
                oldest=min(event_ms)
                if oldest<start_ms:raw_stop='reached_start_boundary';break
                if total>0 and len({int(r['calcTime']) for r in rows})>=total:
                    raw_stop='reached_total';break
                if previous_oldest is not None and oldest>=previous_oldest:
                    raise ValueError('nonadvancing funding history pagination')
                previous_oldest=oldest
            else:raise ValueError('funding pagination limit exhausted')
            if not rows:raise ValueError('empty funding history')
            dump(raw_path,{'symbol':sym,'source':source,'raw_origin':raw_origin,
                           'queried_through_exclusive':str(end),'total_history_rows':total,
                           'pagination_stop':raw_stop,'records':rows})
            df=pd.DataFrame(rows).rename(columns={'calcTime':'fundingTime','lastFundingRate':'fundingRate'})
        df['fundingTime']=pd.to_numeric(df.fundingTime,errors='coerce')
        df['ts']=pd.to_datetime(df.fundingTime,unit='ms',utc=True,errors='coerce')
        if df.ts.isna().any():raise ValueError('invalid funding timestamp; raw response retained')
        for c in ['fundingRate','markPrice','fundingIntervalHours']:
            if c not in df:df[c]=np.nan
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df[(df.ts>=START)&(df.ts<end)].copy()
        duplicate_rows=int(df.duplicated('fundingTime').sum())
        if df.groupby('fundingTime').fundingRate.nunique(dropna=False).gt(1).any():
            raise ValueError('conflicting rates at one funding timestamp; raw response retained')
        df=df.drop_duplicates('fundingTime').sort_values('ts').reset_index(drop=True)
        if df.empty:raise ValueError('no funding history in requested window; raw response retained')
        if 'funding_response_mark_price' not in df:df['funding_response_mark_price']=df.markPrice
        if 'mark_price_source' not in df:
            valid=np.isfinite(df.markPrice)&df.markPrice.gt(0)
            df['mark_price_source']=np.where(valid,'funding_response','missing')
        missing=~np.isfinite(df.markPrice)|df.markPrice.le(0)
        df.loc[missing,'markPrice']=np.nan
        df.loc[missing,'mark_price_source']='missing'
        if 'mark_price_reference_time' not in df:
            df['mark_price_reference_time']=df.ts.where(~missing)
        else:df['mark_price_reference_time']=pd.to_datetime(df.mark_price_reference_time,utc=True)
        proxy_info={'interval':None,'requests':0,'unaligned_events':0,'errors':[]}
        if missing.any():
            refs,ref_times,proxy_info=_funding_mark_open_refs(sym,df.loc[missing,'ts'])
            filled=refs.notna()
            ids=refs.index[filled]
            df.loc[ids,'markPrice']=refs.loc[ids]
            df.loc[ids,'mark_price_reference_time']=ref_times.loc[ids]
            df.loc[ids,'mark_price_source']='mark_kline_open_'+str(proxy_info['interval'])
        missing_mark=~np.isfinite(df.markPrice)|df.markPrice.le(0)
        invalid_rate=~np.isfinite(df.fundingRate)
        invalid_interval=~np.isfinite(df.fundingIntervalHours)|df.fundingIntervalHours.le(0)
        df['mark_price_missing']=missing_mark
        df['funding_input_valid']=~(missing_mark|invalid_rate|invalid_interval)
        df['mark_price_time_offset_seconds']=(df.ts-df.mark_price_reference_time).dt.total_seconds()
        gap_info=_funding_gap_diagnostics(df)
        _funding_write_parquet(path,df)
        original_missing=~np.isfinite(df.funding_response_mark_price)|df.funding_response_mark_price.le(0)
        proxies=df.mark_price_source.str.startswith('mark_kline_open_')
        result={'schema_version':3,'symbol':sym,
                'status':'ok' if df.funding_input_valid.all() and gap_info['gap_count']==0 else 'missing',
                'source':source,'rows':len(df),'first':str(df.ts.min()),'last':str(df.ts.max()),
                'total_history_rows':total,'intervals_hours':sorted(df.fundingIntervalHours.dropna().unique().tolist()),
                'sha256':sha(path),'queried_through_exclusive':str(end),'cache_hit':cached,
                'raw_path':str(raw_path.relative_to(HERE)),'raw_sha256':sha(raw_path),
                'raw_origin':raw_origin,'pagination_stop':raw_stop,'duplicate_time_rows':duplicate_rows,
                'pagination_complete':raw_stop in ('reached_start_boundary','reached_total'),
                'pagination_status':('verified' if raw_stop in ('reached_start_boundary','reached_total') else 'legacy_success_cache_not_revalidated'),
                'validation_kind':('full_pagination' if raw_stop in ('reached_start_boundary','reached_total') else 'unvalidated_legacy'),
                'cache_rejected_reason':cache_rejected,
                'response_missing_mark_count':int(original_missing.sum()),
                'proxy_mark_count':int(proxies.sum()),'missing_mark_count':int(missing_mark.sum()),
                'invalid_rate_count':int(invalid_rate.sum()),'invalid_interval_count':int(invalid_interval.sum()),
                'mark_price_source_counts':{str(k):int(v) for k,v in df.mark_price_source.value_counts().items()},
                'mark_proxy':proxy_info,'event_gap_diagnostics':gap_info}
        # External cross-checks apply only to exactly the audited cache bytes.
        # Keep source completeness separate from available-API count agreement.
        if cached and result['sha256']==old.get('sha256'):
            for key in ('cache_coverage_validation','event_gap_crosscheck','boundary_audit'):
                if key in old:result[key]=old[key]
            validation=result.get('cache_coverage_validation',{})
            if validation.get('status')=='available_api_count_and_boundary_verified':
                result['pagination_status']=validation['status']
                result['pagination_complete']=True
                result['validation_kind']='counts_and_boundary_samples'
        dump(stamp,result)
        return result
    except Exception as e:
        if rows:
            dump(raw_path,{'symbol':sym,'source':source,'raw_origin':raw_origin,
                           'queried_through_exclusive':str(end),'total_history_rows':total,
                           'pagination_stop':raw_stop,'records':rows,'processing_error':str(e)[:240]})
        return {'symbol':sym,'status':'error','source':source,'error':str(e)[:240],
                'raw_path':str(raw_path.relative_to(HERE)) if raw_path.exists() else None}

def prepare_universes(meta,end):
    frames={p.stem:pd.read_parquet(p).set_index('ts') for p in (DATA/'daily').glob('*.parquet')}
    qv=pd.DataFrame({s:d.quote_volume for s,d in frames.items()}).sort_index()
    idx=pd.date_range(START,end-pd.Timedelta(days=1),freq='D',tz='UTC')
    qv=qv.reindex(idx)
    registry={x['symbol']:x for x in meta['symbols']}
    for s in qv.columns:
        on=pd.to_datetime(registry[s]['onboardDate'],unit='ms',utc=True)
        off=pd.to_datetime(registry[s]['deliveryDate'],unit='ms',utc=True)
        qv[s]=qv[s].where((idx>=on)&(idx+pd.Timedelta(days=1)<=off))
    age=qv.notna().cumsum()
    eligible=(age>=60)&qv.notna()
    adv=qv.rolling(30,min_periods=30).mean()
    active={s['symbol'] for s in meta['symbols'] if s['status']=='TRADING'}
    current=adv.iloc[-1].where(eligible.iloc[-1]).dropna()
    current=current[current.index.isin(active)].sort_values(ascending=False).head(10)
    rows=[]
    for t in pd.date_range(pd.Timestamp('2022-04-01',tz='UTC'),end,freq='MS'):
        prev=t-pd.Timedelta(days=1)
        if prev not in adv.index:continue
        rank=adv.loc[prev].where(eligible.loc[prev]).dropna().sort_values(ascending=False).head(10)
        for k,(s,v) in enumerate(rank.items(),1):rows.append({'rebalance':t,'rank':k,'symbol':s,'adv30_usdt':float(v)})
    history=pd.DataFrame(rows);history.to_csv(DATA/'historical_top10.csv',index=False)
    current.rename('adv30_usdt').rename_axis('symbol').reset_index().to_csv(DATA/'current_top10.csv',index=False)
    union=sorted(set(history.symbol)|set(current.index))
    dump(DATA/'universes.json',{'current_as_of':str(end),'current_top10':list(current.index),'historical_union':sorted(history.symbol.unique()),'needed_symbols':union,'method':'30 completed UTC days quote volume; monthly reconstitution uses prior month end; >=60 full live daily bars; registry onboard/delivery dates exclude post-delivery placeholders; current exchange registry has historical survivorship limits'})
    print('CURRENT TOP10\n'+current.div(1e6).round(1).to_string(),flush=True)
    print(f'Historical rebalances: {history.rebalance.nunique()}; union: {len(union)}',flush=True)
    return union

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['daily','funding','all'],default='all');args=parser.parse_args()
    for sub in ['daily','funding']:(DATA/sub).mkdir(parents=True,exist_ok=True)
    server=api('GET','https://fapi.binance.com/fapi/v1/time')
    end=pd.to_datetime(server['serverTime'],unit='ms',utc=True).floor('D')
    if args.stage in ('daily','all'):
        meta=api('GET','https://fapi.binance.com/fapi/v1/exchangeInfo');dump(DATA/'exchange_info.json',meta)
        allmeta=[s for s in meta['symbols'] if s['quoteAsset']=='USDT' and s['contractType']=='PERPETUAL' and s.get('underlyingType')=='COIN' and s['baseAsset'] not in STABLE and s['status']!='PENDING_TRADING']
        symbols=sorted(s['symbol'] for s in allmeta)
        dump(DATA/'acquisition_config.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'server_time':server,'start':str(START),'end_exclusive':str(end),'symbols':symbols,'selection':'exchangeInfo USDT PERPETUAL COIN; exclude stablecoin bases and pending; include settling where API returns history','workers':4,'requests_per_second_max':3.45})
        print(f'Fetching complete daily bars: {len(symbols)} symbols, {START.date()} to {end.date()} exclusive',flush=True)
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fut={pool.submit(daily_one,s,end):s for s in symbols}
            for i,f in enumerate(concurrent.futures.as_completed(fut),1):
                r=f.result();results.append(r)
                if i%25==0 or i==len(fut):
                    print(f'daily {i}/{len(fut)} ok={sum(x["status"] in ("ok","cached") for x in results)} failures={sum(x["status"]=="error" for x in results)}',flush=True)
                    dump(DATA/'daily_manifest.json',results)
        union=prepare_universes(meta,end)
    else:
        cfg=json.loads((DATA/'acquisition_config.json').read_text());end=pd.Timestamp(cfg['end_exclusive'])
        union=json.loads((DATA/'universes.json').read_text())['needed_symbols']
    if args.stage in ('funding','all'):
        print(f'Funding histories for {len(union)} evaluated symbols',flush=True);res=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fut={pool.submit(funding_one,s,end):s for s in union}
            for i,f in enumerate(concurrent.futures.as_completed(fut),1):
                r=f.result();res.append(r)
                if i%5==0 or i==len(fut):
                    print(f'funding {i}/{len(fut)} ok={sum(x["status"]=="ok" for x in res)} failures={sum(x["status"]=="error" for x in res)}',flush=True)
                    dump(DATA/'funding_manifest.json',res)
    print('Acquisition complete',flush=True)
if __name__=='__main__':main()
