"""Descriptive denomination check: BTC -> mBTC, identical economic prices.

Only the input representation changes: OHLC is divided by 1000, base volume
multiplied by 1000, and quote turnover unchanged. Execution prices and returns
always use the original market series. The original dev direction is retained.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from factors import compute_factors
import sys as _sys
_EVAL_ROOT = str(Path(__file__).resolve().parents[1])  # eval/ holds the shared measurement core
if _EVAL_ROOT not in _sys.path: _sys.path.insert(0, _EVAL_ROOT)
from factor_eval.engine import evaluate_factor, _spearman_rows
from run_pipeline import SPLITS

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def main():
 rows=[]
 for mode in ['paper','normalized']:
  obj=pd.read_pickle(OUT/f'factors_current_{mode}.pkl.gz')
  raw=obj['raw'];mask=obj['mask'];changed={k:v.copy() for k,v in raw.items()}
  for key in ['open','high','low','close']:changed[key]['BTCUSDT']/=1000
  changed['volume']['BTCUSDT']*=1000
  new,_=compute_factors(changed,mask,mode=mode)
  original=pd.read_parquet(OUT/f'metrics_current_{mode}.parquet')
  for name in ['alpha042','alpha094','alpha088','alpha040']:
   oldsig=obj['factors'][name];sig=new[name]
   pair,_=_spearman_rows(oldsig.to_numpy(),sig.to_numpy(),mask.to_numpy(),5)
   r=evaluate_factor(sig,raw['open'],raw['close'],mask,horizons=(3,),splits=SPLITS,entry_lag=2)
   for split in SPLITS:
    old=original[(original.factor==name)&(original.h==3)&(original.split==split)].iloc[0]
    newm=r['metrics'][r['metrics'].split==split].iloc[0]
    date=r['ic'][r['ic'].split==split];lo,hi=SPLITS[split]
    sr=pd.Series(pair,index=mask.index).loc[lo:hi]
    rows.append({'mode':mode,'factor':name,'split':split,'original_direction':int(old.sign),
     'original_ic':float(old.ic_mean),'redenominated_ic_fixed_direction':float(newm.ic_mean_raw*old.sign),
     'mean_rank_correlation_of_scores':float(sr.mean()),'n_ic':int(newm.n_ic)})
 pd.DataFrame(rows).to_csv(OUT/'unit_sensitivity.csv',index=False)
 (OUT/'unit_sensitivity.json').write_text(json.dumps({'method':__doc__,'note':'Research diagnostic chosen after inspecting the main run; not additional independent OOT evidence.','rows':rows},indent=2,ensure_ascii=False))
 print(pd.DataFrame(rows).round(5).to_string(index=False))

if __name__=='__main__':main()
