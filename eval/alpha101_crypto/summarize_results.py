"""Build auditable, common-period comparisons; never fill missing PnL.

Directions and factor eligibility come from dev only. Common-period exclusion
is a retrospective missing-data diagnostic, not a deployable trading filter.
Original strict full-split metrics remain unchanged in metrics_*.parquet.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
_EVAL_ROOT = str(Path(__file__).resolve().parents[1])  # eval/ holds the shared measurement core
if _EVAL_ROOT not in _sys.path: _sys.path.insert(0, _EVAL_ROOT)
from factor_eval.engine import _hac_mean_t

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
TAGS=['current_normalized','current_paper','historical_normalized','historical_paper']
SPLITS=['dev','val','oot']
LABELS={'current_normalized':'当前前十 · 归一化（主口径）','current_paper':'当前前十 · 原始输入',
        'historical_normalized':'历史月度前十 · 归一化','historical_paper':'历史月度前十 · 原始输入'}

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,(pd.Timestamp,Path)):return str(x)
 return x

def fmt(x,n=3):return 'NA' if pd.isna(x) else f'{x:.{n}f}'

def common_metrics(tag):
 p=pd.read_parquet(OUT/f'periods_{tag}.parquet')
 p=p[p.label_valid].copy()
 rows=[];excluded=[]
 for (h,split),g in p.groupby(['h','split']):
  pivot=g.pivot(index='signal_time',columns='factor',values='net')
  known=pivot.notna().all(axis=1)
  dates=pivot.index[known]
  for t in pivot.index[~known]:
   miss=g[(g.signal_time==t)&g.net.isna()]
   first=miss.iloc[0]
   excluded.append({'tag':tag,'h':h,'split':split,'signal_time':t,'entry_time':first.entry_time,
                    'exit_time':first.exit_time,'affected_factors':len(miss),
                    'statuses':','.join(sorted(miss.status.unique()))})
  for name,x in g.groupby('factor'):
   y=x[x.signal_time.isin(dates)]
   row={'tag':tag,'h':h,'split':split,'factor':name,'common_n':len(y),
        'common_excluded_n':len(pivot)-len(y),'common_active':int(y.active.sum())}
   for f in ['gross','long_gross','short_gross','trading_cost','funding','net_ex_funding','net']:
    row['common_'+f+'_bp']=float(y[f].mean()*1e4)
   row['common_turnover']=float(y.turnover.mean())
   regular=x.sort_values('signal_time').set_index('signal_time').net
   regular=regular.where(regular.index.isin(dates))
   row['common_net_t_hac']=_hac_mean_t(regular.to_numpy(),4)
   row['common_net_hac_lags_cycles']=4
   sd=y.net.std(ddof=1)
   row['common_sharpe_net']=float(y.net.mean()/sd*np.sqrt(365/h)) if sd>0 else np.nan
   row['common_breakeven_bp']=(row['common_gross_bp']-row['common_funding_bp'])/row['common_turnover'] if row['common_turnover']>0 else np.nan
   for cost in [3.0,6.5,12.0]:
    row['common_net_cost_'+str(cost)+'_bp']=row['common_gross_bp']-row['common_funding_bp']-row['common_turnover']*cost
   rows.append(row)
 return pd.DataFrame(rows),excluded

def main():
 summary={'primary':'current_normalized h=3',
  'data_gate':'dev coverage >=0.5, n_ic >=100, active/scheduled cycles >=0.5; research thresholds',
  'common_period_method':'Within each pool/input/h/split, remove the union of cycles with unknown net among all 101 factors and five baselines. Retain zero-PnL flat cycles; selection is retrospective, never an assumed live skip rule.',
  'tags':{}}
 all_common=[];exclusions=[];joined={}
 for tag in TAGS:
  cm,ex=common_metrics(tag);all_common.append(cm);exclusions+=ex
  m=pd.read_parquet(OUT/f'metrics_{tag}.parquet')
  m=m.merge(cm.drop(columns='tag'),on=['factor','h','split'],validate='one_to_one')
  m['active_fraction']=m.n_active/m.n_periods.replace(0,np.nan)
  m.to_parquet(OUT/f'summary_{tag}.parquet',index=False)
  x=m[m.h==3].copy();x.to_csv(OUT/f'summary_{tag}_h3.csv',index=False);joined[tag]=x
  a=x[x.factor.str.startswith('alpha')]
  d=a[a.split=='dev'].set_index('factor')
  good=d[(d.coverage>=.5)&(d.n_ic>=100)&(d.active_fraction>=.5)&np.isfinite(d.ic_mean)]
  names=good.index
  ic=a[a.factor.isin(names)].pivot(index='factor',columns='split',values='ic_mean')
  net=a[a.factor.isin(names)].pivot(index='factor',columns='split',values='common_net_bp')
  null=pd.read_parquet(OUT/f'null_{tag.split("_")[0]}.parquet')
  p95=null.groupby('split').ic_mean.quantile(.95).reindex(SPLITS)
  costs=a[a.factor.isin(['alpha088','alpha040','alpha061','alpha073'])][['factor','split','common_n','common_net_cost_3.0_bp','common_net_cost_6.5_bp','common_net_cost_12.0_bp']]
  info={'qualified':len(names),'qualified_by_family':good.groupby('family').size().to_dict(),
    'not_qualified':sorted(set(d.index)-set(names)),
    'leaders_dev':good.sort_values(['ic_mean'],ascending=False).head(6).reset_index().to_dict('records'),
    'positive_ic_all3':ic.index[ic.gt(0).all(axis=1)].tolist(),
    'positive_common_net_all3':net.index[net.gt(0).all(axis=1)].tolist(),
    'common_net_positive_counts':net.gt(0).sum().to_dict(),
    'above_single_null_p95_all3':ic.index[ic.gt(p95,axis='columns').all(axis=1)].tolist(),
    'null_single_signal_p95':p95.to_dict(),
    'full_split_net_known':a[a.factor.isin(names)].groupby('split').net_bp.count().to_dict(),
    'scheduled_counts':x.groupby('split').n_periods.first().to_dict(),
    'common_counts':x.groupby('split').common_n.first().to_dict(),
    'cost_sensitivity':costs.to_dict('records')}
  summary['tags'][tag]=info
 pd.concat(all_common,ignore_index=True).to_csv(OUT/'common_period_metrics.csv',index=False)
 pd.DataFrame(exclusions).to_csv(OUT/'common_period_exclusions.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False))
 write_appendix(joined,summary)
 overview_figure(joined['current_normalized'],summary)
 print(json.dumps({k:{n:v[n] for n in ['qualified','qualified_by_family','positive_common_net_all3','above_single_null_p95_all3','common_counts']} for k,v in summary['tags'].items()},ensure_ascii=False,indent=2))

def write_appendix(joined,summary):
 lines=['# Alpha101 v2：101 个因子的完整日线结果','',
  '运行日期：2026-09-10。四组比较均为持有 3 日、信号收盘后额外等待 1 日、单边交易成本 6.5 bp；方向只由各自开发段确定。',
  '', '“通过数据条件”仅表示 dev 覆盖≥50%、有效 IC 日期≥100、激活周期≥50%，不是统计或交易验收。',
  '', '**净 bp 是各组共同可核算周期的均值**：删除该组全部 101 因子与 5 基线中任一净收益未知的周期，含资金费标记价格代理，保留空仓零收益。它不是完整历史净收益；完整口径与逐周期缺口保留在原始结果中。',
  '', '字段：`价量`=不需行业或市值代理的 82 条；`代理`=19 条输入替代公式。`+/-`为该组 dev 冻结方向。IC 与收益使用不同采样：IC 每日，PnL 每 3 日一个不重叠周期。','']
 for tag in TAGS:
  x=joined[tag];a=x[x.factor.str.startswith('alpha')];info=summary['tags'][tag]
  lines += ['## '+LABELS[tag],'',f"共同可核算周期 dev/val/oot：{' / '.join(str(info['common_counts'][s]) for s in SPLITS)}。",'',
   '| 因子 | 输入 | dev 数据条件 | 方向 | dev IC | val IC | oot IC | 净 bp：dev / val / oot |',
   '|---|---|---|---:|---:|---:|---:|---|']
  for name,g in a.groupby('factor',sort=True):
   rows=g.set_index('split');d=rows.loc['dev'];family='价量' if d.family=='price_volume' else '代理'
   gate='不足' if name in info['not_qualified'] else '通过'
   lines.append(f"| {name} | {family} | {gate} | {'+' if d.sign>0 else '−'} | "+' | '.join(fmt(rows.loc[s,'ic_mean']) for s in SPLITS)+' | '+' / '.join(fmt(rows.loc[s,'common_net_bp'],1) for s in SPLITS)+' |')
  lines += ['',f'细项：[CSV](summary_{tag}_h3.csv)，含覆盖、ICIR、HAC t、有效样本、毛收益、交易费用、资金费、完整净收益与共同周期净收益。','']
 (OUT/'alpha101_all_factors.md').write_text('\n'.join(lines)+'\n')

def overview_figure(x,summary):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 a=x[x.factor.str.startswith('alpha')];names=set(a.factor)-set(summary['tags']['current_normalized']['not_qualified'])
 ic=a.pivot(index='factor',columns='split',values='ic_mean');dev=a[a.split=='dev'].set_index('factor')
 fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
 for ax,split in zip(axes[:2],['val','oot']):
  for fam,color in [('price_volume','#2563eb'),('input_proxy','#d97706')]:
   ids=[n for n in names if dev.loc[n,'family']==fam]
   if ids:ax.scatter(ic.loc[ids,'dev'],ic.loc[ids,split],c=color,s=24,alpha=.65,label=fam)
  for name in ['alpha088','alpha040','alpha061','alpha073']:
   ax.annotate(name[-3:],(ic.loc[name,'dev'],ic.loc[name,split]),xytext=(4,4),textcoords='offset points',fontsize=9)
  ax.axhline(0,c='#aaaaaa',lw=.8);ax.set_xlabel('Dev mean RankIC (direction selected here)');ax.set_ylabel(split.upper()+' mean RankIC')
  ax.set_title('Daily IC on mature labels');ax.grid(alpha=.15)
 axes[0].legend(fontsize=8)
 net=a.pivot(index='factor',columns='split',values='common_net_bp')
 labels=['alpha088','alpha040','alpha061','alpha073'];positions=np.arange(len(labels))
 for j,(split,color) in enumerate(zip(SPLITS,['#2563eb','#d97706','#16856b'])):
  axes[2].bar(positions+(j-1)*.25,net.loc[labels,split],.25,label=split,color=color)
 axes[2].axhline(0,c='#888888',lw=.8);axes[2].set_xticks(positions);axes[2].set_xticklabels([s[-3:] for s in labels]);axes[2].legend(fontsize=8)
 axes[2].set_title('Net bp per 3-day cycle, known common periods');axes[2].set_ylabel('Gross - transaction cost - estimated funding');axes[2].grid(axis='y',alpha=.15)
 fig.suptitle('Alpha101 | current top 10 | normalized inputs | h=3, extra wait=1 day')
 fig.supxlabel(f'{len(names)} factors pass the dev data gate. Common-period PnL excludes funding-unknown cycles; it is not a full-history estimate.',fontsize=9)
 dest=OUT/'figures'/'overview.png';dest.parent.mkdir(exist_ok=True);fig.savefig(dest,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)

if __name__=='__main__':main()
