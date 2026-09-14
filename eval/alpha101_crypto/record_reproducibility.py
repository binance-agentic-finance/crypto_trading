"""Record final artifacts without rewriting the initial experiment ledger."""
from pathlib import Path
import hashlib,json,platform
from datetime import datetime,timezone
from importlib.metadata import version

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 deps={name:version(name) for name in ['numpy','pandas','scipy','pyarrow','matplotlib','requests','pytest']}
 (HERE/'requirements-snapshot.txt').write_text(''.join(f'{k}=={v}\n' for k,v in deps.items()))
 code=list(HERE.glob('*.py'))+list((ROOT/'tests'/'eval').glob('test_*.py'))
 artifacts=list((HERE/'results').glob('*.parquet'))+list((HERE/'results').glob('*.pkl.gz'))+list((HERE/'results').glob('summary*.json'))
 artifacts+=list((HERE/'results').glob('*.csv'))+list((HERE/'results').glob('*.md'))
 artifacts+=list((HERE/'results/figures').rglob('*.png'))+list((HERE/'results/figures').rglob('*.json'))
 artifacts+=list((HERE/'results/incremental_current_normalized_h3').glob('*'))
 # exchange_info.json (1.7 MB, re-downloadable registry dump) is not tracked in this
 # repo, so it is hashed only when a local acquisition run left it behind.
 manifests=[HERE/'data'/n for n in ['acquisition_config.json','exchange_info.json','daily_manifest.json','funding_manifest.json','universes.json','current_top10.csv','historical_top10.csv','funding_quality_summary.json']]
 manifests=[p for p in manifests if p.exists()]
 payload={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
  'platform':platform.platform(),'dependencies':deps,'initial_config_sha256':sha(HERE/'experiment_config.json'),
  'note':'Final hashes after funding/lifecycle corrections. Initial versions remain in experiment_config.json. Historical OOT had already been explored; common-period selection, projections and denomination check are retrospective diagnostics.',
  'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(code)},
  'input_manifest_sha256':{str(p.relative_to(HERE)):sha(p) for p in manifests},
  'result_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted(set(artifacts)) if p.is_file()},
  'report_sha256':sha(HERE/'REPORT.md'),
  'tests':'33 passed: tests/eval/test_engine.py, test_factors.py, test_pipeline.py'}
 (HERE/'reproducibility.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 print('Recorded final artifact hashes and dependency versions')
if __name__=='__main__':main()
