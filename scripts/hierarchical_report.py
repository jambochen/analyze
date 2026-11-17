# 報告統一呈現論文的玩家差異模型；與共用係數的上一局模型比較，保留不利結果。
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import json
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import temporal_models as tm
import ordinal
import likelihood as lk

audit = pl.read_csv('outputs/tables/hierarchical_audit.csv')
assert audit.height == 12 and audit['mcse_to_dyadic_se'].max() < .1
exports = json.loads(Path('outputs/tables/likelihood_selection_models.json').read_text())
original = pl.read_csv('outputs/tables/likelihood_selection_prediction.csv')
raw = pl.read_parquet('data/processed/hierarchical_prediction.parquet')
speed_labels = {'Bullet': 'Short', 'Blitz': 'Medium', 'Classical': 'Long'}
speed_type = pl.Enum(list(speed_labels))
columns = ['speed', 'fold', 'group', 'model', 'vs', 'metric', 'n', 'mean', 'diff', 'se', 'lo', 'hi']
parts, curves, changes, manifest, settings, comparisons = [], [], [], [], [], []
for spec in exports:
    speed, fold = spec['speed'], spec['fold']
    prefix = f'hierarchical_{speed}_{fold}_slope'
    chosen = 'slope'
    cal = pl.read_csv(f'outputs/tables/{prefix}_calibration.csv')
    curves.append(cal.filter(pl.col('model').is_in(['baseline', 'recent', chosen])).with_columns(
        pl.when(pl.col('model') == chosen).then(pl.lit('adopted')).otherwise(pl.col('model')).alias('model')))
    source = pl.read_csv(f'outputs/tables/{prefix}_prediction.csv').with_columns(model=pl.lit('adopted'))
    saved = pl.read_parquet(f'data/processed/{prefix}_prediction.parquet')
    start, end = {'Jul-Sep': (1404172800, 1412121600), 'Oct-Dec': (1412121600, 1420070400)}[fold]
    d = raw.filter(pl.col('speed') == speed, pl.col('time').is_between(start, end, closed='left'))
    assert saved.select('game_id', 'a_id').equals(d.select('game_id', 'a_id'))
    base = spec['models']['baseline']
    base_data = tm.offset(d, base['baseline_coef'], 'Lichess')
    p0 = ordinal.probs(base_data['eta'].to_numpy(), base_data['width'].to_numpy())
    delta = 100*np.abs(saved['p_win'].to_numpy()-p0[:, 2])
    changes.append({'speed': speed, 'fold': fold, 'mean_abs_win_pp': delta.mean(), 'p90_abs_win_pp': np.quantile(delta, .9)})
    metadata = json.loads(Path(f'outputs/tables/{prefix}_fit.json').read_text())
    settings.append({'speed': speed, 'fold': fold, 'draws': metadata['draws_per_chain'], 'tune': metadata['tune'],
                     'chains': 4, 'rhat_max': audit.filter(pl.col('speed') == speed, pl.col('fold') == fold, pl.col('kind') == 'slope')['rhat_max'].item()})
    model = {'method': 'correlated_random_intercept_slope', 'posterior': f'data/processed/{prefix}.npz',
             'metadata': f'outputs/tables/{prefix}_fit.json', 'prediction': f'data/processed/{prefix}_prediction.parquet',
             'design_module': 'scripts/hierarchical.py', 'predictive_module': 'scripts/hierarchical_predictive.py',
             'specification': 'docs/hierarchical-replication.md',
             'prediction_rule': 'Posterior mean probabilities; integrate population effects for unseen players.',
             'posterior_sha256': hashlib.sha256(Path(f'data/processed/{prefix}.npz').read_bytes()).hexdigest()}
    comparisons.append(source.filter(pl.col('vs') == 'recent').select(columns))
    parts.append(source.filter(pl.col('vs') == 'baseline', pl.col('group').is_in(['all', 'novice', 'established'])).select(columns))
    manifest.append({'speed': speed, 'fold': fold, 'train_end_exclusive': spec['train_end_exclusive'],
                     'selection': 'Paper-based model specified for all time controls; not selected by test performance', 'model': model})
parts.append(original.filter(pl.col('model') == 'recent', pl.col('vs') == 'baseline').select(columns))
p = pl.concat(parts).with_columns(pl.col('speed').cast(speed_type)).sort('speed', 'fold')
c = pl.concat(curves).with_columns(pl.col('speed').cast(speed_type)).sort('speed', 'fold')
changes = pl.DataFrame(changes).with_columns(pl.col('speed').cast(speed_type)).sort('speed', 'fold')
assert p.filter(pl.col('model') == 'adopted').height == 6*3*3
p.write_csv('outputs/tables/hierarchical_report_prediction.csv')
c.write_csv('outputs/tables/hierarchical_report_calibration.csv')
changes.write_csv('outputs/tables/hierarchical_report_changes.csv')
pl.DataFrame(settings).write_csv('outputs/tables/hierarchical_report_settings.csv')
Path('outputs/tables/hierarchical_report_models.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
pooled = pl.read_csv('outputs/tables/hierarchical_prediction_pooled.csv').filter(pl.col('kind') == 'slope', pl.col('vs') == 'recent')
ll = pooled.filter(pl.col('metric') == 'log_loss')
ll = ll.with_columns(p_holm=pl.Series(lk.holm(ll['p'].to_numpy())))
pooled.join(ll.select('speed', 'p_holm'), on='speed').write_csv('outputs/tables/hierarchical_report_comparison.csv')
pl.concat(comparisons).write_csv('outputs/tables/hierarchical_report_periods.csv')

plt.rcParams.update({'svg.fonttype': 'path', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, group, title in zip(axes, ['all', 'novice', 'established'], ['All games', 'At least one high-RD player', 'Both players low-RD']):
    for j, (model, label, color) in enumerate([('recent', 'With previous-game data', '#b05a16'), ('adopted', 'Hierarchical model', '#33414a')]):
        d = p.filter(pl.col('group') == group, pl.col('metric') == 'score_mse', pl.col('model') == model)
        reference = d['mean'].to_numpy()-d['diff'].to_numpy()
        ax.errorbar(-100*d['diff'].to_numpy()/reference, np.arange(d.height)+(j-.5)*.18,
                    xerr=196*d['se'].to_numpy()/reference, fmt='o', color=color, capsize=3, label=label)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{speed_labels[r['speed']]} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel='Relative score-MSE reduction (%)')
    ax.axvline(0, color='grey', linestyle='--')
    ax.grid(axis='x', alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig('outputs/figures/hierarchical_report_gain.svg', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.6))
for column, label, marker in [('mean_abs_win_pp', 'Mean', 'o'), ('p90_abs_win_pp', '90th percentile', 'x')]:
    ax.scatter(changes[column], np.arange(changes.height), label=label, marker=marker, s=55)
ax.set(yticks=np.arange(changes.height), yticklabels=[f"{speed_labels[r['speed']]} / {r['fold']}" for r in changes.iter_rows(named=True)], xlabel='Absolute win-probability change (percentage points)')
ax.invert_yaxis()
ax.grid(alpha=.15)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('outputs/figures/hierarchical_report_changes.svg', bbox_inches='tight')
plt.close(fig)

# 用共同分箱比較相同對局，不把不同模型各自分箱的摘要直接排名。
gaps = c.filter(pl.col('group') == 'all', pl.col('binning') == 'baseline_prediction').group_by('speed', 'fold', 'model', 'target').agg(
    (100*((pl.col('predicted')-pl.col('observed')).abs()*pl.col('n')).sum()/pl.col('n').sum()).alias('gap_pp'))
gaps.write_csv('outputs/tables/hierarchical_report_calibration_gap.csv')
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, target in zip(axes, ['loss', 'draw', 'win']):
    for j, (model, label, color) in enumerate([('baseline', 'Baseline', '#888888'), ('recent', 'With previous-game data', '#b05a16'), ('adopted', 'Hierarchical model', '#247a70')]):
        d = gaps.filter(pl.col('target') == target, pl.col('model') == model).sort('speed', 'fold')
        ax.scatter(d['gap_pp'], np.arange(d.height)+(j-1)*.18, label=label, color=color)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{speed_labels[r['speed']]} / {r['fold']}" for r in d.iter_rows(named=True)], title=target.title(), xlabel='Weighted absolute calibration gap (pp)')
    ax.grid(axis='x', alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig('outputs/figures/hierarchical_report_classes.svg', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fold in zip(axes, ['Jul-Sep', 'Oct-Dec']):
    for model, label, color in [('baseline', 'Baseline', '#888888'), ('recent', 'With previous-game data', '#b05a16'), ('adopted', 'Hierarchical model', '#247a70')]:
        d = c.filter(pl.col('speed') == 'Blitz', pl.col('fold') == fold, pl.col('group') == 'all', pl.col('target') == 'score', pl.col('binning') == 'baseline_prediction', pl.col('model') == model).sort('bin')
        reference = c.filter(pl.col('speed') == 'Blitz', pl.col('fold') == fold, pl.col('group') == 'all', pl.col('target') == 'score', pl.col('binning') == 'baseline_prediction', pl.col('model') == 'baseline').sort('bin')
        assert d['n'].equals(reference['n']) and d['observed'].equals(reference['observed'])
        ax.plot(reference['predicted'], (d['observed']-d['predicted'])*100, marker='o', label=label, color=color)
    ax.axhline(0, color='grey', linestyle='--')
    ax.set(title=f'Medium / {fold}', xlabel='Baseline mean expected score', ylabel='Observed - predicted score (pp)')
    ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig('outputs/figures/hierarchical_report_score.svg', bbox_inches='tight')
plt.close(fig)
print(ll.select('speed', 'diff', 'p_holm'))
