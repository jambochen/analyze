# 整理本地三類預測結果，不讀取文獻的得分 cross entropy。
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

comparison = pl.read_csv('outputs/tables/trueskill_comparison.csv')
main = pl.read_csv('outputs/tables/trueskill_decision.csv')
comparison.filter(pl.col('fold') == 'pooled').join(main.select('speed','p_holm'),on='speed').write_csv('outputs/tables/trueskill_report_comparison.csv')
comparison.filter(pl.col('fold') != 'pooled',pl.col('vs') == 'recent').write_csv('outputs/tables/trueskill_report_periods.csv')
cal = pl.read_csv('outputs/tables/trueskill_calibration.csv').filter(pl.col('group') == 'all',pl.col('binning') == 'baseline_prediction')
assert cal.group_by('speed','fold','target','bin').agg(pl.col('n').n_unique(),pl.col('observed').n_unique()).select(
    (pl.col('n') == 1).all() & (pl.col('observed') == 1).all()).item()
gaps = cal.group_by('speed','fold','model','target').agg(
    (100*((pl.col('predicted')-pl.col('observed')).abs()*pl.col('n')).sum()/pl.col('n').sum()).alias('gap_pp'))
gaps.write_csv('outputs/tables/trueskill_report_calibration.csv')
plt.rcParams.update({'svg.fonttype':'path','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
labels = {'Bullet':'Short','Blitz':'Medium','Classical':'Long'}
speed_type = pl.Enum(list(labels))
fig,axes = plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,target in zip(axes,['loss','draw','win']):
    for j,(model,label,color) in enumerate([('baseline','Rating baseline','#888888'),('recent','Previous-game model','#b05a16'),('trueskill','TrueSkill','#247a70')]):
        d = gaps.filter(pl.col('model') == model,pl.col('target') == target).with_columns(pl.col('speed').cast(speed_type)).sort('speed','fold')
        ax.scatter(d['gap_pp'],np.arange(len(d))+(j-1)*.18,label=label,color=color)
    ax.set(yticks=np.arange(len(d)),yticklabels=[f"{labels[r['speed']]} / {r['fold']}" for r in d.iter_rows(named=True)],
           title=target.title(),xlabel='Weighted absolute calibration gap (pp)')
    ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,0,1,.9))
fig.savefig('outputs/figures/trueskill_report_calibration.svg',bbox_inches='tight')
plt.close(fig)
print(gaps.filter(pl.col('model') == 'trueskill').join(gaps.filter(pl.col('model') == 'recent'),on=['speed','fold','target'],suffix='_recent')
      .select('speed','fold','target',(pl.col('gap_pp')-pl.col('gap_pp_recent')).alias('gap_difference_pp')).sort('speed','fold','target'))
