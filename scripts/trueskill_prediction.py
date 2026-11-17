# 有限候選只用訓練內最後一月選擇；測試期間僅按可用賽果更新玩家狀態。
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl
from scipy.special import ndtri
from scipy.stats import norm
import likelihood as lk
import likelihood_rs
import rd_spline as spline
import state_adjusted as sa
import temporal_models as tm
from hierarchical_predictive import calibration_rows
from trueskill_model import replay_arrays

all_games = pl.read_parquet('data/processed/trueskill_games.parquet')
all_rows = pl.read_parquet('data/processed/hierarchical_prediction.parquet')
exports = json.loads(Path('outputs/tables/likelihood_selection_models.json').read_text())
configs = np.array([(s,t,d,np.sqrt(2)*ndtri((1+d)/2)) for s,t,d in itertools.product([1.,2.,4.],[0.,.02,.1],[.01,.03,.1])])
folds = [('Jul-Sep','2014-06-01','2014-07-01','2014-10-01'),('Oct-Dec','2014-09-01','2014-10-01','2015-01-01')]
comparisons, tuning, parameters, calibration, means, audit = [], [], [], [], [], []
for speed in ['Bullet','Blitz','Classical']:
    saved_parts = []
    for fold, valid_start, test_start, test_end in folds:
        tick = time.monotonic()
        vstart, tstart, tend = [int(datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp()) for x in [valid_start,test_start,test_end]]
        train = all_games.filter(pl.col('speed') == speed,pl.col('time') < tstart)
        valid = (train['time'].to_numpy() >= vstart) & (train['last_move_ms'].to_numpy() < tstart*1000)
        weight = train['weight'].to_numpy().astype(float)*valid
        stats, _ = likelihood_rs.trueskill_replay(*replay_arrays(train,tstart*1000),weight,configs,False)
        assert np.isfinite(stats).all() and (stats[:,1] == weight.sum()).all()
        chosen = int(np.argmin(stats[:,0]))
        for k, (s,t,d,e) in enumerate(configs):
            tuning.append(dict(speed=speed,fold=fold,candidate=k,sigma=s,tau=t,draw_probability=d,draw_margin=e,
                               validation_log_loss=stats[k,0],n=int(stats[k,1]),clipped=int(stats[k,2]),selected=k==chosen))
        config = configs[chosen:chosen+1].copy()
        g = all_games.filter(pl.col('speed') == speed,pl.col('time') < tend)
        stats, p = likelihood_rs.trueskill_replay(*replay_arrays(g,tend*1000),g['weight'].to_numpy().astype(float),config,True)
        assert np.allclose(p.sum(1),1,atol=1e-14) and (p > 0).all() and np.isfinite(p).all()
        predicted = g.select('game_id','white_id').with_columns(w_loss=pl.Series(p[:,0]),w_draw=pl.Series(p[:,1]),w_win=pl.Series(p[:,2]))
        d = (all_rows.filter(pl.col('speed') == speed,pl.col('time') >= tstart,pl.col('time') < tend)
             .join(predicted,on='game_id',how='left',maintain_order='left'))
        assert not d.select(pl.any_horizontal(pl.col('w_loss','w_draw','w_win').is_null()).any()).item()
        reverse = d['a_id'].to_numpy() != d['white_id'].to_numpy()
        p = d.select('w_loss','w_draw','w_win').to_numpy()
        p[reverse] = p[reverse,::-1]
        export = next(x for x in exports if x['speed'] == speed and x['fold'] == fold)
        P = {'trueskill':p}
        for name in ['baseline','recent']:
            model = export['models'][name if name == 'baseline' else export['aliases'][name]]
            P[name] = spline.probabilities(tm.offset(d,model['baseline_coef'],'Lichess'),model)
        y = d['y'].to_numpy()
        loss = {name:{**sa.losses(prob,y),'score_mse':(y/2-prob[:,2]-prob[:,1]/2)**2} for name,prob in P.items()}
        # 保存的參考模型須能重現原表，才可解讀新模型差值。
        published = pl.read_csv('outputs/tables/likelihood_selection_prediction.csv').filter(
            pl.col('speed') == speed,pl.col('fold') == fold,pl.col('group') == 'all',pl.col('model') == 'recent',pl.col('vs') == 'baseline')
        for row in published.iter_rows(named=True):
            assert np.isclose((loss['recent'][row['metric']]-loss['baseline'][row['metric']]).mean(),row['diff'],atol=1e-12)
        expected = pl.read_parquet(f'data/processed/hierarchical_{speed}_{fold}_slope_prediction.parquet').select('game_id','a_id')
        assert expected.equals(d.select('game_id','a_id'))
        saved = d.select('game_id','a_id','b_id','y','novice').with_columns(
            fold=pl.lit(fold),p_loss=pl.Series(p[:,0]),p_draw=pl.Series(p[:,1]),p_win=pl.Series(p[:,2]),
            **{f'{name}_{metric}':pl.Series(values) for name,metrics in loss.items() for metric,values in metrics.items()})
        saved.write_parquet(f'data/processed/trueskill_{speed}_{fold}_prediction.parquet')
        saved_parts.append(saved)
        groups = [('all',np.ones(len(d),bool)),('novice',d['novice'].to_numpy()),('established',~d['novice'].to_numpy())]
        calibration.extend(dict(speed=speed,fold=fold,**r) for r in calibration_rows(y,P,groups))
        for group, mask in groups:
            for reference in ['baseline','recent']:
                for metric in ['log_loss','score_mse','rps']:
                    delta = (loss['trueskill'][metric]-loss[reference][metric])[mask]
                    se = lk.dyad_se((delta-delta.mean())/len(delta),d['a_id'].to_numpy()[mask],d['b_id'].to_numpy()[mask])
                    comparisons.append(dict(speed=speed,fold=fold,group=group,vs=reference,metric=metric,n=len(delta),
                                            diff=delta.mean(),se=se,lo=delta.mean()-1.96*se,hi=delta.mean()+1.96*se,
                                            p=2*norm.sf(abs(delta.mean()/se))))
            for name, metrics in loss.items():
                for metric, values in metrics.items():
                    means.append(dict(speed=speed,fold=fold,group=group,model=name,metric=metric,n=int(mask.sum()),mean=values[mask].mean()))
        parameters.append(dict(speed=speed,fold=fold,sigma=config[0,0],tau=config[0,1],draw_probability=config[0,2],draw_margin=config[0,3],
                               beta=1.,mu=0.,validation_start=valid_start,test_start=test_start,test_end=test_end,validation_rows=int(weight.sum()),
                               replay_games=len(g),test_games=d['game_id'].n_unique(),test_rows=len(d),seconds=time.monotonic()-tick,clipped=int(stats[0,2])))
        audit.append(dict(speed=speed,fold=fold,rows_match=True,baseline_reproduced=True,probability_sum_error=np.abs(p.sum(1)-1).max(),min_probability=p.min()))
        print(speed,fold,'done','seconds',round(time.monotonic()-tick,2),'parameters',config[0,:3],flush=True)
    pooled = pl.concat(saved_parts)
    assert pooled.select(pl.struct('game_id','a_id').n_unique()).item() == len(pooled)
    for reference in ['baseline','recent']:
        for metric in ['log_loss','score_mse','rps']:
            delta = (pooled[f'trueskill_{metric}']-pooled[f'{reference}_{metric}']).to_numpy()
            se = lk.dyad_se((delta-delta.mean())/len(delta),pooled['a_id'].to_numpy(),pooled['b_id'].to_numpy())
            comparisons.append(dict(speed=speed,fold='pooled',group='all',vs=reference,metric=metric,n=len(delta),
                                    diff=delta.mean(),se=se,lo=delta.mean()-1.96*se,hi=delta.mean()+1.96*se,p=2*norm.sf(abs(delta.mean()/se))))
            for name in ['trueskill',reference]:
                means.append(dict(speed=speed,fold='pooled',group='all',model=name,metric=metric,n=len(delta),mean=pooled[f'{name}_{metric}'].mean()))
comparison = pl.DataFrame(comparisons)
main = comparison.filter(pl.col('fold') == 'pooled',pl.col('group') == 'all',pl.col('metric') == 'log_loss',pl.col('vs') == 'recent')
main = main.with_columns(p_holm=pl.Series(lk.holm(main['p'].to_numpy())))
comparison.write_csv('outputs/tables/trueskill_comparison.csv')
main.write_csv('outputs/tables/trueskill_decision.csv')
pl.DataFrame(tuning).write_csv('outputs/tables/trueskill_tuning.csv')
pl.DataFrame(parameters).write_csv('outputs/tables/trueskill_parameters.csv')
pl.DataFrame(means).unique().write_csv('outputs/tables/trueskill_means.csv')
pl.DataFrame(calibration).write_csv('outputs/tables/trueskill_calibration.csv')
pl.DataFrame(audit).write_csv('outputs/tables/trueskill_audit.csv')
sources = ['scripts/trueskill_model.py','scripts/trueskill_prediction.py','scripts/trueskill_prepare.py','rust/likelihood/src/trueskill.rs',
           'rust/likelihood/Cargo.lock','docs/trueskill-replication.md','data/processed/trueskill_games.parquet','outputs/tables/likelihood_selection_models.json']
Path('outputs/tables/trueskill_manifest.json').write_text(json.dumps(dict(
    backend='Rust one-versus-one Gaussian moment matching; libm erfc; Rayon independent candidates',reference='Python trueskill 0.4.5 scipy backend',
    events='created_ms predictions; last_move_ms updates; strict earlier timestamps',
    main_metric='three-class log_loss',hyperparameters=parameters,
    source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources}),ensure_ascii=False,indent=2)+'\n')
print(main,flush=True)
