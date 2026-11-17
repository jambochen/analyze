# 核對正式輸出的樣本、loss、對稱性與兩段合併；不重新選擇模型。
import os
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import polars as pl
import likelihood as lk

comparison = pl.read_csv('outputs/tables/trueskill_comparison.csv')
rows=[]
for speed in ['Bullet','Blitz','Classical']:
    parts=[]
    for fold in ['Jul-Sep','Oct-Dec']:
        d=pl.read_parquet(f'data/processed/trueskill_{speed}_{fold}_prediction.parquet')
        original=pl.read_parquet(f'data/processed/hierarchical_{speed}_{fold}_slope_prediction.parquet')
        assert d.select('game_id','a_id','b_id','y').equals(original.select('game_id','a_id','b_id','y'))
        p=d.select('p_loss','p_draw','p_win').to_numpy();y=d['y'].to_numpy()
        assert np.allclose(p.sum(1),1,atol=1e-14) and (p>0).all()
        assert np.allclose(-np.log(p[np.arange(len(y)),y]),d['trueskill_log_loss'],atol=1e-14)
        assert np.allclose((y/2-p[:,2]-p[:,1]/2)**2,d['trueskill_score_mse'],atol=1e-14)
        mirror=d.join(d,on='game_id',suffix='_other').filter(pl.col('a_id')!=pl.col('a_id_other'))
        assert np.allclose(mirror['p_win'],mirror['p_loss_other'],atol=1e-14)
        assert np.allclose(mirror['p_draw'],mirror['p_draw_other'],atol=1e-14)
        for ref in ['baseline','recent']:
            for metric in ['log_loss','score_mse','rps']:
                assert np.allclose(d[f'{ref}_{metric}'],original[f'{ref}_{metric}'],atol=1e-13)
        parts.append(d)
        rows.append(dict(speed=speed,fold=fold,rows=len(d),games=d['game_id'].n_unique(),probabilities_valid=True,reference_reproduced=True))
    d=pl.concat(parts)
    for row in comparison.filter(pl.col('speed')==speed,pl.col('fold')=='pooled').iter_rows(named=True):
        delta=(d[f"trueskill_{row['metric']}"]-d[f"{row['vs']}_{row['metric']}"]).to_numpy()
        se=lk.dyad_se((delta-delta.mean())/len(delta),d['a_id'].to_numpy(),d['b_id'].to_numpy())
        assert np.isclose(delta.mean(),row['diff'],atol=1e-14)
        assert np.isclose(se,row['se'],atol=1e-14)
assert pl.read_csv('outputs/tables/trueskill_tuning.csv').height==6*27
assert pl.read_csv('outputs/tables/trueskill_parameters.csv')['clipped'].sum()==0
out=pl.DataFrame(rows)
out.write_csv('outputs/tables/trueskill_final_audit.csv')
print(out)
