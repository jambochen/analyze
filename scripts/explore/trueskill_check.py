# 數值、時間資訊與耗時核對；不讀取正式測試段的預測結果。
import sys
sys.path.insert(0, 'scripts')
import time
import numpy as np
import polars as pl
from scipy.special import ndtri
import likelihood_rs
from trueskill_model import replay_arrays, reference

g = pl.read_parquet('data/processed/trueskill_games.parquet').filter(pl.col('speed') == 'Blitz').head(2000)
a, b, y, events = replay_arrays(g, int(g['last_move_ms'].max())+1)
rows = []
for sigma, tau, draw in [(2., .02, .1), (1., 0., .01), (4., .1, .03)]:
    config = np.array([[sigma, tau, draw, np.sqrt(2)*ndtri((1+draw)/2)]])
    tick = time.monotonic()
    p_ref = reference(a,b,y,events,config[0])
    python_seconds = time.monotonic()-tick
    tick = time.monotonic()
    stat, p = likelihood_rs.trueskill_replay(a,b,y,events,np.ones(len(g)),config,True)
    rust_seconds = time.monotonic()-tick
    error = np.abs(p-p_ref).max()
    assert error < 1e-10, error
    assert np.allclose(p.sum(1),1,atol=1e-14)
    assert (p > 0).all() and np.isfinite(p).all()
    _, rev = likelihood_rs.trueskill_replay(b,a,2-y,events,np.ones(len(g)),config,True)
    assert np.allclose(p,rev[:,::-1],atol=1e-12)
    cut = int(g['created_ms'][1000])
    ca, cb, cy, cevents = replay_arrays(g,cut)
    _, partial = likelihood_rs.trueskill_replay(ca,cb,cy,cevents,np.ones(len(g)),config,True)
    earlier = g['created_ms'].to_numpy() < cut
    assert np.array_equal(p[earlier],partial[earlier])
    future_y = y.copy()
    future_y[g['last_move_ms'].to_numpy() >= cut] = 2-future_y[g['last_move_ms'].to_numpy() >= cut]
    _, changed = likelihood_rs.trueskill_replay(a,b,future_y,events,np.ones(len(g)),config,True)
    assert np.array_equal(p[earlier],changed[earlier])
    rows.append(dict(sigma=sigma,tau=tau,draw=draw,games=len(g),max_probability_error=error,
                     python_seconds=python_seconds,rust_seconds=rust_seconds,clipped=stat[0,2]))
# 少數玩家反覆交手，含高機率的勝局、罕見逆轉與和局。
a = np.zeros(500, dtype=np.int64); b = np.ones(500, dtype=np.int64)
y = np.full(500,2,dtype=np.int64); y[::31]=1; y[::53]=0
events = np.column_stack((np.arange(1,501),-np.arange(1,501))).ravel()
_, p = likelihood_rs.trueskill_replay(a,b,y,events,np.ones(500),config,True)
p_ref = reference(a,b,y,events,config[0])
assert np.max(np.abs(p-p_ref)) < 1e-10
pl.DataFrame(rows).write_csv('outputs/tables/trueskill_check.csv')
print(pl.DataFrame(rows),flush=True)
# 以完整的第一段訓練序列量測單一候選，供正式 grid 耗時估計。
g = pl.read_parquet('data/processed/trueskill_games.parquet').filter(pl.col('speed') == 'Blitz',pl.col('time') < 1404172800)
args = replay_arrays(g,1404172800000)
tick = time.monotonic()
stat, _ = likelihood_rs.trueskill_replay(*args,np.ones(len(g)),config,False)
seconds = time.monotonic()-tick
pl.DataFrame([{'speed':'Blitz','games':len(g),'seconds':seconds,'candidates':27,'serial_grid_seconds':27*seconds}]).write_csv('outputs/tables/trueskill_benchmark.csv')
print('full_training_seconds',seconds,'serial_grid_seconds',27*seconds,flush=True)
