# 月檔可能含跨月對局；以開局時間選取，沿用報告的評估紀錄。
import polars as pl

g = (pl.scan_parquet('data/processed/lichess_games.parquet')
     .filter(pl.col('start') >= pl.datetime(2014, 1, 1, time_zone='UTC'),
             pl.col('start') < pl.datetime(2015, 1, 1, time_zone='UTC'),
             pl.col('tournament_system') == 'arena', pl.col('speed').is_in(['Bullet', 'Blitz', 'Classical']))
     .select('game_id', 'speed', 'white_id', 'black_id', time=pl.col('start').dt.epoch('s'),
             y=pl.col('result').replace_strict({'0-1': 0, '1/2-1/2': 1, '1-0': 2}, return_dtype=pl.Int64))
     .join(pl.scan_parquet('data/processed/game_meta.parquet').select('game_id', 'created_ms', 'last_move_ms'), on='game_id')
     .collect().sort('created_ms', 'game_id'))
t = pl.read_parquet('data/processed/hierarchical_prediction.parquet').select('game_id', 'a_id', 'b_id', 'y', 'time', 'speed')
assert not g.select(pl.any_horizontal(pl.all().is_null()).any()).item()
assert g['game_id'].n_unique() == len(g)
assert (g['last_move_ms'] > g['created_ms']).all()
check = t.join(g, on='game_id', suffix='_white')
assert len(check) == len(t)
assert (check['time'] == check['time_white']).all()
assert (check['speed'] == check['speed_white']).all()
assert check.select((pl.col('y') == pl.when(pl.col('a_id') == pl.col('white_id')).then(pl.col('y_white')).otherwise(2-pl.col('y_white'))).all()).item()
weights = t.group_by('game_id').agg(weight=pl.len())
g = g.join(weights, on='game_id', how='left').with_columns(pl.col('weight').fill_null(0))
g.write_parquet('data/processed/trueskill_games.parquet')
g.group_by('speed').agg(games=pl.len(), eval_rows=pl.col('weight').sum(),
                        max_start_difference_ms=(pl.col('time')*1000-pl.col('created_ms')).abs().max()).write_csv('outputs/tables/trueskill_sample.csv')
print(pl.read_csv('outputs/tables/trueskill_sample.csv'))
