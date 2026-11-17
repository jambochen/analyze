# 摘錄文獻 Table 2 的 Chess 列；這是引用資料，不是本專案的實驗輸出。
from pathlib import Path
import re
import polars as pl

source = Path('ref/competitive-games/tang.txt').read_text()
values = re.search(r'^\s*Chess\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+-\s*$', source, re.M).groups()
pl.DataFrame({
    'model': ['Elo', 'Glicko', 'TrueSkill', 'Elo2k'],
    'cross_entropy': [float(x) for x in values],
    'source': ['Tang et al. (2025), Table 2, Chess'] * 4,
    'url': ['https://arxiv.org/html/2502.10985v1#S3.T2'] * 4,
    'evidence': ['published_result_not_local_experiment'] * 4,
}).write_csv('outputs/tables/literature_rating_report.csv')
