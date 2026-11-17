# 同一個已配適模型的推論比較；只整理保存的係數、Hessian 與 dyadic standard error。
import polars as pl

coef = (pl.read_csv('outputs/tables/expected_score_baselines_coef.csv')
        .filter(pl.col('sample') == 'all', pl.col('model') == 'rd_ordinal', pl.col('param') == 'slope_g_diff')
        .with_columns(iid_lo=pl.col('est')-1.96*pl.col('se_iid'), iid_hi=pl.col('est')+1.96*pl.col('se_iid'),
                      dyadic_lo=pl.col('est')-1.96*pl.col('se_dyadic'), dyadic_hi=pl.col('est')+1.96*pl.col('se_dyadic')))
coef.write_csv('outputs/tables/dependence_report.csv')
tests = pl.read_csv('outputs/tables/expected_score_baselines_tests.csv').filter(pl.col('sample') == 'all')
assert tests.height == 6 and (tests['p_holm'] < .05).all()
assert coef.height == 3 and (coef['se_dyadic'] > coef['se_iid']).all()
print(coef.select('speed', 'est', 'se_iid', 'se_dyadic'))
