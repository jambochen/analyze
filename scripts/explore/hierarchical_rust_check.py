import sys
sys.path.insert(0, 'scripts')
import numpy as np
import polars as pl
import hierarchical as hr
import hierarchical_rust as rr

rng = np.random.default_rng(191)
n, players = 1200, 37
X = rng.normal(size=(n, 10))
Z = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
a, b = rng.integers(players, size=(2, n))
ha, hb = rng.uniform(-1, 1, size=(2, n))
y = rng.integers(3, size=n)
rows = []
for slope in [False, True]:
    model = hr.ordinal_model(X, Z, a, b, ha, hb, y, slope)
    lp = model.compile_logp()
    dg = model.compile_dlogp()
    rust = rr.compiled_model(X, Z, a, b, ha, hb, y, slope)
    serial = rr.compiled_model(X, Z, a, b, ha, hb, y, slope, blocks=1)
    swapped = rr.compiled_model(-X, Z, b, a, hb, ha, 2-y, slope)
    names = [v.name for v in model.value_vars]
    assert names == ['beta', 'width_beta', 'sd_log__', 'z']+(['rho_unit_logodds__'] if slope else [])
    point = model.initial_point()
    sizes = [point[name].size for name in names]
    rust_shapes = {v.name: tuple(v.shape) for v in rust._variables}
    expand_reference = model.compile_fn(model.replace_rvs_by_values([model[name] for name in rust_shapes]),
                                        inputs=model.value_vars, on_unused_input='ignore')
    # PyTensor 對固定 prior scale 的 log 常數可能以 float32 摺疊；核對參數改變後的 logp 差及全部梯度。
    normalization_offset = None
    for case in range(10):
        theta = rust._make_initial_points(case+98)
        if case >= 6:
            theta[10] = [-15., 4., -8., 1.][case-6]
            if slope:
                theta[-1] = [-12., 12., -5., 5.][case-6]
        parts = np.split(theta, np.cumsum(sizes)[:-1])
        values = {name: part.reshape(point[name].shape) for name, part in zip(names, parts)}
        expanded = rust._make_expand_func(1, 2, 0)(theta)
        assert list(expanded) == list(rust_shapes)
        for name, expected in zip(expanded, expand_reference(values)):
            assert expanded[name].shape == rust_shapes[name]
            np.testing.assert_allclose(expanded[name], expected, rtol=2e-10, atol=2e-10)
        ref, grad = lp(values), dg(values)
        actual, actual_grad = rust._make_logp_func()(theta)
        one, one_grad = serial._make_logp_func()(theta)
        reverse, reverse_grad = swapped._make_logp_func()(theta)
        if case == 0:
            normalization_offset = actual-ref
            assert abs(normalization_offset) < 1e-6
        np.testing.assert_allclose(actual-ref, normalization_offset, rtol=0, atol=2e-8)
        np.testing.assert_allclose(actual_grad, grad, rtol=2e-10, atol=2e-8)
        np.testing.assert_allclose([one, reverse], actual, rtol=2e-12, atol=2e-8)
        np.testing.assert_allclose([one_grad, reverse_grad], [actual_grad]*2, rtol=2e-10, atol=2e-8)
        rows.append({'slope': slope, 'case': case, 'logp_error': abs(actual-ref),
                     'centered_logp_error': abs(actual-ref-normalization_offset),
                     'gradient_max_error': np.max(np.abs(actual_grad-grad)),
                     'swap_gradient_error': np.max(np.abs(reverse_grad-actual_grad))})
pl.DataFrame(rows).write_csv('outputs/tables/hierarchical_rust_check.csv')
print(pl.DataFrame(rows))
