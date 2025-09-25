from typing import NamedTuple

import jax
import jax.lax.linalg as lax_linalg
import jax.numpy as jnp
import jax.scipy.linalg as scipy_linalg


class BLR(NamedTuple):
    eta: jax.Array  # natural parameter: precision @ mean
    R: jax.Array  # upper-triangular Cholesky factor of precision
    lam: float | None = None

    @classmethod
    def create(cls, mu, cov):
        assert mu.ndim == 1
        R = scipy_linalg.cholesky(jnp.linalg.inv(cov))
        eta = R.T @ (R @ mu)
        return cls(eta=eta, R=R)

    @classmethod
    def create_LSQ(cls, dim: int, lam: float = 1.0):
        R = jnp.sqrt(lam) * jnp.eye(dim)
        eta = jnp.zeros((dim,))
        return cls(eta=eta, R=R, lam=jnp.float32(lam))

    def rank1_update(self, x: jax.Array, y: jax.Array, sigma: float = 1.0, decay: float = 1.0) -> 'BLR':
        """Update with a new observation f(x) = y + N(0, sigma^2) (optionally decay old ones)"""
        # NOTE: this does not quite reflect D-LinUCB: we also decay the prior here!
        x, y = x / sigma, y / sigma
        R_ = lax_linalg.cholesky_update(self.R * jnp.sqrt(decay), x)
        eta_ = self.eta * decay + (x * y)
        return self._replace(eta=eta_, R=R_)

    def update(self, X: jax.Array, y: jax.Array, sigma: float = 1.0) -> 'BLR':
        """Update with a batch of observation f(x_i) = y_i + N(0, sigma^2)
        X is (batch, dim), y (batch,)
        """
        assert X.ndim == 2 and X.shape == (y.shape[0], self.dim)
        if X.shape[0] == 0:
            return self
        X, y = X / sigma, y / sigma
        R_ = scipy_linalg.cholesky(self.prec + X.T @ X)
        eta_ = self.eta + jnp.vecmat(y, X)
        return self._replace(R=R_, eta=eta_)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean mu"""
        return scipy_linalg.cho_solve((self.R, False), self.eta)  # P^-1 @ eta = mean

    def sample(self, key, sample_shape=()) -> jnp.ndarray:
        """Sample x~N(mu, cov)"""
        return self.transform(jax.random.normal(key, sample_shape + self.eta.shape))

    def sample_ellipsoid(self, key, sample_shape=(), beta: float = 1.0):
        """Sample uniformly from the confidence set C = {x: ||x - mu||^2_{cov^-1} <= beta}"""
        eps = jax.random.ball(key, self.dim, shape=sample_shape)
        return self.transform(eps * beta)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """N(x | mu, cov)"""
        assert x.shape[-1] == self.dim
        # log N(x|μ,Σ) = -1/2 [ d log(2π) + log det Σ + (x-μ)^T Σ^{-1} (x-μ) ]
        log_const = self.dim * jnp.log(2 * jnp.pi) + self.cov_log_det
        y = self.inv_transform(x)
        return -0.5 * (log_const + jnp.linalg.vecdot(y, y))

    def ucb(self, x, beta: float):
        """
        returns max_{theta in C} <x, theta> where C = {x : ||x - mean||^2_{prec} <= beta}
        """
        # <x, mean> + beta * sqrt(x.T @ cov @ x)
        return jnp.linalg.vecdot(x, self.mean) + beta * self.cov_norm(x)

    @property
    def prec_log_det(self) -> jax.Array:
        """log det(prec)"""
        return 2 * jnp.sum(jnp.log(jnp.diag(self.R)))

    @property
    def true_prec_log_det(self) -> jax.Array:
        """log det(prec) - log det (lam * I)"""
        assert self.lam is not None, 'only available for LSQ'
        return jnp.maximum(jnp.finfo(self.R).eps, self.prec_log_det - self.dim * jnp.log(self.lam))

    @property
    def cov_log_det(self) -> jax.Array:
        """log det(cov)"""
        return -self.prec_log_det

    @property
    def prec(self) -> jax.Array:
        """Precision matrix."""
        return self.R.T @ self.R

    @property
    def cov(self) -> jax.Array:
        """Covariance matrix."""
        # cov = (R.T @ R)^-1 = R^-1 @ R^-T
        X = self.half_cov_mul(jnp.eye(self.dim))
        return X @ X.T

    def cov_norm(self, x: jax.Array):
        """sqrt(x.T @ cov @ x)"""
        return jnp.linalg.vector_norm(self.half_cov_mul(x), axis=-1)

    def prec_norm(self, x: jax.Array):
        """sqrt(x.T @ P @ x)"""
        return jnp.linalg.vector_norm(self.half_prec_mul(x), axis=-1)

    def get_transformed_components(self, A: jax.Array, b: jax.Array = 0.0) -> tuple[jax.Array, jax.Array]:
        """
        in: A, b of shapes [n, dim], [n]
        out: mu = A @ mu + b and cov = A @ cov @ A.T
        """
        assert A.ndim == 2 and A.shape[1] == self.eta.shape[0]
        mu = A @ self.mean + b
        X = self.half_cov_mul(A)
        cov = X @ X.T
        return mu, cov

    def get_transformed(self, A: jax.Array, b: jax.Array = 0.0) -> 'BLR':
        """
        warning: slow O(d^3)
        in: A, b of shapes [n, dim], [n]
        out: new BLR(mu = A @ mu + b, cov = A @ cov @ A.T)
        """
        return BLR.create(*self.get_transformed_components(A, b))

    def transform(self, x: jax.Array):
        """
        in: batch of vectors x [*batch_shape, dim]
        out: y = mu + cov^1/2 @ x of same shape
        """
        return self.mean + self.half_cov_mul(x)

    def inv_transform(self, x: jax.Array):
        """
        in: batch of vectors x [*batch_shape, dim]
        out: y = cov^{-1/2} @ (x - mu) of same shape
        """
        return self.half_prec_mul(x - self.mean)

    def half_cov_mul(self, x: jax.Array):
        """
        in: batch of vectors x [*batch_shape, dim]
        out: cov^{1/2} @ x of shape [*batch_shape, dim]
        """
        # y.T @ y = x.T @ inv(R.T @ R) @ x = x.T @ inv(R) @ inv(R).T @ x
        # ==> y = inv(R).T @ x ==> y.T = x.T @ inv(R) ==> y.T @ R = x.T
        assert x.shape[-1] == self.dim
        xT = x.reshape(-1, self.dim)
        yT = lax_linalg.triangular_solve(a=self.R, b=xT, left_side=False, lower=False, transpose_a=False)
        return yT.reshape(x.shape)

    def half_prec_mul(self, x: jax.Array):
        """
        in: batch of vectors x [*batch_shape, dim]
        out: cov^{-1/2} @ x of shape [*batch_shape, dim]
        """
        # y.T @ y = x.T @ R.T @ R @ x = (R @ x).T (R @ x)
        x_ = x.reshape(-1, self.dim)
        y = jnp.einsum('xy, by -> bx', self.R, x_)
        return y.reshape(x.shape)

    @property
    def cov_max_upper_bound(self):
        return 1.0 / jnp.diag(self.R).min()

    @property
    def prec_max_upper_bound(self):
        return jnp.diag(self.R).max()

    @property
    def dim(self):
        return self.eta.shape[-1]


if __name__ == '__main__':
    # Lightweight tests to validate the BLR implementation

    key = jax.random.key(0)

    def assert_allclose(a, b, tol=1e-6, msg=''):
        ok = jnp.allclose(a, b, atol=tol, rtol=tol)
        if not bool(ok):
            raise AssertionError(f'Assertion failed: {msg} | max abs diff = {jnp.max(jnp.abs(a - b))}')

    def test_create_and_basic_properties():
        mu = jnp.array([0.5, -1.0, 2.0])
        A = jnp.array([[2.0, -1.0, 0.0], [0.0, 1.5, 0.0], [0.5, 0.2, 1.0]])
        cov = A @ A.T + jnp.eye(3) * 0.5  # SPD
        C = BLR.create(mu, cov)

        # mean and cov should match
        assert_allclose(C.mean, mu, tol=1e-6, msg='mean')
        assert_allclose(C.cov, cov, tol=1e-6, msg='cov')

        # precision and its Cholesky should be consistent
        P = jnp.linalg.inv(cov)
        print('cov condition number:', jnp.linalg.cond(cov))
        assert_allclose(C.prec, P, tol=1e-6, msg='precision')  # TODO: fails
        # determinant relations
        assert_allclose(jnp.linalg.slogdet(C.prec)[1], C.prec_log_det, tol=1e-6, msg='prec log det')
        assert_allclose(jnp.linalg.slogdet(C.cov)[1], C.cov_log_det, tol=1e-6, msg='cov log det')

        # half muls should agree with norms
        x = jnp.array([1.0, 2.0, -3.0])
        cn = C.cov_norm(x)
        pn = C.prec_norm(x)
        assert_allclose(cn**2, x @ C.cov @ x, tol=1e-6, msg='cov norm')  # TODO: fails
        assert_allclose(pn**2, x @ C.prec @ x, tol=1e-6, msg='prec norm')

    def test_create_LSQ_and_rank1_update():
        d = 4
        mu = jnp.arange(d, dtype=jnp.float32) / 10.0
        lam = 2.5
        C = BLR.create_LSQ(mu, lam=lam)
        # R should be diagonal sqrt(lam)
        assert_allclose(C.R, jnp.eye(d) * jnp.sqrt(lam), tol=1e-8, msg='LSQ R diag')
        assert_allclose(C.prec, lam * jnp.eye(d), tol=1e-6, msg='LSQ precision')
        assert_allclose(C.mean, mu, tol=1e-6, msg='LSQ mean')

        # Single update should match analytic rank-1 update
        x = jnp.array([0.1, -0.2, 0.3, -0.4])
        y = 0.75
        C2 = C.rank1_update(x, y)
        assert_allclose(C2.prec, C.prec + jnp.outer(x, x), tol=1e-6, msg='rank1 precision update')
        assert_allclose(C2.eta, C.eta + x * y, tol=1e-6, msg='rank1 eta update')

    def test_log_prob_matches_closed_form():
        mu = jnp.array([0.2, -0.1])
        cov = jnp.array([[1.2, 0.3], [0.3, 0.9]])
        C = BLR.create(mu, cov)
        x = jnp.array([0.3, -0.4])
        xc = x - mu
        quad = xc @ jnp.linalg.inv(cov) @ xc
        target = -0.5 * (2 * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(cov)[1] + quad)
        got = C.log_prob(x)
        assert_allclose(got, target, tol=1e-6, msg='log_prob')  # TODO: fails

    def test_transform_and_sampling_shapes():
        mu = jnp.zeros((3,))
        cov = jnp.diag(jnp.array([0.5, 1.0, 2.0]))
        C = BLR.create(mu, cov)

        # Transform standard normals -> samples from N(mu, cov)
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (5, 3))
        s = C.transform(z)
        assert s.shape == (5, 3)
        s2 = C.sample(k2, sample_shape=(7,))
        assert s2.shape == (7, 3)

    def test_confidence_bound_and_ucb():
        d = 3
        mu = jnp.zeros((d,))
        lam = 1e-2
        C = BLR.create_LSQ(mu, lam=lam)
        beta = C.confidence_bound(S=1.0, delta=1e-3)
        assert bool(jnp.isfinite(beta))
        x = jnp.array([1.0, 0.0, 0.0])
        ucb = C.ucb(x, beta=beta)
        # Should be mean dot x + beta * ||x||_{cov}
        cn = C.cov_norm(x)
        assert_allclose(ucb, jnp.dot(C.mean, x) + beta * cn, tol=1e-6, msg='ucb')

    # Run tests
    test_create_and_basic_properties()
    print('basic properties passed')
    test_create_LSQ_and_rank1_update()
    print('create LSQ and rank1 update passed')
    test_log_prob_matches_closed_form()
    print('transforms and sampling passed')
    test_transform_and_sampling_shapes()
    print('confidence bound and ucb passed')
    test_confidence_bound_and_ucb()

    print('All BLR tests passed.')
