"""Baseline: time-weighted Poisson attack/defence ratings, fit by penalised MLE.

Every team gets an attack and a defence parameter. Goals are Poisson with

    mu_home = exp(atk_home - def_away + adv * is_home_advantage + shared @ b_s
                  + home_only @ b_h)
    mu_away = exp(atk_away - def_home + shared @ b_s)

Matches are down-weighted exponentially by age, so the ratings track form.
The L2 penalty both identifies the model (attack/defence are otherwise only
identified up to a constant) and shrinks sparse teams toward the global mean,
which is the frequentist stand-in for partial pooling.

`shared` and `home_only` are optional match-level covariate matrices, used to
carry tournament effects (see tournament.py). `shared` shifts both teams' rates
-- the goal level of the match. `home_only` shifts the home rate alone, so it
interacts with home advantage; pass it already multiplied by the home flag if
that is what you mean, since the model adds it verbatim. Both are unpenalised,
like `adv`. With neither supplied the model is exactly the original one.

This is deliberately the simple model. The hierarchical Bayesian version has to
beat it on held-out log score or it is not worth the complexity. So does any
covariate added here.
"""

import numpy as np
from scipy.optimize import minimize


class PoissonStrength:
    def __init__(self, n_teams, half_life_days=730.0, ridge=0.05,
                 team_importance=False, ridge_imp=0.25):
        self.n = n_teams
        self.half_life = half_life_days
        self.ridge = ridge
        self.team_importance = team_importance
        self.ridge_imp = ridge_imp
        self.atk = np.zeros(n_teams)
        self.dfn = np.zeros(n_teams)
        self.imp = np.zeros(n_teams)
        self.adv = 0.0
        self.b_shared = np.zeros(0)
        self.b_home = np.zeros(0)

    def _weights(self, days_before):
        return 0.5 ** (days_before / self.half_life)

    @staticmethod
    def _as2d(z, n_rows):
        if z is None:
            return np.zeros((n_rows, 0))
        z = np.asarray(z, dtype=float)
        return z.reshape(-1, 1) if z.ndim == 1 else z

    def fit(self, h, a, gh, ga, home_adv_flag, days_before,
            shared=None, home_only=None, competitive=None):
        w = self._weights(days_before)
        n = self.n
        zs = self._as2d(shared, len(h))
        zh = self._as2d(home_only, len(h))
        ks, kh = zs.shape[1], zh.shape[1]
        ki = n if self.team_importance else 0
        comp = (np.zeros(len(h)) if competitive is None
                else np.asarray(competitive, dtype=float))

        def negll(theta):
            atk, dfn, adv = theta[:n], theta[n:2 * n], theta[2 * n]
            bs = theta[2 * n + 1:2 * n + 1 + ks]
            bh = theta[2 * n + 1 + ks:2 * n + 1 + ks + kh]
            imp = theta[2 * n + 1 + ks + kh:]
            common = zs @ bs
            # A team's importance effect lifts its attack and its defence
            # together, so it enters each rate as a difference of the two
            # teams' terms -- one parameter per team, not two.
            gap = (imp[h] - imp[a]) * comp if ki else 0.0
            lh = atk[h] - dfn[a] + adv * home_adv_flag + common + zh @ bh + gap
            la = atk[a] - dfn[h] + common - gap
            mh, ma = np.exp(lh), np.exp(la)
            ll = np.sum(w * (gh * lh - mh + ga * la - ma))
            pen = self.ridge * (atk @ atk + dfn @ dfn)
            g_atk = np.zeros(n)
            g_dfn = np.zeros(n)
            rh = w * (gh - mh)
            ra = w * (ga - ma)
            np.add.at(g_atk, h, rh)
            np.add.at(g_atk, a, ra)
            np.add.at(g_dfn, a, -rh)
            np.add.at(g_dfn, h, -ra)
            g_adv = np.sum(rh * home_adv_flag)
            parts = [g_atk - 2 * self.ridge * atk,
                     g_dfn - 2 * self.ridge * dfn,
                     [g_adv],
                     zs.T @ (rh + ra),
                     zh.T @ rh]
            if ki:
                g_imp = np.zeros(n)
                d = (rh - ra) * comp
                np.add.at(g_imp, h, d)
                np.add.at(g_imp, a, -d)
                pen = pen + self.ridge_imp * (imp @ imp)
                parts.append(g_imp - 2 * self.ridge_imp * imp)
            grad = np.concatenate(parts)
            return -(ll - pen), -grad

        theta0 = np.zeros(2 * n + 1 + ks + kh + ki)
        res = minimize(negll, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.atk, self.dfn, self.adv = res.x[:n], res.x[n:2 * n], res.x[2 * n]
        self.b_shared = res.x[2 * n + 1:2 * n + 1 + ks]
        self.b_home = res.x[2 * n + 1 + ks:2 * n + 1 + ks + kh]
        self.imp = (res.x[2 * n + 1 + ks + kh:] if ki else np.zeros(n))
        self.converged = res.success
        return self

    def rates(self, h, a, home_adv_flag, shared=None, home_only=None,
              competitive=None):
        zs = self._as2d(shared, len(h))
        zh = self._as2d(home_only, len(h))
        comp = (np.zeros(len(h)) if competitive is None
                else np.asarray(competitive, dtype=float))
        common = zs @ self.b_shared
        gap = (self.imp[h] - self.imp[a]) * comp
        mh = np.exp(self.atk[h] - self.dfn[a] + self.adv * home_adv_flag
                    + common + zh @ self.b_home + gap)
        ma = np.exp(self.atk[a] - self.dfn[h] + common - gap)
        return mh, ma

    def strength(self, i, competitive=True):
        """Expected goal difference against an average team on neutral ground.

        With `team_importance` on, the reference opponent has an importance
        effect of zero (the ridge centres them there), so the team's own term
        applies in full and `competitive=True` reads as competitive strength --
        which is the quantity this project is actually about. Without it the
        term is zero and this is the original definition.
        """
        e = self.imp[i] if competitive else 0.0
        return np.exp(self.atk[i] + e) - np.exp(-(self.dfn[i] + e))
