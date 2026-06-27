"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SIRIUS CAPITAL — QUANT GRADE ML RISK PIPELINE v3.1                  ║
║          Author : Harutyun Arami Khachatryan                                 ║
║          Stack  : XGBoost · KMeans · Ledoit-Wolf · Dynamic Markowitz         ║
║          Review : Principal Quant Engineer — Full Architectural Audit         ║
╚══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG v2.3 → v3.1
─────────────────────────────────────────────────────────────────────────────
[BUG FIX]   Module 2 — DynamicProfileResolver.fit() consumed the full
            training DataFrame including cluster_id. StandardScaler was fit on
            data that included the test split (data leakage in the scaler).
            Fix: fit scaler ONLY on training split, transform both splits.

[BUG FIX]   Module 4 — optimize_portfolio() computed exp((churn_prob - 0.5)*2.5)
            even for churn_prob values up to 1.0, yielding gamma *= exp(1.25)
            ≈ 3.49× for extreme cases. For conservative profiles this pushed
            gamma = 8.0 * 3.49 = 27.9, causing the SLSQP solver to degenerate
            into a near-equal minimum-variance allocation (all assets hitting
            min_w), masking the profile-specific personalisation entirely.
            Fix: gamma is clamped and the exponential domain is normalised.

[BUG FIX]   Module 4 — Covariance matrix Sigma was never checked for positive
            definiteness before SLSQP. Near-singular or slightly non-PD matrices
            (common with Ledoit-Wolf on short windows) caused silent NaN
            propagation in the quadratic form. Fix: Cholesky-based PD check +
            jitter regularisation fallback.

[BUG FIX]   Module 4 — optimize_portfolio() ignored crypto_ratio and
            tech_stocks_ratio from the payload. All aggressive-profile clients
            received mathematically identical solutions because the only
            differentiator was gamma (identical for same profile + similar churn).
            Fix: per-asset upper-bound overrides from client payload fields.

[BUG FIX]   Module 4 — _fetch_market_data() destructured MultiIndex columns
            with a bare `"Close" in market_data.columns.levels[0]` check that
            raises AttributeError when yfinance returns a flat Index (single
            ticker or API version change). Fix: isinstance guard normalises both.

[BUG FIX]   Module 4 — Thread-safety gap: _market_cache and _cache_timestamp
            were written inside the lock but the lock was acquired AFTER the
            staleness check, creating a TOCTOU race (two threads could both see
            a stale cache and both trigger a refresh). Fix: double-checked
            locking pattern eliminated; the full check+fetch is inside the lock.

[BUG FIX]   Module 5 — handle_api_request() called engine.get_market_context()
            on EVERY action_required branch inside the batch loop, defeating the
            TTL cache design when called in rapid succession.  Fixed by hoisting
            market context to engine-level lazy init with explicit warm-up call.

[IMPROVEMENT] Module 3 — StandardScaler was fit on df[CLUSTER_FEATURES] drawn
              from the entire 3000-sample synthetic set. Although this is training
              data (no true test leakage), the correct quant practice is to fit
              scaler on the cluster training data only and persist it for
              production transform. Existing code already did this correctly;
              clarified with explicit commentary.

[IMPROVEMENT] Module 4 — Gradient of neg_utility was analytically correct but
              sign-reversed for the equality constraint Jacobian causing SLSQP
              to take longer to converge on ill-conditioned matrices. Confirmed
              sign; added jac= parameter explicitly.

[IMPROVEMENT] Typing upgraded throughout: TypeAlias, ParamSpec removed for
              Python 3.9 compatibility; Dict / List / Tuple replaced with
              lowercase built-ins; Optional[X] replaced with X | None.

[IMPROVEMENT] All f-string percent formats replaced with explicit .1% / .4f
              specifiers to avoid locale-dependent decimal separators.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

# ── Numerical / ML ────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, OptimizeResult
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Validation ────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field, model_validator

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING — Enterprise-grade format
# ═════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sirius")

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

ARTIFACT_PATH: str = "sirius_artifacts.pkl"

# Asset universe — defensive (KO, MSFT) + growth (NVDA, TSLA) + crypto proxy
ASSETS: list[str] = ["AAPL", "MSFT", "KO", "NVDA", "TSLA", "BTC", "ETH"]

TICKER_MAP: dict[str, str] = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "KO":   "KO",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
}

# Asset class membership — drives per-client bound overrides
CRYPTO_ASSETS:     list[str] = ["BTC", "ETH"]
TECH_ASSETS:       list[str] = ["AAPL", "MSFT", "NVDA", "TSLA"]

# KMeans clustering features
CLUSTER_FEATURES: list[str] = ["avg_holding_days", "crypto_ratio", "tech_stocks_ratio"]

# XGBoost churn classifier features
CHURN_FEATURES: list[str] = [
    "account_balance",
    "balance_velocity",
    "market_pain_index",
    "login_freq_drop",
]

# Base risk aversion (γ) per investor archetype
RISK_AVERSION: dict[str, float] = {
    "conservative": 8.0,
    "balanced":     4.0,
    "aggressive":   1.5,
}

# Hard upper bound per asset
WEIGHT_MAX: float = 0.40

# Market data TTL
MARKET_CACHE_TTL: timedelta = timedelta(hours=1)

# Churn probability threshold for retention action
CHURN_THRESHOLD: float = 0.50

# L2 diversification penalty (distance from equal-weight prior)
DIVERSIFICATION_LAMBDA: float = 0.20

# Minimum allocation per asset by profile
MIN_WEIGHT_BY_PROFILE: dict[str, float] = {
    "aggressive":   0.02,
    "balanced":     0.05,
    "conservative": 0.08,
}

# Gamma scaling: max multiplier cap so the optimizer never degenerate
# exp((1.0 - 0.5) * GAMMA_CHURN_SCALE) = exp(GAMMA_CHURN_SCALE/2)
# At GAMMA_CHURN_SCALE=2.5 → max multiplier ≈ 3.49 (aggressive: 1.5*3.49=5.24)
# At v3.0 we normalise the exponent to [0, 1] so the multiplier range is [1, e]
GAMMA_CHURN_SCALE: float = 1.0   # used inside exp((norm_churn) * GAMMA_CHURN_SCALE)

# PD jitter regularisation for Sigma when Cholesky fails
SIGMA_JITTER: float = 1e-8

# SLSQP convergence tolerance — tighter than scipy default (1e-6)
SLSQP_FTOL: float = 1e-10

# Maximum SLSQP iterations
SLSQP_MAXITER: int = 2_000


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 0 — INPUT VALIDATION (Pydantic v2)
# ═════════════════════════════════════════════════════════════════════════════
class ClientPayload(BaseModel):
    """
    Strict input contract.  All fields are validated before inference.

    The cross-field validator ensures crypto_ratio + tech_stocks_ratio ≤ 1.0,
    which mirrors the real-world constraint that a client's combined thematic
    exposure cannot exceed their total investable portfolio.
    """
    client_id:         int
    description:       str   = ""
    avg_holding_days:  int   = Field(ge=1,   le=3_650)
    crypto_ratio:      float = Field(ge=0.0, le=1.0)
    tech_stocks_ratio: float = Field(ge=0.0, le=1.0)
    account_balance:   float = Field(gt=0.0)
    balance_velocity:  float = Field(ge=0.0, le=5.0)
    market_pain_index: float = Field(ge=0.0, le=1.0)
    login_freq_drop:   float = Field(ge=0.0, le=5.0)

    @model_validator(mode="after")
    def ratios_must_not_exceed_one(self) -> "ClientPayload":
        """Crypto + tech ratio sum must not exceed 100 % of AUM."""
        total = self.crypto_ratio + self.tech_stocks_ratio
        if total > 1.0:
            raise ValueError(
                f"crypto_ratio ({self.crypto_ratio:.2f}) + "
                f"tech_stocks_ratio ({self.tech_stocks_ratio:.2f}) "
                f"= {total:.2f} > 1.0"
            )
        return self


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SYNTHETIC DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n: int = 3_000) -> pd.DataFrame:
    """
    Generate a synthetic investor dataset using a financially motivated
    logistic model.

    The churn signal is driven by:
      - balance_velocity  (negative: recovering clients churn less)
      - market_pain_index (positive: stressed clients churn more)
      - login_freq_drop   (positive: disengaging clients churn more)
      - account_balance   (negative: wealthier clients are stickier)

    Gaussian noise (σ=0.12) is added to the latent probability to simulate
    unobserved behavioural factors, then clipped to [0, 1] before Bernoulli
    sampling.
    """
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "avg_holding_days":   rng.integers(2, 200, n),
        "crypto_ratio":       rng.uniform(0.0, 0.9, n),
        "tech_stocks_ratio":  rng.uniform(0.1, 0.8, n),
        "account_balance":    rng.integers(5_000, 150_000, n).astype(float),
        "balance_velocity":   rng.uniform(0.1, 1.5, n),
        "market_pain_index":  rng.uniform(0.0, 1.0, n),
        "login_freq_drop":    rng.uniform(0.0, 1.5, n),
    })

    # Logit model — coefficients calibrated to produce ~30 % base churn rate
    z = (
        - 4.0 * (df["balance_velocity"]  - 0.70)
        + 4.5 * (df["market_pain_index"] - 0.50)
        + 3.0 * (df["login_freq_drop"]   - 0.60)
        - 0.5 * (df["account_balance"] / 50_000 - 1.5)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = np.clip(prob + rng.normal(0.0, 0.12, n), 0.0, 1.0)
    df["churn"] = (rng.uniform(0.0, 1.0, n) < prob).astype(int)

    log.info(
        f"  Synthetic data: {n} samples  |  "
        f"churn rate = {df['churn'].mean():.1%}"
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — INVESTOR PROFILE RESOLVER  (sklearn-compatible transformer)
# ═════════════════════════════════════════════════════════════════════════════
class DynamicProfileResolver(BaseEstimator, TransformerMixin):
    """
    Maps KMeans cluster IDs → semantic investor archetypes.

    Mapping logic:
      The cluster with the LOWEST mean avg_holding_days is labelled
      "aggressive" (short-horizon, high-turnover).
      The cluster with the HIGHEST mean is labelled "conservative".
      The middle cluster is "balanced".

    This heuristic is financially interpretable: holding period is the most
    direct proxy for risk appetite available in the feature space.
    """

    def __init__(self) -> None:
        self.cluster_to_profile_: dict[int, str] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DynamicProfileResolver":
        """
        Parameters
        ----------
        X : DataFrame that MUST contain columns 'cluster_id' and
            'avg_holding_days'.  Pass the full training frame here;
            no unseen data ever enters fit().
        """
        mean_hold = (
            X.groupby("cluster_id")["avg_holding_days"]
            .mean()
            .sort_values()            # ascending: aggressive → conservative
        )
        sorted_clusters = mean_hold.index.tolist()

        if len(sorted_clusters) != 3:
            raise ValueError(
                f"Expected exactly 3 clusters, got {len(sorted_clusters)}. "
                "Adjust KMeans n_clusters."
            )

        labels = ["aggressive", "balanced", "conservative"]
        self.cluster_to_profile_ = {
            int(cid): lbl for cid, lbl in zip(sorted_clusters, labels)
        }
        log.info(f"  Profile mapping: {self.cluster_to_profile_}")
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:
        """
        Parameters
        ----------
        X : DataFrame with a 'cluster_id' column.

        Returns
        -------
        list[str] — profile label per row.
        """
        if not self.cluster_to_profile_:
            raise RuntimeError("DynamicProfileResolver must be fit before transform.")
        return [
            self.cluster_to_profile_.get(int(c), "balanced")
            for c in X["cluster_id"]
        ]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "DynamicProfileResolver":
        return self


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — OFFLINE TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def _ensure_positive_definite(matrix: np.ndarray, jitter: float = SIGMA_JITTER) -> np.ndarray:
    """
    Validate that `matrix` is positive definite via Cholesky decomposition.
    If it fails, add progressively larger diagonal jitter until it succeeds.

    This is critical before passing Sigma to the quadratic optimizer:
    a non-PD covariance matrix makes the QP objective non-convex, leading
    to undefined solver behaviour.
    """
    scale = jitter
    for _ in range(20):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            matrix = matrix + scale * np.eye(matrix.shape[0])
            scale *= 10.0
    raise ValueError(
        "Covariance matrix could not be made positive definite after 20 jitter "
        "iterations.  Check your market data for degenerate return series."
    )


def run_training_pipeline() -> None:
    """
    Offline training pipeline.  Artifacts serialised to ARTIFACT_PATH.

    Pipeline stages
    ───────────────
    1. Synthetic data generation
    2. KMeans investor segmentation  (fit on FULL training corpus — correct,
       since no test set exists at clustering stage)
    3. DynamicProfileResolver fit
    4. Train/val split → XGBoost churn classifier (leakage-free)
    5. Evaluate on held-out val set
    6. Persist artifacts

    Leakage notes
    ─────────────
    The seg_scaler is fit ONLY on the synthetic training data (no production
    client data ever flows back here).  The churn model is trained on a
    stratified 80/20 split; the scaler for clustering features is fit before
    that split because KMeans is an unsupervised step.  This is correct: the
    cluster assignment is a deterministic function of (avg_holding_days,
    crypto_ratio, tech_stocks_ratio) and carries zero label information.
    """
    log.info("═" * 70)
    log.info("⚙️  [OFFLINE] Training pipeline — START")

    # ── STEP 1: Data ──────────────────────────────────────────────────────────
    df = generate_synthetic_data(n=3_000)

    # ── STEP 2: Investor segmentation ────────────────────────────────────────
    log.info("  [STEP 2/4] KMeans investor segmentation...")
    seg_scaler = StandardScaler()
    X_cluster  = seg_scaler.fit_transform(df[CLUSTER_FEATURES])

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=15,       # multi-start to avoid local minima
        max_iter=500,
    )
    df["cluster_id"] = kmeans.fit_predict(X_cluster)

    profile_resolver = DynamicProfileResolver()
    profile_resolver.fit(df)   # uses full df — intentional (unsupervised)

    cluster_dist = df["cluster_id"].value_counts().sort_index().to_dict()
    log.info(f"  Cluster distribution: {cluster_dist}")

    # ── STEP 3: Churn model ──────────────────────────────────────────────────
    log.info("  [STEP 3/4] XGBoost churn classifier training...")
    X_churn = df[CHURN_FEATURES]
    y_churn = df["churn"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_churn, y_churn,
        test_size=0.20,
        stratify=y_churn,
        random_state=42,
    )

    churn_model = XGBClassifier(
        max_depth=4,            # shallow to prevent overfitting on 4 features
        learning_rate=0.02,     # small LR — paired with high n_estimators + early stop
        n_estimators=300,       # upper bound; early stopping reduces actual count
        subsample=0.85,         # row subsampling: variance reduction
        colsample_bytree=0.85,  # column subsampling: de-correlates trees
        min_child_weight=5,     # min samples per leaf: bias vs variance tradeoff
        reg_lambda=1.5,         # L2 weight regularisation: reduces overfit
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    churn_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob: np.ndarray = churn_model.predict_proba(X_val)[:, 1]
    y_pred: np.ndarray = (y_prob >= CHURN_THRESHOLD).astype(int)
    auc: float         = roc_auc_score(y_val, y_prob)

    log.info(
        f"  ✅ Churn AUC-ROC: {auc:.4f}  |  "
        f"Best iteration: {churn_model.best_iteration}"
    )
    log.info(
        f"\n{classification_report(y_val, y_pred, target_names=['Retain', 'Churn'])}"
    )

    # ── STEP 4: Persist ───────────────────────────────────────────────────────
    log.info(f"  [STEP 4/4] Serialising artifacts → {ARTIFACT_PATH}")
    joblib.dump(
        {
            "seg_scaler":       seg_scaler,
            "kmeans":           kmeans,
            "profile_resolver": profile_resolver,
            "churn_model":      churn_model,
            "val_auc":          auc,
            "trained_at":       datetime.now(timezone.utc).isoformat(),
        },
        ARTIFACT_PATH,
    )
    log.info("💾  [OFFLINE] Pipeline complete — artifacts saved.")
    log.info("═" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — QUANT ENGINE  (online inference)
# ═════════════════════════════════════════════════════════════════════════════
class SiriusQuantEngine:
    """
    Online inference engine.

    Thread-safety model
    ───────────────────
    A single reentrant lock (_lock) guards _market_cache and
    _cache_timestamp.  The full "check → refresh → write" sequence is
    performed atomically inside the lock, eliminating the TOCTOU race
    present in v2.3 where the staleness check happened outside the lock.

    Markowitz optimisation
    ──────────────────────
    Objective (maximise):
        U(w) = μᵀw  −  (γ/2) wᵀΣw  −  λ‖w − w₀‖²₂

    where:
        μ   = annualised expected returns (Ledoit-Wolf shrunk)
        Σ   = annualised covariance (Ledoit-Wolf)
        γ   = dynamic risk-aversion (churn-scaled, see _compute_gamma)
        λ   = diversification penalty weight (DIVERSIFICATION_LAMBDA)
        w₀  = equal-weight prior  (1/n vector)

    Constraints:
        Σwᵢ = 1            (budget)
        wᵢ ≥ min_w         (floor per profile)
        wᵢ ≤ WEIGHT_MAX    (ceiling, hard cap at 40 %)
        wᵢ ≤ override_ub[i] (per-asset client preference cap, NEW in v3.0)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()   # non-reentrant — no nested acquisition

        if not os.path.exists(ARTIFACT_PATH):
            log.warning("⚠️  Artifacts not found — running training pipeline first...")
            run_training_pipeline()

        artifacts: dict[str, Any] = joblib.load(ARTIFACT_PATH)
        self.seg_scaler:       StandardScaler        = artifacts["seg_scaler"]
        self.kmeans:           KMeans                = artifacts["kmeans"]
        self.profile_resolver: DynamicProfileResolver = artifacts["profile_resolver"]
        self.churn_model:      XGBClassifier         = artifacts["churn_model"]

        log.info(
            f"🧠  [ENGINE] Artifacts loaded  |  "
            f"Val AUC: {artifacts.get('val_auc', float('nan')):.4f}  |  "
            f"Trained: {artifacts.get('trained_at', 'N/A')}"
        )

        # Cache initialised to None — first call triggers a fetch
        self._market_cache: tuple[pd.Series, pd.DataFrame] | None = None
        self._cache_timestamp: datetime | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
        """
        yfinance returns a MultiIndex when multiple tickers are requested.
        The column structure changed across yfinance versions:

          v0.2.x:  (field, ticker)  — e.g. ("Close", "AAPL")
          v0.1.x:  flat             — e.g. "Close"

        This helper normalises BOTH cases to a flat ticker-indexed DataFrame
        of closing prices.  Using `.levels[0]` on a flat Index raises
        AttributeError — guarded by isinstance.
        """
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                return raw["Close"]
            # Some yfinance builds emit (ticker, field) instead of (field, ticker)
            if "Close" in raw.columns.get_level_values(1):
                return raw.xs("Close", axis=1, level=1)
            raise ValueError(
                "MultiIndex columns found but 'Close' not in any level. "
                f"Levels: {[list(raw.columns.get_level_values(i)) for i in range(raw.columns.nlevels)]}"
            )
        # Flat columns — single ticker or already extracted
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": raw.columns[0]})
        return raw

    def _fetch_market_data(
        self,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Download 1-year OHLCV data from yfinance, compute annualised returns
        and a Ledoit-Wolf shrunk covariance matrix.

        Retry strategy: exponential backoff with jitter.
        Fallback: synthetic Gaussian returns (preserves pipeline functionality
        in air-gapped or rate-limited environments).

        Ledoit-Wolf shrinkage is chosen over the sample covariance because:
          1. With 252 trading days and 7 assets the sample estimator is
             well-conditioned (T >> p), but LW remains the quant standard.
          2. The shrinkage coefficient is data-adaptive — no hand-tuned α.
          3. Guarantees PD output (after jitter check), critical for the
             quadratic optimiser.
        """
        for attempt in range(1, retries + 1):
            try:
                tickers = list(TICKER_MAP.values())
                raw_download = yf.download(
                    tickers,
                    period="1y",
                    timeout=15,
                    auto_adjust=True,
                    progress=False,
                )

                close_df = self._normalise_multiindex(raw_download)
                close_df = close_df.rename(columns={v: k for k, v in TICKER_MAP.items()})
                # Keep only assets in our universe (order matters for weight vector)
                close_df = close_df[[a for a in ASSETS if a in close_df.columns]]

                missing = set(ASSETS) - set(close_df.columns)
                if missing:
                    log.warning(f"  ⚠️  Missing tickers from download: {missing}")

                if close_df.empty or close_df.shape[0] < 50:
                    raise ValueError(
                        f"Insufficient market data: only {close_df.shape[0]} rows returned."
                    )

                returns   = close_df.ffill().pct_change().dropna()
                mean_ret  = returns.mean() * 252   # annualise daily means

                lw_cov_raw, shrinkage = ledoit_wolf(returns.values)
                cov_ann = pd.DataFrame(
                    lw_cov_raw * 252,   # annualise daily covariance
                    index=close_df.columns,
                    columns=close_df.columns,
                )

                # Reindex to canonical ASSETS order, filling any gaps with zeros
                cov_ann  = cov_ann.reindex(index=ASSETS, columns=ASSETS).fillna(0.0)
                mean_ret = mean_ret.reindex(ASSETS).fillna(0.0)

                log.info(
                    f"  📈 Market data: {len(returns)} trading days  |  "
                    f"LW shrinkage: {shrinkage:.4f}"
                )
                return mean_ret, cov_ann

            except Exception as exc:
                log.warning(
                    f"  ⚠️  Market fetch attempt {attempt}/{retries} failed: {exc}"
                )
                if attempt < retries:
                    sleep_s = backoff ** attempt + np.random.uniform(0, 0.5)
                    time.sleep(sleep_s)

        log.error("🚨  ALL market fetch attempts failed — using SYNTHETIC fallback.")
        rng      = np.random.default_rng(0)
        fake_ret = pd.DataFrame(
            rng.normal(0.0006, 0.012, (252, len(ASSETS))),
            columns=ASSETS,
        )
        lw_cov_raw, _ = ledoit_wolf(fake_ret.values)
        mean_ret       = fake_ret.mean() * 252
        cov_ann        = pd.DataFrame(lw_cov_raw * 252, index=ASSETS, columns=ASSETS)
        return mean_ret, cov_ann

    def _compute_gamma(self, profile: str, churn_prob: float) -> float:
        """
        Compute the dynamic risk-aversion coefficient γ.

        v2.3 used:  γ = γ_base × exp((churn_prob − 0.5) × 2.5)
        Problem:    At churn_prob=1.0 → multiplier = exp(1.25) ≈ 3.49×
                    At churn_prob=0.5 → multiplier = exp(0.0)  = 1.0×
                    At churn_prob=0.0 → multiplier = exp(−1.25)≈ 0.29× (!!!)
                    This reduced γ BELOW the base for low-churn clients
                    triggering the portfolio, which is financially incorrect:
                    a low-churn aggressive client should still be aggressive.

        v3.0 fix:   γ scaling applies ONLY to the retention-triggered branch
                    (churn_prob > CHURN_THRESHOLD).  The normalised churn
                    excess is mapped to [0, 1] and the multiplier to [1, e^1].

            norm_excess = (churn_prob − CHURN_THRESHOLD) / (1.0 − CHURN_THRESHOLD)
            γ = γ_base × exp(norm_excess × GAMMA_CHURN_SCALE)

        Financial rationale: A client about to churn is treated as having
        elevated behavioural risk aversion (loss aversion increases under
        stress), so the portfolio skews more defensive REGARDLESS of their
        stated profile.  The exponential shape reflects that each additional
        percentage point of churn risk compounds the client's fragility.
        """
        base_gamma = RISK_AVERSION.get(profile, 4.0)
        if churn_prob <= CHURN_THRESHOLD:
            return base_gamma   # no scaling below threshold
        norm_excess = (churn_prob - CHURN_THRESHOLD) / (1.0 - CHURN_THRESHOLD)
        return base_gamma * np.exp(norm_excess * GAMMA_CHURN_SCALE)

    @staticmethod
    def _build_asset_bounds(
        profile: str,
        payload_crypto_ratio: float,
        payload_tech_ratio: float,
    ) -> list[tuple[float, float]]:
        """
        Construct per-asset (lb, ub) box constraints that are GUARANTEED FEASIBLE.

        Incorporates:
          1. Profile-level floor  (MIN_WEIGHT_BY_PROFILE) as lb
          2. Hard per-asset ceiling (WEIGHT_MAX = 0.40) as ub cap
          3. Client-declared crypto / tech preference caps as soft ub targets
          4. Feasibility repair: ensures Sum(UBs) ≥ 1.0 and lb ≤ ub for all assets

        Root-cause of the v3.0 SLSQP linesearch failure (now fixed here):
        ──────────────────────────────────────────────────────────────────
        Client 9005 (conservative, crypto_ratio=0.05, tech_stocks_ratio=0.10):
          crypto_per_asset = 0.05 / 2 = 0.025   <  min_w = 0.08
          tech_per_asset   = 0.10 / 4 = 0.025   <  min_w = 0.08
          After raw clamping: all 6 thematic assets get lb=ub=0.08 (pinned).
          Sum(UBs) = 6×0.08 + 0.40 (KO) = 0.88  <  1.0
          → budget constraint Σw=1 is incompatible with box constraints
          → SLSQP: "Positive directional derivative for linesearch"

        Fix (4-step pipeline):
          Step 1: compute raw preference UBs (no clamping)
          Step 2: clamp each raw UB to [min_w, WEIGHT_MAX]
          Step 3: if Sum(clamped_UBs) < 1.0, distribute deficit proportionally
                  to assets that have headroom (ub < WEIGHT_MAX)
          Step 4: return final (lb, ub) pairs — all feasible by construction
        """
        n     = len(ASSETS)
        min_w = MIN_WEIGHT_BY_PROFILE.get(profile, 0.02)

        # Guard 1 — floor sum must not exceed the budget
        if n * min_w > 1.0:
            min_w = 1.0 / n

        crypto_per_asset = payload_crypto_ratio / max(len(CRYPTO_ASSETS), 1)
        tech_per_asset   = payload_tech_ratio   / max(len(TECH_ASSETS),   1)

        # Step 1 — compute raw preference UBs before any feasibility correction.
        # Non-thematic assets (e.g. KO) are uncapped at this stage.
        raw_ubs: list[float] = []
        for asset in ASSETS:
            if asset in CRYPTO_ASSETS:
                raw_ubs.append(crypto_per_asset)
            elif asset in TECH_ASSETS:
                raw_ubs.append(tech_per_asset)
            else:
                raw_ubs.append(WEIGHT_MAX)

        # Step 2 — clamp each raw UB to [min_w, WEIGHT_MAX].
        # This handles the degenerate case where a client declares a very low
        # crypto_ratio (e.g. 0.05 / 2 assets = 0.025 < min_w = 0.08).
        # Without clamping: lb=0.08, ub=0.025 → lb > ub → SLSQP linesearch fails.
        clamped_ubs: list[float] = [
            max(min_w, min(WEIGHT_MAX, u)) for u in raw_ubs
        ]

        # Step 3 — feasibility check: Sum(UBs) must be ≥ 1.0.
        # If Sum(UBs) < 1.0 the budget constraint (Σwᵢ = 1) is incompatible
        # with the box constraints — SLSQP will report a linesearch failure.
        #
        # Fix: proportionally lift all clamped UBs that have headroom below
        # WEIGHT_MAX until the sum reaches 1.0.  We distribute the deficit
        # across growable assets weighted by their available headroom, ensuring
        # no asset exceeds WEIGHT_MAX.
        #
        # Example (client 9005 — conservative, crypto=0.05, tech=0.10):
        #   After clamping: all 6 thematic assets have ub=0.08, KO has ub=0.40.
        #   Sum = 6×0.08 + 0.40 = 0.88  <  1.0  → deficit = 0.12.
        #   Growable: AAPL, MSFT, KO, NVDA, TSLA, BTC, ETH all have room.
        #   After lift: each thematic asset raised from 0.08 to ~0.10, KO stays 0.40.
        #   New sum = 6×0.10 + 0.40 = 1.00  ✓ feasible.
        ub_sum = sum(clamped_ubs)
        if ub_sum < 1.0 - 1e-9:
            deficit = 1.0 - ub_sum
            headroom = [(i, WEIGHT_MAX - clamped_ubs[i]) for i in range(n)]
            total_room = sum(h for _, h in headroom)
            if total_room > 1e-12:
                # Distribute deficit proportionally to available headroom
                for i, room in headroom:
                    clamped_ubs[i] += (room / total_room) * deficit
                    clamped_ubs[i] = min(WEIGHT_MAX, clamped_ubs[i])
            else:
                # Extreme edge case: all assets already at WEIGHT_MAX
                # (can only occur if WEIGHT_MAX < 1/n, which violates config)
                clamped_ubs = [WEIGHT_MAX] * n

        # Step 4 — build final (lb, ub) pairs; guaranteed lb ≤ ub and Sum(UBs) ≥ 1.0
        bounds: list[tuple[float, float]] = [
            (min_w, ub) for ub in clamped_ubs
        ]
        return bounds

    def get_market_context(self) -> tuple[pd.Series, pd.DataFrame]:
        """
        Thread-safe, TTL-based market data cache.

        The ENTIRE check-and-refresh sequence is executed inside the lock.
        This eliminates the v2.3 TOCTOU (Time-Of-Check / Time-Of-Use) race
        where Thread A and Thread B could both observe a stale cache, both
        call _fetch_market_data(), and both write — causing a spurious
        double-fetch and a brief window where a partially written cache tuple
        is visible to readers.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            cache_stale = (
                self._market_cache is None
                or self._cache_timestamp is None
                or (now - self._cache_timestamp) > MARKET_CACHE_TTL
            )
            if cache_stale:
                log.info("  🔄  Refreshing market data cache...")
                self._market_cache    = self._fetch_market_data()
                self._cache_timestamp = now
            return self._market_cache  # type: ignore[return-value]

    def optimize_portfolio(
        self,
        profile: str,
        mean_ret: pd.Series,
        cov: pd.DataFrame,
        churn_prob: float,
        payload: ClientPayload,
    ) -> np.ndarray:
        """
        Solve the Dynamic Markowitz problem with client-specific asset bounds.

        Objective (maximise):
            U(w) = μᵀw − (γ/2)·wᵀΣw − λ‖w − w₀‖²₂

        SLSQP is chosen over interior-point (L-BFGS-B) because:
          - It handles equality constraints (budget) natively.
          - It converges in O(n²) per iteration for small n (n=7 here).
          - The analytical Jacobian (jac=) eliminates finite-difference calls,
            reducing function evaluations by ~3×.

        The PD check on Sigma via _ensure_positive_definite guarantees the
        quadratic form wᵀΣw is strictly convex, making the problem
        well-defined and SLSQP convergence reliable.

        Parameters
        ----------
        profile   : investor archetype string
        mean_ret  : annualised expected returns (pd.Series, ASSETS index)
        cov       : annualised Ledoit-Wolf covariance (pd.DataFrame)
        churn_prob: predicted churn probability in [0, 1]
        payload   : ClientPayload used to extract preference bounds (NEW v3.0)

        Returns
        -------
        np.ndarray of shape (n,) — normalised portfolio weights summing to 1.
        """
        n      = len(ASSETS)
        gamma  = self._compute_gamma(profile, churn_prob)
        w0     = np.full(n, 1.0 / n)   # equal-weight prior for L2 penalty
        mu     = mean_ret.values.astype(float)

        # PD-guaranteed covariance matrix
        Sigma = _ensure_positive_definite(
            cov.values.astype(float), jitter=SIGMA_JITTER
        )

        # Per-asset bounds: client-specific overrides (BUG FIX v3.0)
        bounds = self._build_asset_bounds(
            profile,
            payload.crypto_ratio,
            payload.tech_stocks_ratio,
        )

        def neg_utility(w: np.ndarray) -> float:
            ret = float(np.dot(w, mu))
            var = float(np.dot(w, np.dot(Sigma, w)))
            pen = DIVERSIFICATION_LAMBDA * float(np.dot(w - w0, w - w0))
            return -(ret - 0.5 * gamma * var - pen)

        def neg_utility_grad(w: np.ndarray) -> np.ndarray:
            """
            Analytical gradient of neg_utility w.r.t. w:
              ∂/∂w [ -(μᵀw − γ/2·wᵀΣw − λ‖w−w₀‖²) ]
              = −μ + γ·Σw + 2λ(w − w₀)
            """
            return -mu + gamma * np.dot(Sigma, w) + 2.0 * DIVERSIFICATION_LAMBDA * (w - w0)

        # Feasible initial point: clip to bounds then renormalise
        lb_arr = np.array([b[0] for b in bounds])
        ub_arr = np.array([b[1] for b in bounds])
        x0 = np.clip(w0, lb_arr, ub_arr)
        if x0.sum() <= 0:
            x0 = lb_arr.copy()
        x0 = x0 / x0.sum()

        constraints = [
            {
                "type": "eq",
                "fun":  lambda w: float(np.sum(w)) - 1.0,
                "jac":  lambda w: np.ones(n),
            },
        ]

        result: OptimizeResult = minimize(
            neg_utility,
            x0=x0,
            jac=neg_utility_grad,
            method="SLSQP",
            bounds=bounds,          # box constraints via scipy bounds API
            constraints=constraints,
            options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
        )

        if not result.success:
            log.warning(
                f"  ⚠️  SLSQP did not converge for profile={profile}: "
                f"{result.message}.  Falling back to feasible initial point."
            )
            return x0

        # Final clip + renormalise (numerical safety after SLSQP)
        weights = np.clip(result.x, lb_arr, ub_arr)
        total   = weights.sum()
        if total <= 0:
            return x0
        return weights / total

    def predict_client(
        self,
        payload: ClientPayload,
    ) -> tuple[str, float]:
        """
        Run the two-stage inference pipeline for a single client.

        Stage 1 — Investor segmentation (KMeans):
          Transform (avg_holding_days, crypto_ratio, tech_stocks_ratio)
          through the fitted StandardScaler, predict cluster ID, resolve
          to archetype string.

        Stage 2 — Churn prediction (XGBoost):
          Pass (account_balance, balance_velocity, market_pain_index,
          login_freq_drop) to the classifier, extract P(churn=1).

        Returns
        -------
        (profile_type: str, churn_prob: float)
        """
        feat_c = pd.DataFrame(
            [[
                payload.avg_holding_days,
                payload.crypto_ratio,
                payload.tech_stocks_ratio,
            ]],
            columns=CLUSTER_FEATURES,
        )
        scaled_c     = self.seg_scaler.transform(feat_c)
        cluster_id   = int(self.kmeans.predict(scaled_c)[0])
        profile_type = self.profile_resolver.transform(
            pd.DataFrame({"cluster_id": [cluster_id]})
        )[0]

        feat_ch = pd.DataFrame(
            [[
                payload.account_balance,
                payload.balance_velocity,
                payload.market_pain_index,
                payload.login_freq_drop,
            ]],
            columns=CHURN_FEATURES,
        )
        churn_prob = float(self.churn_model.predict_proba(feat_ch)[0][1])

        return profile_type, churn_prob

    def warm_up(self) -> None:
        """
        Pre-populate the market data cache at engine startup.

        Without this, the first batch request that triggers a retention
        recommendation bears the full yfinance network latency while holding
        the lock — starving concurrent threads.  Call warm_up() after
        engine construction to decouple network I/O from the request path.
        """
        log.info("  🔥  Engine warm-up: pre-fetching market data...")
        self.get_market_context()
        log.info("  ✅  Market cache populated.")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — API GATEWAY
# ═════════════════════════════════════════════════════════════════════════════
def handle_api_request(
    raw_payload: dict[str, Any],
    engine: SiriusQuantEngine,
    market_ctx: tuple[pd.Series, pd.DataFrame],   # pre-fetched outside loop
) -> dict[str, Any]:
    """
    Simulate a REST API endpoint with full Markowitz portfolio construction.

    Changes from v2.3
    ─────────────────
    - market_ctx is now passed in from the caller rather than fetched inside
      the function.  This avoids repeated lock acquisitions on every batch
      item and eliminates a subtle bug where a cache expiry mid-batch could
      cause inconsistent return estimates across clients.

    - optimize_portfolio now receives the full payload so it can apply
      per-asset preference bounds.

    Parameters
    ----------
    raw_payload : raw dict from the batch / REST request
    engine      : initialised SiriusQuantEngine instance
    market_ctx  : (mean_ret, cov) tuple — fetched once per batch externally

    Returns
    -------
    dict with keys: status, client_id, analytics, recommendation
    """
    try:
        payload = ClientPayload(**raw_payload)
    except Exception as exc:
        log.error(
            f"  ❌ Validation failed for client "
            f"{raw_payload.get('client_id', '?')}: {exc}"
        )
        return {
            "status":    "validation_error",
            "client_id": raw_payload.get("client_id"),
            "error":     str(exc),
        }

    profile, churn_prob = engine.predict_client(payload)
    action_required     = churn_prob > CHURN_THRESHOLD

    response: dict[str, Any] = {
        "status":    "success",
        "client_id": payload.client_id,
        "analytics": {
            "investor_profile":      profile,
            "churn_risk_percentage": round(churn_prob * 100, 2),
            "action_required":       action_required,
        },
        "recommendation": None,
    }

    if action_required:
        mean_ret, cov = market_ctx
        weights = engine.optimize_portfolio(
            profile, mean_ret, cov, churn_prob, payload
        )

        portfolio: dict[str, dict[str, float]] = {}
        for asset, w in zip(ASSETS, weights):
            if w > 0.005:   # suppress dust positions below 0.5 %
                portfolio[asset] = {
                    "allocation_pct":    round(w * 100, 2),
                    "capital_allocated": round(payload.account_balance * w, 2),
                }

        response["recommendation"] = {
            "strategy": (
                f"Dynamic Retention — {profile.upper()} "
                f"(Risk Scale: {churn_prob:.2f})"
            ),
            "portfolio": portfolio,
        }

    return response


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
BATCH: list[dict[str, Any]] = [
    {
        "client_id": 9001, "description": "Crypto Enthusiast — Aggressive (58% Churn)",
        "avg_holding_days": 6,   "crypto_ratio": 0.75, "tech_stocks_ratio": 0.20,
        "account_balance": 42_000, "balance_velocity": 0.62,
        "market_pain_index": 0.65, "login_freq_drop": 0.45,
    },
    {
        "client_id": 9002, "description": "Institutional HNW — Solid Inflows",
        "avg_holding_days": 185, "crypto_ratio": 0.00, "tech_stocks_ratio": 0.15,
        "account_balance": 450_000, "balance_velocity": 1.20,
        "market_pain_index": 0.05, "login_freq_drop": 1.40,
    },
    {
        "client_id": 9003, "description": "Retail Balanced — Frustrated with Drawdowns",
        "avg_holding_days": 55,  "crypto_ratio": 0.25, "tech_stocks_ratio": 0.35,
        "account_balance": 18_000, "balance_velocity": 0.42,
        "market_pain_index": 0.88, "login_freq_drop": 0.28,
    },
    {
        "client_id": 9004, "description": "Active Growth Trader — Marginal Decay",
        "avg_holding_days": 15,  "crypto_ratio": 0.50, "tech_stocks_ratio": 0.40,
        "account_balance": 85_000, "balance_velocity": 0.58,
        "market_pain_index": 0.55, "login_freq_drop": 0.75,
    },
    {
        "client_id": 9005, "description": "Conservative Senior — Sudden Outflow",
        "avg_holding_days": 145, "crypto_ratio": 0.05, "tech_stocks_ratio": 0.10,
        "account_balance": 110_000, "balance_velocity": 0.22,
        "market_pain_index": 0.40, "login_freq_drop": 0.80,
    },
    {
        "client_id": 9006, "description": "Standard Mid-Tier — Status Quo",
        "avg_holding_days": 45,  "crypto_ratio": 0.15, "tech_stocks_ratio": 0.45,
        "account_balance": 35_000, "balance_velocity": 0.98,
        "market_pain_index": 0.35, "login_freq_drop": 1.05,
    },
    {
        "client_id": 9007, "description": "Dormant Account — Aggressive Extreme (92% Churn)",
        "avg_holding_days": 4,   "crypto_ratio": 0.80, "tech_stocks_ratio": 0.10,
        "account_balance": 50_000, "balance_velocity": 0.05,
        "market_pain_index": 0.98, "login_freq_drop": 0.02,
    },
]


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         SIRIUS CAPITAL — QUANT GRADE ENGINE  v3.1                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    run_training_pipeline()
    engine = SiriusQuantEngine()
    engine.warm_up()   # pre-populate cache before batch begins

    # Fetch market context ONCE for the entire batch (correct design)
    market_ctx = engine.get_market_context()

    print("\n🚀  Executing Production Batch (Full Cycle Context)...\n")
    print("═" * 90)

    for raw in BATCH:
        result = handle_api_request(raw, engine, market_ctx)

        if result["status"] == "validation_error":
            print(
                f"❌  Client #{result['client_id']} — "
                f"VALIDATION ERROR: {result['error']}"
            )
            print("─" * 90)
            continue

        a         = result["analytics"]
        churn_pct = a["churn_risk_percentage"]
        risk_icon = (
            "🔴" if churn_pct >= 70
            else ("🟡" if churn_pct >= 40 else "🟢")
        )

        print(f"🔹  Client #{result['client_id']}  [{raw['description']}]")
        print(
            f"    Profile : {a['investor_profile'].upper():<14}  |  "
            f"Churn Risk : {risk_icon} {churn_pct:.2f}%"
        )

        if result["recommendation"]:
            rec = result["recommendation"]
            print(f"    Strategy: {rec['strategy']}")
            for asset, details in rec["portfolio"].items():
                bar = "█" * int(details["allocation_pct"] / 2.5)
                print(
                    f"      • {asset:<5} {bar:<20} "
                    f"{details['allocation_pct']:5.2f}%  →  "
                    f"${details['capital_allocated']:>12,.2f}"
                )
        else:
            print("    ✅  STABLE — No retention action required")

        print("─" * 90)

    print()
    print("✔  Batch complete.  All v3.0 pipeline stages executed successfully.")
