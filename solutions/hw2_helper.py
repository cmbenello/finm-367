from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
import matplotlib.pyplot as plt

# Data Loading
@dataclass
class ProSharesData:
    hedge: pd.DataFrame     
    factors: pd.DataFrame   
    other: pd.DataFrame    
    meta: pd.DataFrame      

def _normalize_date_index(df):
    df = df.copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_data(xlsx_path):
    xlsx_path = Path(xlsx_path)
    hedge = _normalize_date_index(pd.read_excel(xlsx_path, sheet_name="hedge_fund_series"))
    factors = _normalize_date_index(pd.read_excel(xlsx_path, sheet_name="merrill_factors"))
    other = _normalize_date_index(pd.read_excel(xlsx_path, sheet_name="other_data"))
    meta = pd.read_excel(xlsx_path, sheet_name="descriptions")
    return ProSharesData(hedge=hedge, factors=factors, other=other, meta=meta)

def align_on_index(*dfs):
    common = dfs[0].index
    for d in dfs[1:]:
        common = common.intersection(d.index)
    return [d.loc[common] for d in dfs]


# Performance Stats
def performance_stats(returns, rf=None, periods=12):
    r = returns.copy()
    mu = r.mean() * periods
    vol = r.std(ddof=1) * np.sqrt(periods)
    if rf == None:
        sharpe = np.where(vol != 0, (r.mean()) / r.std(ddof=1) * np.sqrt(periods), np.nan)
    else:
        sharpe = np.where(vol != 0, (r.mean() - rf) / r.std(ddof=1) * np.sqrt(periods), np.nan)
    out = pd.DataFrame({"mean": mu, "vol": vol, "sharpe": sharpe}, index=returns.columns)
    return out

def to_cumulative_returns(r):
    return (1 + r.fillna(0)).cumprod() - 1

@dataclass
class DrawdownDetail:
    peak_date: Optional[pd.Timestamp]
    trough_date: Optional[pd.Timestamp]
    recovery_date: Optional[pd.Timestamp]
    max_drawdown: float  

def max_drawdown_path(r):
    s = pd.Series(r).dropna()
    if s.empty:
        return DrawdownDetail(None, None, None, np.nan)

    wealth = (1.0 + s).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])

    pre = wealth.loc[:trough_date]
    at_peak = pre.eq(pre.cummax())
    at_peak = at_peak.loc[at_peak.index < trough_date]

    if at_peak.any():
        peak_date = at_peak.index[at_peak].max()
    else:
        peak_date = pre.index.min()

    recovery_date = None
    if peak_date is not None:
        peak_level = wealth.loc[peak_date]
        after = wealth.loc[trough_date:]
        recovered = after[after >= peak_level]
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
        return DrawdownDetail(peak_date, trough_date, recovery_date, max_dd)

def tail_risk_stats(returns, alpha=0.05):

    def _skew(x):
        x = pd.Series(x).dropna()
        return stats.skew(x, bias=False) if len(x) > 2 else np.nan

    def _ex_kurt(x):
        x = pd.Series(x).dropna()
        return stats.kurtosis(x, fisher=True, bias=False) if len(x) > 3 else np.nan

    q = returns.quantile(alpha, interpolation="linear")

    cvar = {}
    dd = {}
    for c in returns.columns:
        series = returns[c].dropna()
        if series.empty or pd.isna(q[c]):
            cvar[c] = np.nan
            dd[c] = DrawdownDetail(None, None, None, np.nan)
            continue
        var_c = q[c]
        cvar[c] = series[series <= var_c].mean()
        dd[c] = max_drawdown_path(series)

    out = pd.DataFrame({
        "skewness": returns.apply(_skew, axis=0),
        "excess_kurtosis": returns.apply(_ex_kurt, axis=0),
        f"VaR_{alpha:.2f}": q,
        f"CVaR_{alpha:.2f}": pd.Series(cvar),
        "max_drawdown": pd.Series({k: v.max_drawdown for k, v in dd.items()}),
        "dd_peak": pd.Series({k: v.peak_date for k, v in dd.items()}),
        "dd_trough": pd.Series({k: v.trough_date for k, v in dd.items()}),
        "dd_recovery": pd.Series({k: v.recovery_date for k, v in dd.items()}),
    }).loc[returns.columns]

    return out

# Regressions
def regress_vs_market(returns, market, rf=0, periods=12):
    aligned = pd.concat([returns, market.rename("market")], axis=1).dropna()
    m = aligned["market"]
    out = []
    rf_ann = rf * periods
    for col in returns.columns:
        y = aligned[col]
        X = sm.add_constant(m)
        model = sm.OLS(y, X).fit()
        beta = model.params["market"]
        alpha = model.params["const"]
        r2 = float(model.rsquared)

        mean_ann = y.mean() * periods
        treynor = np.nan
        if beta != 0:
            treynor = (mean_ann - rf_ann) / beta

        ir = alpha /  model.resid.std()
        ir_ann = ir * np.sqrt(periods) if pd.notnull(ir) else np.nan
        
        out.append({
            "alpha": alpha, 
            "beta": beta, 
            "r2": r2, 
            "treynor": treynor, 
            "ir": ir_ann})
    return pd.DataFrame(out, index=returns.columns)

@dataclass
class OLSReplication:
    alpha: float
    betas: pd.Series
    r2: float
    tracking_error_ann: float
    fitted: pd.Series
    residuals: pd.Series

def ols_replication(target, factors, add_const=True, periods=12):
    df = pd.concat([target.rename("target"), factors], axis=1).dropna()
    y = df["target"]
    X = sm.add_constant(df[factors.columns]) if add_const else df[factors.columns]
    model = sm.OLS(y, X).fit()
    params = model.params.copy()
    alpha = float(params.pop("const")) if add_const and "const" in params.index.union(["const"]) and "const" in model.params else 0.0
    betas = params
    resid = model.resid
    te = resid.std(ddof=1) * np.sqrt(periods)
    return OLSReplication(alpha=alpha, betas=betas, r2=float(model.rsquared), tracking_error_ann=float(te), fitted=model.fittedvalues, residuals=resid)

def rolling_replication_oos(target, factors, window=60, add_const= True):
    df = pd.concat([target.rename("target"), factors], axis=1).dropna()
    dates = df.index
    cols = factors.columns.tolist()
    oos = pd.Series(index=dates, dtype=float)
    for i in range(window, len(df)):
        train = df.iloc[i-window:i]
        test = df.iloc[i:i+1]
        y = train["target"]
        X_train = sm.add_constant(train[cols]) if add_const else train[cols]
        model = sm.OLS(y, X_train).fit()
        X_test = sm.add_constant(test[cols]) if add_const else test[cols]
        X_test = X_test.reindex(columns=X_train.columns, fill_value=1.0 if add_const else 0.0)
        oos.iloc[i] = float(model.predict(X_test).iloc[0])
    return oos

@dataclass
class ConstrainedReplication:
    betas: pd.Series
    fitted: pd.Series
    residuals: pd.Series
    r2: float
    tracking_error_ann: float


def nnls_replication(target, factors, periods=12, intercept=False):
    df = pd.concat([target.rename("target"), factors], axis=1).dropna()
    y = df["target"].to_numpy()
    X = df[factors.columns].to_numpy()
    cols = list(factors.columns)
    if intercept:
        X = np.column_stack([np.ones(len(df)), X])
        cols = ["intercept"] + cols
    b, _ = optimize.nnls(X, y)
    fitted = pd.Series(X @ b, index=df.index)
    resid = df["target"] - fitted
    r2 = 1 - np.var(resid, ddof=1) / np.var(df["target"], ddof=1)
    te = resid.std(ddof=1) * np.sqrt(periods)
    
    return ConstrainedReplication(
        betas=pd.Series(b, index=cols),
        fitted=fitted,
        residuals=resid,
        r2=float(r2),
        tracking_error_ann=float(te),
    )

def nnls_report(target, factors, periods=12, intercept=True):
    res = nnls_replication(target, factors, periods, intercept)
    y = pd.concat([pd.Series(target).rename("y"), res.fitted.rename("yhat")], axis=1).dropna()
    c = float(np.corrcoef(y["y"], y["yhat"])[0,1]) if len(y) > 1 else np.nan
    return {"betas": res.betas, "r2": float(res.r2), "tracking_error_ann": float(res.tracking_error_ann), "corr": c, "fitted": res.fitted}


def bounded_ls_replication(target, factors, lower, upper, periods = 12, intercept=False):
    df = pd.concat([target.rename("target"), factors], axis=1).dropna()
    y = df["target"].to_numpy()
    X = df[factors.columns].to_numpy()
    p = X.shape[1] + int(intercept)
    cols = list(factors.columns)
    if intercept:
        X = np.column_stack([np.ones(len(df)), X])
        cols = ["intercept"] + cols

    lo = np.asarray(lower)
    hi = np.asarray(upper)

    if lo.ndim == 0:
        lo = np.full(p, float(lo))
        if intercept:
            lo[0] = -np.inf
    elif lo.size == p:
        lo = lo.astype(float)
    elif intercept and lo.size == p - 1:
        lo = np.concatenate((np.array([-np.inf]), lo.astype(float)))
    else:
        raise ValueError("lower bounds must be scalar or length equal to number of parameters")

    if hi.ndim == 0:
        hi = np.full(p, float(hi))
        if intercept:
            hi[0] = np.inf
    elif hi.size == p:
        hi = hi.astype(float)
    elif intercept and hi.size == p - 1:
        hi = np.concatenate((np.array([np.inf]), hi.astype(float)))
    else:
        raise ValueError("upper bounds must be scalar or length equal to number of parameters")

    res = optimize.lsq_linear(X, y, bounds=(lo, hi), lsmr_tol='auto', verbose=0)
    b = res.x
    fitted = pd.Series(X @ b, index=df.index)
    resid = df["target"] - fitted
    r2 = 1 - np.var(resid, ddof=1) / np.var(df["target"], ddof=1)
    te = resid.std(ddof=1) * np.sqrt(periods)
    return ConstrainedReplication(
        betas=pd.Series(b, index=cols),
        fitted=fitted,
        residuals=resid,
        r2=float(r2),
        tracking_error_ann=float(te),
    )

def bounded_ls_report(target, factors, lower, upper, periods=12, intercept=True):
    res = bounded_ls_replication(target, factors, lower, upper, periods, intercept)
    y = pd.concat([pd.Series(target).rename("y"), res.fitted.rename("yhat")], axis=1).dropna()
    c = float(np.corrcoef(y["y"], y["yhat"])[0,1]) if len(y) > 1 else np.nan
    return {"betas": res.betas, "r2": float(res.r2), "tracking_error_ann": float(res.tracking_error_ann), "corr": c, "fitted": res.fitted}


def constrained_sum_replication(target, factors, lower, upper, include, lsum, usum, periods=12):
    df = pd.concat([pd.Series(target).rename("target"), factors], axis=1).dropna()
    y = df["target"].to_numpy()
    X = df[factors.columns].to_numpy()
    n = X.shape[1]
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (n,))
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (n,))
    idx = [factors.columns.get_loc(c) for c in include]
    w = np.zeros(n); w[idx] = 1.0
    def obj(b):
        e = X.dot(b) - y
        return 0.5 * np.dot(e, e)
    cons = [
        {"type":"ineq","fun":lambda b,w=w,usum=usum: usum - np.dot(w,b)},
        {"type":"ineq","fun":lambda b,w=w,lsum=lsum: np.dot(w,b) - lsum},
    ]
    res = optimize.minimize(obj, x0=np.zeros(n), method="SLSQP", bounds=list(zip(lower, upper)), constraints=cons)
    b = res.x
    fitted = pd.Series(X.dot(b), index=df.index)
    resid = df["target"] - fitted
    r2 = 1 - np.var(resid, ddof=1) / np.var(df["target"], ddof=1)
    te = resid.std(ddof=1) * np.sqrt(periods)
    return ConstrainedReplication(betas=pd.Series(b, index=factors.columns), fitted=fitted, residuals=resid, r2=float(r2), tracking_error_ann=float(te))

def replication_report(target, factors, add_const=True, periods=12):
    df = pd.concat([pd.Series(target).rename("target"), factors], axis=1).dropna()
    rep = ols_replication(df["target"], df[factors.columns], add_const=add_const, periods=periods)
    y = df["target"]
    yhat = rep.fitted
    mkt = np.nan
    if "SPY US Equity" in rep.betas.index:
        mkt = float(rep.betas["SPY US Equity"])
    return {
        "alpha": float(rep.alpha),
        "betas": rep.betas,
        "r2": float(rep.r2),
        "tracking_error_ann": float(rep.tracking_error_ann),
        "mean_fitted": float(yhat.mean()),
        "mean_actual": float(y.mean()),
        "corr": float(np.corrcoef(y, yhat)[0,1]) if len(y) > 1 else np.nan,
        "beta_mkt": mkt
    }

def compare_replications_table(target, factors, periods=12):
    a = replication_report(target, factors, True, periods)
    b = replication_report(target, factors, False, periods)
    t = pd.DataFrame(
        [
            [a["alpha"], a["beta_mkt"], a["r2"], a["tracking_error_ann"], a["mean_fitted"], a["mean_actual"], a["corr"], float(a["betas"].abs().sum())],
            [b["alpha"], b["beta_mkt"], b["r2"], b["tracking_error_ann"], b["mean_fitted"], b["mean_actual"], b["corr"], float(b["betas"].abs().sum())],
        ],
        columns=["alpha","beta_mkt","r2","te_ann","mean_fitted","mean_actual","corr","beta_abs_sum"],
        index=["with_const","no_const"]
    )
    return t, a["betas"], b["betas"]


def correlation_matrix(returns):
    corr_mat = returns.corr()
    np.fill_diagonal(corr_mat.values,np.nan)
    return corr_mat

def plot_correlation_heatmap(corr):
    np.fill_diagonal(corr.copy().values, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, interpolation='nearest')
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    plt.show()

def cumulative_returns_panel(df):
    return (1 + df.fillna(0)).cumprod() - 1

__all__ = [
    "load_data",
    "performance_stats",
    "tail_risk_stats",
    "regress_vs_market",
    "correlation_matrix",
    "plot_correlation_heatmap",
    "ols_replication",
    "rolling_replication_oos",
    "align_on_index",
    "compare_replications_table",
    "nnls_report",
    "bounded_ls_report",
    "constrained_sum_replication",
    "cumulative_returns_panel",
]



