from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np

from .base import BaseReliabilityEstimator, FitInfo


# =========================
# Core helpers (existing)
# =========================

@dataclass(frozen=True)
class CronbachDetails:
    n_objects_used: int
    n_items: int
    alpha: float
    item_variances: np.ndarray
    total_score_variance: float
    mean_inter_item_cov: float
    mean_item_variance: float


def _nanvar(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanvar(x, axis=axis, ddof=1)


def _pairwise_cov_mean(X: np.ndarray) -> float:
    n_items = X.shape[1]
    covs = []
    for i in range(n_items):
        xi = X[:, i]
        for j in range(i + 1, n_items):
            xj = X[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if np.sum(mask) < 2:
                continue
            cov = np.cov(xi[mask], xj[mask], ddof=1)[0, 1]
            covs.append(cov)
    return float(np.mean(covs)) if covs else float("nan")


def _cronbach_alpha_pairwise(X: np.ndarray) -> Tuple[float, CronbachDetails]:
    if X.ndim != 2:
        raise ValueError("CronbachAlpha expects a 2D matrix.")

    n_objects, n_items = X.shape
    if n_items < 2:
        raise ValueError("Cronbach's alpha requires at least 2 items.")

    item_vars = _nanvar(X, axis=0)
    total_scores = np.nansum(X, axis=1)
    mask_total = ~np.isnan(total_scores)
    n_objects_used = int(np.sum(mask_total))

    if n_objects_used < 2:
        raise ValueError("Not enough valid rows to compute total variance.")

    total_var = float(np.nanvar(total_scores, ddof=1))
    mean_cov = _pairwise_cov_mean(X)
    mean_item_var = float(np.nanmean(item_vars))

    k = n_items
    denom = mean_item_var + (k - 1) * mean_cov
    alpha = float((k * mean_cov) / denom) if denom > 0 else float("nan")

    details = CronbachDetails(
        n_objects_used=n_objects_used,
        n_items=n_items,
        alpha=alpha,
        item_variances=item_vars,
        total_score_variance=total_var,
        mean_inter_item_cov=mean_cov,
        mean_item_variance=mean_item_var,
    )
    return alpha, details


# =========================
# NEW: Diagnostics helpers
# =========================

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _reverse_item(col: np.ndarray) -> np.ndarray:
    """
    Reverse-code a numeric item column using min+max reflection.
    NaNs are preserved.
    """
    col = np.asarray(col, dtype=float)
    mask = ~np.isnan(col)
    if not np.any(mask):
        return col.copy()
    mn = np.nanmin(col)
    mx = np.nanmax(col)
    rev = col.copy()
    rev[mask] = (mn + mx) - col[mask]
    return rev


def cronbach_diagnostics(
    X: np.ndarray,
    overall_alpha: float,
    *,
    item_names: Optional[List[str]] = None,
    citc_threshold: float = 0.30,
    remove_delta: float = 0.01,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Item-level diagnostics for Cronbach's alpha.
    Uses pairwise-complete strategy (consistent with main alpha).
    """
    X = np.asarray(X, dtype=float)
    n_objects, n_items = X.shape

    if not isinstance(item_names, list) or len(item_names) != n_items:
        item_names = [f"Q{i+1}" for i in range(n_items)]

    items: List[Dict[str, object]] = []

    for j in range(n_items):
        col = X[:, j]
        others = np.delete(X, j, axis=1)

        # variance
        col_nonan = col[~np.isnan(col)]
        var_j = float(np.var(col_nonan, ddof=1)) if col_nonan.size >= 2 else float("nan")

        # CITC
        mask_pair = (~np.isnan(col)) & (~np.isnan(others).any(axis=1))
        citc = _safe_corr(col[mask_pair], others[mask_pair].sum(axis=1))

        # --- CITC after reversing (simulation) ---
        citc_reversed = None
        citc_improves_after_reverse = False

        if np.isfinite(citc):
            col_rev = _reverse_item(col)
            mask_rev = (~np.isnan(col_rev)) & (~np.isnan(others).any(axis=1))
            if np.sum(mask_rev) >= 2:
                citc_reversed = _safe_corr(
                    col_rev[mask_rev],
                    others[mask_rev].sum(axis=1),
                )
                if np.isfinite(citc_reversed) and citc_reversed > abs(citc):
                    citc_improves_after_reverse = True

        # alpha if deleted
        mask_del = ~np.isnan(others).any(axis=1)
        alpha_del, _ = _cronbach_alpha_pairwise(others[mask_del])

        # --- flags / reasons ---
        flag = "ok"
        reasons: List[str] = []

        # delta vs overall
        delta_alpha_if_deleted = None
        remove_candidate = False
        if np.isfinite(alpha_del) and np.isfinite(overall_alpha):
            delta_alpha_if_deleted = float(alpha_del - overall_alpha)
            if delta_alpha_if_deleted > float(remove_delta):
                remove_candidate = True

        if np.isfinite(alpha_del) and np.isfinite(overall_alpha) and alpha_del > overall_alpha + 1e-9:
            flag = "problematic"
            reasons.append("alpha_if_deleted_gt_overall")

        if np.isfinite(citc) and citc < citc_threshold:
            if flag == "ok":
                flag = "review"
            reasons.append("low_item_total_corr")

        if np.isfinite(citc) and citc < 0:
            flag = "problematic"
            reasons.append("negative_item_total_corr")

        if np.isfinite(var_j) and var_j <= eps:
            if flag == "ok":
                flag = "review"
            reasons.append("near_zero_variance")

        # --- auto reverse detection (heuristic) ---
        suggest_reverse = False
        reverse_reason = None
        neg_citc = np.isfinite(citc) and citc < 0
        worsens_alpha = np.isfinite(alpha_del) and np.isfinite(overall_alpha) and (alpha_del > overall_alpha + 1e-9)

        if neg_citc and worsens_alpha and bool(citc_improves_after_reverse):
            suggest_reverse = True
            reverse_reason = "negative_citc_and_alpha_improves_when_deleted"
            reasons.append("possible_reverse_key")

        items.append(

            {
                "item_index": j,
                "item_name": item_names[j],
                "variance": var_j,
                "corrected_item_total_corr": citc,
                "citc_reversed": citc_reversed,
                "citc_improves_after_reverse": citc_improves_after_reverse,
                "alpha_if_deleted": alpha_del,
                "delta_alpha_if_deleted": delta_alpha_if_deleted,
                "remove_candidate": remove_candidate,
                "flag": flag,
                "reasons": reasons,
                "suggest_reverse": suggest_reverse,
                "reverse_reason": reverse_reason,
            }
        )

    # ---------- alpha after cleaning (simulation) ----------
    # Strategy: remove items that increase alpha when deleted
    removed = []
    for it in items:
        a_del = it.get("alpha_if_deleted")
        if np.isfinite(overall_alpha) and isinstance(a_del, (int, float)) and np.isfinite(a_del):
            if a_del > overall_alpha + 1e-9:
                removed.append(int(it["item_index"]))

    removed = sorted(set(removed))
    kept = [i for i in range(n_items) if i not in removed]

    alpha_after = None
    delta = None

    if len(kept) >= 2:
        X_kept = X[:, kept]
        try:
            alpha_after, _ = _cronbach_alpha_pairwise(X_kept)
        except Exception:
            alpha_after = float("nan")

        if np.isfinite(overall_alpha) and isinstance(alpha_after, (int, float)) and np.isfinite(alpha_after):
            delta = float(alpha_after - overall_alpha)

    cleaning = {
        "strategy": "remove_alpha_if_deleted_gt_overall",
        "removed_items": removed,
        "kept_items": kept,
        "alpha_before": float(overall_alpha) if np.isfinite(overall_alpha) else overall_alpha,
        "alpha_after": alpha_after,
        "delta": delta,
        "n_items_before": int(n_items),
        "n_items_after": int(len(kept)),
    }

    return {
        "overall_alpha": overall_alpha,
        "n_items": n_items,
        "items": items,
        "citc_threshold": citc_threshold,
        "remove_delta": remove_delta,
        "cleaning": cleaning,
    }


def cronbach_diagnostics_alpha_if_deleted(
    X: np.ndarray,
    *,
    overall_alpha: float,
    item_names: list[str] | None = None,
    remove_delta: float = 0.01,
) -> dict:
    """
    Build diagnostics including Alpha-if-deleted and remove-candidates.
    Returns JSON-serializable dict.
    """
    X = np.asarray(X, dtype=float)
    n_objects, n_items = X.shape

    # item names fallback
    if not isinstance(item_names, list) or len(item_names) != n_items:
        item_names = [f"Q{i+1}" for i in range(n_items)]

    rows: list[dict] = []
    alpha_if_deleted: list[float | None] = []

    for j in range(n_items):
        Xj = np.delete(X, j, axis=1)  # remove column j

        # NaN handling: listwise per-computation (simple & robust)
        # drop rows that have any NaN in remaining columns
        mask = ~np.isnan(Xj).any(axis=1)
        Xj_clean = Xj[mask]

        a_del = float("nan")
        if Xj_clean.shape[0] >= 2 and Xj_clean.shape[1] >= 2:
            # якщо у тебе є своя функція альфи (NaN-safe), виклич її тут
            a_del = _cronbach_alpha_pairwise(Xj_clean)

        alpha_if_deleted.append(None if np.isnan(a_del) else float(a_del))

    for j in range(n_items):
        a_del = alpha_if_deleted[j]
        delta = None if a_del is None or np.isnan(overall_alpha) else float(a_del - float(overall_alpha))
        remove_candidate = (delta is not None) and (delta > float(remove_delta))

        rows.append(
            {
                "item_index": j,
                "item_name": item_names[j],
                "item_name": item_names[j],
                "alpha_total": None if np.isnan(overall_alpha) else float(overall_alpha),
                "alpha_if_deleted": a_del,
                "delta_alpha_if_deleted": delta,
                "remove_candidate": bool(remove_candidate),
            }
        )

    # summary
    deltas = [r["delta_alpha_if_deleted"] for r in rows if r["delta_alpha_if_deleted"] is not None]
    best = None
    if deltas:
        best_i = int(np.nanargmax(np.array([d if d is not None else -1e9 for d in deltas])))
        # best_i here is index within filtered list; simpler: compute directly
        best = max(rows, key=lambda r: (r["delta_alpha_if_deleted"] is not None, r["delta_alpha_if_deleted"] or -1e9))

    return {
        "alpha_total": None if np.isnan(overall_alpha) else float(overall_alpha),
        "remove_delta": float(remove_delta),
        "table": rows,
        "best_remove_candidate": None if best is None else {
            "item_index": best["item_index"],
            "item_name": best["item_name"],
            "alpha_if_deleted": best["alpha_if_deleted"],
            "delta_alpha_if_deleted": best["delta_alpha_if_deleted"],
        },
    }


# =========================
# Estimator
# =========================

class CronbachAlpha(BaseReliabilityEstimator):
    def __init__(self, *, nan_policy: str = "pairwise") -> None:
        super().__init__()
        if nan_policy != "pairwise":
            raise ValueError("Only 'pairwise' nan_policy is supported.")
        self.nan_policy = nan_policy

    def fit(self, X) -> "CronbachAlpha":
        # Preserve item names if a pandas DataFrame is provided
        item_names = None
        if hasattr(X, "to_numpy") and hasattr(X, "columns"):
            try:
                item_names = [str(c) for c in list(X.columns)]
                X = X.to_numpy(dtype=float)
            except Exception:
                # Fallback: treat as array-like
                item_names = None

        X = np.asarray(X, dtype=float)
        has_missing = bool(np.isnan(X).any())
        if has_missing:
            self.warnings_.append("Missing values detected")

        if X.ndim != 2:
            raise ValueError("X must be 2D: (n_objects, n_items).")

        n_objects, n_items = X.shape
        has_missing = bool(np.isnan(X).any())

        if n_items < 2:
            raise ValueError("Cronbach's alpha requires at least 2 items.")

        alpha, details = _cronbach_alpha_pairwise(X)

        self.fit_info_ = FitInfo(
            n_objects=n_objects,
            n_items=n_items,
            n_raters=1,
            has_missing=has_missing,
        )

        self.score_ = float(alpha)

        # existing details
        self.details_ = {
            "n_objects_used": details.n_objects_used,
            "n_items": details.n_items,
            "total_score_variance": details.total_score_variance,
            "mean_inter_item_cov": details.mean_inter_item_cov,
            "mean_item_variance": details.mean_item_variance,
            "item_variances": details.item_variances.tolist(),
        }

        if item_names is not None:
            self.details_["item_names"] = item_names

        self.diagnostics_ = cronbach_diagnostics(X, overall_alpha=self.score_, item_names=item_names)

        # --- Reverse-coding simulation (no data modification, just a what-if alpha) ---
        try:
            items = self.diagnostics_.get("items", []) if isinstance(self.diagnostics_, dict) else []
            reverse_idxs = [
                int(it.get("item_index"))
                for it in items
                if isinstance(it, dict) and bool(it.get("suggest_reverse"))
            ]
            reverse_names = [
                (items[i].get("item_name") if i < len(items) else None) for i in reverse_idxs
            ]
            # Clean names list
            reverse_names_clean: List[str] = [str(n) for n in reverse_names if n is not None]

            alpha_after_reverse = None
            if reverse_idxs:
                X_rev = np.asarray(X, dtype=float).copy()
                for j in reverse_idxs:
                    if 0 <= j < X_rev.shape[1]:
                        X_rev[:, j] = _reverse_item(X_rev[:, j])
                alpha_after_reverse, _ = _cronbach_alpha_pairwise(X_rev)

            # Store into diagnostics for JSON/Excel export
            # Store into diagnostics for JSON/Excel export
            if isinstance(self.diagnostics_, dict):
                # --- Remove-items simulation (what-if alpha after dropping candidates) ---
                remove_idxs: List[int] = [
                    int(it.get("item_index"))
                    for it in items
                    if isinstance(it, dict) and bool(it.get("remove_candidate"))
                ]
                remove_idxs = sorted(set([i for i in remove_idxs if 0 <= i < X.shape[1]]))
                remove_names = [
                    (items[i].get("item_name") if i < len(items) else None) for i in remove_idxs
                ]
                remove_names_clean: List[str] = [str(n) for n in remove_names if n is not None]

                alpha_after_remove = None
                if remove_idxs and X.shape[1] - len(remove_idxs) >= 2:
                    keep = [j for j in range(X.shape[1]) if j not in set(remove_idxs)]
                    X_rm = np.asarray(X, dtype=float)[:, keep]
                    alpha_after_remove, _ = _cronbach_alpha_pairwise(X_rm)
                # --- Reverse + Remove simulation (reverse candidates, then remove candidates) ---
                alpha_after_reverse_and_remove = None
                try:
                    # X_rev має існувати, якщо reverse_idxs є (у тебе вже є reverse-sim)
                    X_tmp = X_rev if reverse_idxs else np.asarray(X, dtype=float)

                    if remove_idxs:
                        # після видалення має лишитись >=2 items, інакше alpha не має сенсу
                        if X_tmp.shape[1] - len(remove_idxs) >= 2:
                            keep = [j for j in range(X_tmp.shape[1]) if j not in set(remove_idxs)]
                            X_rrm = X_tmp[:, keep]
                            alpha_after_reverse_and_remove, _ = _cronbach_alpha_pairwise(X_rrm)
                    else:
                        # якщо нема що видаляти — цей сценарій = reverse only
                        alpha_after_reverse_and_remove = alpha_after_reverse
                except Exception:
                    alpha_after_reverse_and_remove = None

                self.diagnostics_["simulation"] = {
                    "alpha_original": float(self.score_) if self.score_ is not None else None,
                    "alpha_after_reverse_only": float(alpha_after_reverse) if alpha_after_reverse is not None else None,
                    "alpha_after_remove_only": float(alpha_after_remove) if alpha_after_remove is not None else None,
                    "reverse_candidate_indices": reverse_idxs,
                    "reverse_candidate_items": reverse_names_clean,
                    "remove_candidate_indices": remove_idxs,
                    "remove_candidate_items": remove_names_clean,
                    "remove_delta": float(self.diagnostics_.get("params", {}).get("remove_delta", 0.01)),
                    "alpha_after_reverse_and_remove": float(
                        alpha_after_reverse_and_remove) if alpha_after_reverse_and_remove is not None else None,

                }
        except Exception:
            # Never fail fit() due to simulation helpers
            pass

        # expose diagnostics via details for JSON/Excel export
        self.details_["diagnostics"] = self.diagnostics_

        return self
