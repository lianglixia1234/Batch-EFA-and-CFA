# 模块4: Multi-Factor CFA - 多因子验证性因子分析（数据集合并 + 多因子相关模型）
"""
承接「数据清洗」模块生成的子数据集，提供数据集合并功能，
并支持对合并数据用 Multi-factor correlated model 做 CFA：
潜变量之间自由相关，潜变量与 method 正交，method 方差固定为 1。
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from datetime import date
from typing import List, Optional, Tuple, Dict, Any

# semopy 延迟导入：在实际使用时才加载，减少启动时间
try:
    from db_save import save_formula_params, build_formula_params_json
    _DB_SAVE_AVAILABLE = True
except ImportError:
    save_formula_params = build_formula_params_json = None
    _DB_SAVE_AVAILABLE = False
# from semopy import Model — 延迟到函数内导入

from modules.n2_analysis import calculate_advanced_stats
from modules.n1_analysis import cronbach_alpha
from modules.utils import smart_multiselect, parse_item_col, sort_item_cols_by_number

# ==============================================================================
# 工具函数
# ==============================================================================

def get_dataframe_status(df: pd.DataFrame) -> str:
    """
    判断子数据集状态：Ready（可合并）或 Incomplete（不完整）。
    Ready: 至少 10 行、至少 1 列、均为数值列、无整列缺失。
    """
    if df is None or df.empty:
        return "Incomplete"
    n_rows, n_cols = df.shape
    if n_rows < 10 or n_cols < 1:
        return "Incomplete"
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < n_cols:
        return "Incomplete"
    if numeric.isna().all(axis=0).any():
        return "Incomplete"
    return "Ready"


def get_dataframe_date(name: str) -> str:
    """从 session 中获取子数据集的更新时间，无则返回占位符。"""
    if "sub_datasets_updated" not in st.session_state:
        return "—"
    ts = st.session_state.sub_datasets_updated.get(name)
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


def merge_dataframes(names: List[str], sub_datasets: dict) -> Optional[pd.DataFrame]:
    """
    将选中的多个子数据集按行索引内连接合并（相同样本保留）。
    列名若重复则加前缀（数据集名）避免冲突。
    """
    if not names:
        return None
    dfs = []
    for name in names:
        if name not in sub_datasets:
            continue
        df = sub_datasets[name].copy()
        df.columns = [f"{name}__{c}" for c in df.columns]
        dfs.append(df)
    if not dfs:
        return None
    merged = pd.concat(dfs, axis=1, join="inner")
    return merged


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """读取上传的 Excel/CSV 文件为 DataFrame。"""
    file_name = str(getattr(uploaded_file, "name", "")).lower()
    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    return pd.read_excel(uploaded_file)


def _default_measure_name_from_filename(file_name: str, fallback_idx: int) -> str:
    """从文件名提取默认 measure 名称。"""
    base = re.sub(r"\.[^.]+$", "", str(file_name or "")).strip()
    base = re.sub(r"\s+", " ", base)
    return base or f"Measure{fallback_idx}"


def _sanitize_name(s: str) -> str:
    """将列名/因子名转为 semopy 可用的标识符（仅保留字母、数字、下划线，其余替换为下划线）。"""
    return re.sub(r"[^\w]+", "_", str(s)).strip("_") or "x"


def _is_reverse_method_item(col_name: str) -> bool:
    """仅当题目文本去尾部空白/标点后以 r 结尾时，判定为反向题。"""
    _, _, text = parse_item_col(str(col_name))
    s = (text or str(col_name)).strip()
    s = s.replace("ｒ", "r").replace("Ｒ", "R")
    s = re.sub(r"""[\s\u3000\)\]）】》〉'"“”’`~!@#$%^&*+=|\\/:;,.?，。！？、；：-]+$""", "", s)
    return s.lower().endswith("r")


def _reset_smart_multiselect_cache(key_suffix: str) -> None:
    """清理 smart_multiselect 的 checkbox/选择缓存，强制按最新默认值重建。"""
    cb_prefix = f"cb_{key_suffix}_"
    for k in list(st.session_state.keys()):
        if k.startswith(cb_prefix):
            del st.session_state[k]
    for k in (f"{key_suffix}_last_selected", f"{key_suffix}_control_action"):
        st.session_state.pop(k, None)


def _get_merge_df_and_names(select_key: str = "n3_cfa_merge_select") -> Tuple[Optional[pd.DataFrame], List[str]]:
    """返回当前选中的合并数据与对应的子数据集名称顺序。"""
    selected_id = st.session_state.get(select_key, "")
    if not selected_id:
        return None, []
    if selected_id == "__current__":
        df = st.session_state.get("n3_merged_df")
        names = st.session_state.get("n3_merged_names", [])
        return df, names
    saved = st.session_state.get("n3_saved_merges", {})
    if selected_id not in saved:
        return None, []
    rec = saved[selected_id]
    return rec["df"], rec["names"]


def run_multifactor_cfa(
    df: pd.DataFrame,
    dataset_names_in_order: List[str],
    method_item_columns: List[str],
) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[Dict], Optional[str], str]:
    """
    多因子相关模型 CFA：每个子数据集 = 一个潜变量，method 因子由用户指定题目。
    潜变量之间自由相关，潜变量与 method 正交，method 方差固定为 1。
    返回 (model, estimates_df, stats_dict, error_msg, model_syntax)。
    """
    if df is None or df.empty or not dataset_names_in_order:
        return None, None, None, "无合并数据或因子列表为空。", ""

    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
    if df_clean.empty:
        return None, None, None, "清理后无有效样本。", ""

    orig_cols = list(df_clean.columns)
    safe_map = {c: _sanitize_name(c) for c in orig_cols}
    df_safe = df_clean.rename(columns=safe_map)
    safe_cols = list(df_safe.columns)

    factor_prefixes = [_sanitize_name(n) + "__" for n in dataset_names_in_order]
    factor_cols: List[List[str]] = []
    for prefix in factor_prefixes:
        cols = [c for c in safe_cols if c.startswith(prefix)]
        if not cols:
            return None, None, None, f"未找到前缀 {prefix!r} 对应的题目列。", ""
        # 按题号稳定排序，确保 marker（首题）可复现
        cols = sort_item_cols_by_number(cols)
        factor_cols.append(cols)

    # 方案A：session 内 method 题目按“原始列名”存储，建模时统一映射到 safe 列名
    method_items = []
    for m in method_item_columns or []:
        if m in safe_map:          # 原始列名
            method_items.append(safe_map[m])
        elif m in safe_cols:       # 已经是 safe 列名（兼容）
            method_items.append(m)
    method_items = list(dict.fromkeys(method_items))
    if not method_items:
        return None, None, None, "请至少选择 1 个方法因子题目后再运行 Multi-Factor CFA。", ""

    factor_names = [f"F{i+1}" for i in range(len(factor_cols))]
    lines = []
    # 显式 marker-variable：每个潜变量首题载荷固定为 1，潜变量方差自由估计
    for fname, cols in zip(factor_names, factor_cols):
        first, rest = cols[0], cols[1:]
        if rest:
            lines.append(f"{fname} =~ 1*{first} + {' + '.join(rest)}")
        else:
            lines.append(f"{fname} =~ 1*{first}")
    if method_items:
        lines.append(f"method =~ {' + '.join(method_items)}")
        for fname in factor_names:
            lines.append(f"{fname} ~~ 0*method")
        lines.append("method ~~ 1*method")

    model_desc = "\n".join(lines)
    used_cols = []
    for cols in factor_cols:
        used_cols.extend(cols)
    used_cols.extend(method_items)
    used_cols = list(dict.fromkeys(used_cols))
    df_fit = df_safe[used_cols].copy()
    # 稳健处理：强制转数值，避免 semopy 在 np.isnan(object) 上报错
    df_fit = df_fit.apply(pd.to_numeric, errors="coerce")
    bad_cols = [c for c in df_fit.columns if df_fit[c].isna().all()]
    if bad_cols:
        return None, None, None, f"以下题目列无法转换为数值（全为缺失）：{', '.join(bad_cols[:8])}", model_desc
    df_fit = df_fit.dropna(axis=0, how="any")
    if df_fit.empty:
        return None, None, None, "数值化后无有效样本（存在缺失/非数值）。", model_desc

    import semopy
    from semopy import Model
    try:
        model = Model(model_desc)
        model.fit(df_fit)
    except Exception as e:
        return None, None, None, f"模型拟合失败: {e}", model_desc

    try:
        estimates = model.inspect(std_est=True)
        estimates = estimates.rename(columns={
            "lval": "LHS", "op": "op", "rval": "RHS",
            "Estimate": "Estimate", "Std. Err": "Std.Err",
            "z-value": "z-value", "p-value": "P(>|z|)",
            "Est. Std": "Std.all",
        })
        fit_stats = semopy.calc_stats(model)
        stats_dict = fit_stats.iloc[0].to_dict() if not fit_stats.empty else {}
        n = len(df_fit)
        stats_dict = calculate_advanced_stats(model, fit_stats, n, df_fit)
    except Exception as e:
        return model, None, None, f"统计量计算失败: {e}", model_desc

    return model, estimates, stats_dict, None, model_desc


def run_higherorder_cfa(
    df: pd.DataFrame,
    dataset_names_in_order: List[str],
    method_item_columns: List[str],
    higher_order_name: str = "HigherOrderFactor",
) -> Tuple[Optional[Any], Optional[pd.DataFrame], Optional[Dict], Optional[str], str]:
    """
    高阶因子模型 CFA：
    - 一阶因子 F1, F2, F3... = 各子数据集的题目；每个一阶因子的首题载荷固定为 1。
    - 高阶因子 =~ 1*F1 + F2 + F3...（对 F1 的载荷固定为 1），高阶因子方差自由估计。
    - method 因子由用户指定题目，与各一阶因子及高阶因子正交，method 方差固定为 1。
    返回 (model, estimates_df, stats_dict, error_msg, model_syntax)。
    """
    if df is None or df.empty or not dataset_names_in_order:
        return None, None, None, "无合并数据或因子列表为空。", ""

    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
    if df_clean.empty:
        return None, None, None, "清理后无有效样本。", ""

    orig_cols = list(df_clean.columns)
    safe_map = {c: _sanitize_name(c) for c in orig_cols}
    df_safe = df_clean.rename(columns=safe_map)
    safe_cols = list(df_safe.columns)

    factor_prefixes = [_sanitize_name(n) + "__" for n in dataset_names_in_order]
    factor_cols: List[List[str]] = []
    for prefix in factor_prefixes:
        cols = [c for c in safe_cols if c.startswith(prefix)]
        if not cols:
            return None, None, None, f"未找到前缀 {prefix!r} 对应的题目列。", ""
        # 按题号稳定排序，确保 marker（首题）可复现
        cols = sort_item_cols_by_number(cols)
        factor_cols.append(cols)

    # 方案A：session 内 method 题目按“原始列名”存储，建模时统一映射到 safe 列名
    method_items = []
    for m in method_item_columns or []:
        if m in safe_map:          # 原始列名
            method_items.append(safe_map[m])
        elif m in safe_cols:       # 已经是 safe 列名（兼容）
            method_items.append(m)
    method_items = list(dict.fromkeys(method_items))
    if not method_items:
        return None, None, None, "请至少选择 1 个方法因子题目后再运行 Higher-order CFA。", ""

    factor_names = [f"F{i+1}" for i in range(len(factor_cols))]
    hof = _sanitize_name(higher_order_name) or "HOF"
    lines = []
    # 一阶因子：每个因子的首题载荷固定为 1
    for fname, cols in zip(factor_names, factor_cols):
        first, rest = cols[0], cols[1:]
        if rest:
            lines.append(f"{fname} =~ 1*{first} + {' + '.join(rest)}")
        else:
            lines.append(f"{fname} =~ 1*{first}")
    # 高阶因子：对第一个一阶因子的载荷固定为 1，方差自由估计
    lines.append(f"{hof} =~ 1*{factor_names[0]} + {' + '.join(factor_names[1:])}")
    if method_items:
        lines.append(f"method =~ {' + '.join(method_items)}")
        for fname in factor_names:
            lines.append(f"{fname} ~~ 0*method")
        lines.append(f"{hof} ~~ 0*method")
        lines.append("method ~~ 1*method")

    model_desc = "\n".join(lines)
    used_cols = []
    for cols in factor_cols:
        used_cols.extend(cols)
    used_cols.extend(method_items)
    used_cols = list(dict.fromkeys(used_cols))
    df_fit = df_safe[used_cols].copy()
    # 稳健处理：强制转数值，避免 semopy 在 np.isnan(object) 上报错
    df_fit = df_fit.apply(pd.to_numeric, errors="coerce")
    bad_cols = [c for c in df_fit.columns if df_fit[c].isna().all()]
    if bad_cols:
        return None, None, None, f"以下题目列无法转换为数值（全为缺失）：{', '.join(bad_cols[:8])}", model_desc
    df_fit = df_fit.dropna(axis=0, how="any")
    if df_fit.empty:
        return None, None, None, "数值化后无有效样本（存在缺失/非数值）。", model_desc

    import semopy
    from semopy import Model
    try:
        model = Model(model_desc)
        model.fit(df_fit)
    except Exception as e:
        return None, None, None, f"模型拟合失败: {e}", model_desc

    try:
        estimates = model.inspect(std_est=True)
        estimates = estimates.rename(columns={
            "lval": "LHS", "op": "op", "rval": "RHS",
            "Estimate": "Estimate", "Std. Err": "Std.Err",
            "z-value": "z-value", "p-value": "P(>|z|)",
            "Est. Std": "Std.all",
        })
        fit_stats = semopy.calc_stats(model)
        stats_dict = fit_stats.iloc[0].to_dict() if not fit_stats.empty else {}
        n = len(df_fit)
        stats_dict = calculate_advanced_stats(model, fit_stats, n, df_fit)
    except Exception as e:
        return model, None, None, f"统计量计算失败: {e}", model_desc

    return model, estimates, stats_dict, None, model_desc


def _to_num(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x in ("", "-", "nan", "NaN", "None"):
                return np.nan
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _get_any_stat(stats_dict: Dict[str, Any], keys: List[str]) -> float:
    norm = {re.sub(r"[^a-z0-9]+", "", str(k).lower()): v for k, v in (stats_dict or {}).items()}
    for k in keys:
        if k in stats_dict:
            v = _to_num(stats_dict[k])
            if not np.isnan(v):
                return v
    for k in keys:
        nk = re.sub(r"[^a-z0-9]+", "", k.lower())
        if nk in norm:
            v = _to_num(norm[nk])
            if not np.isnan(v):
                return v
    return np.nan


def _split_prefixed_col(col: str) -> Tuple[str, str]:
    if "__" in str(col):
        p, b = str(col).split("__", 1)
        return p + "__", b
    return "", str(col)


def _safe_filename_part(name: str, fallback: str) -> str:
    """清洗文件名片段（保留 +），避免跨平台非法字符问题。"""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(name or "")).strip(" .")
    return s or fallback


def _cr_from_std_loadings(lambda_std: np.ndarray) -> float:
    """CR = (sum(lambda_std)^2) / ((sum(lambda_std)^2) + sum(1-lambda_std^2))."""
    if lambda_std is None:
        return np.nan
    arr = np.asarray(lambda_std, dtype=float).reshape(-1)
    if arr.size == 0 or np.isnan(arr).any():
        return np.nan
    S = float(np.sum(arr))
    E = float(np.sum(1.0 - arr ** 2))
    den = (S ** 2) + E
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float((S ** 2) / den)


def _build_factor_cov_matrix_from_estimates(estimates: pd.DataFrame, factor_names: List[str]) -> np.ndarray:
    """从估计结果提取潜变量协方差矩阵 Phi（含方差与协方差）。"""
    m = len(factor_names)
    phi = np.full((m, m), np.nan, dtype=float)
    idx = {f: i for i, f in enumerate(factor_names)}
    if estimates is None or estimates.empty or not {"LHS", "RHS", "op", "Estimate"}.issubset(estimates.columns):
        return phi
    vv = estimates[estimates["op"] == "~~"]
    for _, r in vv.iterrows():
        l = str(r["LHS"])
        rr = str(r["RHS"])
        if l in idx and rr in idx:
            i, j = idx[l], idx[rr]
            v = _to_num(r.get("Estimate", np.nan))
            phi[i, j] = v
            phi[j, i] = v
    return phi


def _build_multifactor_report_tables(
    cfa_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    stats_dict: Dict[str, Any],
    latent_display_names: List[str],
    measuregroup_title: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    构建多因子相关模型下载表：
    1) 每题一行（含每个潜变量的标准化/非标准化载荷）
    2) 潜变量方差-协方差矩阵
    3) 题目方差-协方差矩阵
    """
    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]
    factor_to_display = {f: latent_display_names[i] for i, f in enumerate(factor_names)}

    # 原列 -> 安全列映射（需与 run_multifactor_cfa 一致）
    safe_map = {c: _sanitize_name(c) for c in cfa_df.columns}
    safe_to_orig = {v: k for k, v in safe_map.items()}

    # 因子对应题目（原列名）
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in cfa_df.columns if str(c).startswith(prefix)]
        # 按题号排序（以去前缀后的题目名排序）
        items_sorted = sorted(items, key=lambda x: (
            parse_item_col(_split_prefixed_col(x)[1])[1] is None,
            parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
            _split_prefixed_col(x)[1]
        ))
        factor_items_orig[factor_names[i]] = items_sorted

    # 载荷映射：兼容 =~ 与 ~ 两种输出
    load_unstd: Dict[Tuple[str, str], float] = {}
    load_std: Dict[Tuple[str, str], float] = {}
    if estimates is not None and not estimates.empty and {"LHS", "op", "RHS"}.issubset(estimates.columns):
        eq_rows = estimates[(estimates["op"] == "=~") & (estimates["LHS"].isin(factor_names))]
        for _, r in eq_rows.iterrows():
            f = str(r["LHS"])
            item_safe = str(r["RHS"])
            load_unstd[(f, item_safe)] = _to_num(r.get("Estimate", np.nan))
            load_std[(f, item_safe)] = _to_num(r.get("Std.all", np.nan))
        reg_rows = estimates[(estimates["op"] == "~") & (estimates["RHS"].isin(factor_names))]
        for _, r in reg_rows.iterrows():
            f = str(r["RHS"])
            item_safe = str(r["LHS"])
            if (f, item_safe) not in load_unstd:
                load_unstd[(f, item_safe)] = _to_num(r.get("Estimate", np.nan))
                load_std[(f, item_safe)] = _to_num(r.get("Std.all", np.nan))

    # Composite Reliability (CR_G): 对 overall equal-weight composite 计算一次并复用到每题行
    cr_overall = np.nan
    cr_reason = ""
    try:
        ordered_item_cols: List[str] = []
        ordered_item_factor: List[str] = []
        for f in factor_names:
            for col in factor_items_orig.get(f, []):
                ordered_item_cols.append(col)
                ordered_item_factor.append(f)
        if not ordered_item_cols:
            cr_reason = "CR 未计算：未找到用于 CR 的题目列。"
        else:
            x_cr = cfa_df[ordered_item_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
            if x_cr.empty:
                cr_reason = "CR 未计算：用于 CR 的有效样本为空（题目存在缺失/非数值）。"
            else:
                sigma = x_cr.cov().values
                s_vec = np.sqrt(np.diag(sigma))
                phi = _build_factor_cov_matrix_from_estimates(estimates, factor_names)
                if phi.shape[0] != len(factor_names) or not np.all(np.isfinite(phi)):
                    cr_reason = "CR 未计算：潜变量协方差矩阵 Φ 无法从估计结果中完整提取。"
                elif (not np.all(np.isfinite(s_vec))) or np.any(s_vec <= 0):
                    cr_reason = "CR 未计算：题目标准差（来自协方差矩阵对角线）存在无效值。"
                else:
                    lambda_mat = np.zeros((len(ordered_item_cols), len(factor_names)), dtype=float)
                    f_idx = {f: i for i, f in enumerate(factor_names)}
                    miss_items = []
                    for i, (col, f) in enumerate(zip(ordered_item_cols, ordered_item_factor)):
                        safe_col = safe_map.get(col, _sanitize_name(col))
                        lv = _to_num(load_unstd.get((f, safe_col), np.nan))
                        if np.isnan(lv):
                            miss_items.append(col)
                            continue
                        lambda_mat[i, f_idx[f]] = lv
                    if miss_items:
                        cr_reason = f"CR 未计算：以下题目缺少主因子非标准化载荷：{', '.join([str(x) for x in miss_items[:6]])}"
                    else:
                        m = len(factor_names)
                        a = np.full((m, 1), 1.0 / m, dtype=float)
                        var_g = float((a.T @ phi @ a).item())
                        if not np.isfinite(var_g) or var_g <= 0:
                            cr_reason = "CR 未计算：equal-weight composite 的方差 Var(G) 非正或无效。"
                        else:
                            cov_xg = (lambda_mat @ phi @ a).reshape(-1)
                            lambda_std_g = cov_xg / (s_vec * np.sqrt(var_g))
                            cr_overall = _cr_from_std_loadings(lambda_std_g)
                            if np.isnan(cr_overall):
                                cr_reason = "CR 未计算：标准化载荷生成后分母无效。"
    except Exception as cr_e:
        cr_overall = np.nan
        cr_reason = f"CR 未计算：计算过程异常（{cr_e}）。"

    fit_vals = {
        "chi2_user_model": _get_any_stat(stats_dict, ["chi2", "Chi2"]),
        "df_user_model": _get_any_stat(stats_dict, ["DoF", "dof", "df"]),
        "p_value_user_model": _get_any_stat(stats_dict, ["chi2 p-value", "p-value", "pvalue", "p_value"]),
        "CFI": _get_any_stat(stats_dict, ["CFI"]),
        "TLI": _get_any_stat(stats_dict, ["TLI"]),
        "RMSEA": _get_any_stat(stats_dict, ["RMSEA"]),
        "SRMR": _get_any_stat(stats_dict, ["SRMR", "srmr"]),
        "GFI": _get_any_stat(stats_dict, ["GFI"]),
        "AGFI": _get_any_stat(stats_dict, ["AGFI"]),
        "NFI": _get_any_stat(stats_dict, ["NFI"]),
        "LogL": _get_any_stat(stats_dict, ["LogL", "logl", "LogLik", "loglik", "log_likelihood", "log-likelihood"]),
        "AIC": _get_any_stat(stats_dict, ["AIC"]),
        "BIC": _get_any_stat(stats_dict, ["BIC"]),
        "SABIC": _get_any_stat(stats_dict, ["SABIC"]),
    }

    rows = []
    for f in factor_names:
        items = factor_items_orig.get(f, [])
        # 同一潜变量 Cronbach alpha 相同
        factor_numeric = cfa_df[items].apply(pd.to_numeric, errors="coerce") if items else pd.DataFrame()
        alpha = cronbach_alpha(factor_numeric) if not factor_numeric.empty else np.nan
        for col in items:
            prefix, base = _split_prefixed_col(col)
            pre, num, text = parse_item_col(base)
            item_number_efa = num if pre == "EFA" else np.nan
            if pre is None and num is None:
                m = re.search(r"(\d+)", base.split("_", 1)[0])
                num_fallback = int(m.group(1)) if m else np.nan
                item_number_efa = num_fallback
            item_text = text or base
            reverse = 1 if _is_reverse_method_item(base) else 0

            row = {
                "measuregroup_title": measuregroup_title,
                "measure_id": factor_to_display[f],
                "item_number_EFA": item_number_efa,
                "item_text": item_text,
                "reverse": reverse,
                **fit_vals,
                "item_mean": _to_num(pd.to_numeric(cfa_df[col], errors="coerce").mean()),
                "item_sd": _to_num(pd.to_numeric(cfa_df[col], errors="coerce").std()),
                "cronbach_alpha": _to_num(alpha),
                "Composite Reliability (CR)": _to_num(cr_overall),
            }
            safe_col = safe_map.get(col, _sanitize_name(col))
            for fx in factor_names:
                disp = factor_to_display[fx]
                row[f"unstandardised_loading_{disp}"] = load_unstd.get((fx, safe_col), np.nan)
                row[f"standardised_loading_{disp}"] = load_std.get((fx, safe_col), np.nan)
            rows.append(row)

    items_sheet = pd.DataFrame(rows)
    if not items_sheet.empty:
        for c in ("item_number_EFA",):
            if c in items_sheet.columns:
                items_sheet[c] = pd.to_numeric(items_sheet[c], errors="coerce")
        sort_cols = [c for c in ("item_number_EFA",) if c in items_sheet.columns]
        if sort_cols:
            items_sheet = items_sheet.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    # 潜变量方差-协方差矩阵
    cov_df = pd.DataFrame(np.nan, index=latent_display_names, columns=latent_display_names)
    if estimates is not None and not estimates.empty and {"LHS", "op", "RHS", "Estimate"}.issubset(estimates.columns):
        vv = estimates[estimates["op"] == "~~"].copy()
        for _, r in vv.iterrows():
            l = str(r["LHS"])
            rr = str(r["RHS"])
            if l in factor_names and rr in factor_names:
                dl, dr = factor_to_display[l], factor_to_display[rr]
                cov_df.loc[dl, dr] = _to_num(r.get("Estimate", np.nan))
                cov_df.loc[dr, dl] = _to_num(r.get("Estimate", np.nan))

    # 题目方差-协方差矩阵（按潜变量顺序与题号升序）
    ordered_item_cols: List[str] = []
    ordered_item_labels: List[str] = []
    for f in factor_names:
        for col in factor_items_orig.get(f, []):
            _, base = _split_prefixed_col(col)
            _, num, text = parse_item_col(base)
            if num is None:
                m = re.search(r"(\d+)", base.split("_", 1)[0])
                num = int(m.group(1)) if m else np.nan
            label = f"{factor_to_display[f]}::{int(num)}_{(text or base)}" if not pd.isna(num) else f"{factor_to_display[f]}::{(text or base)}"
            ordered_item_cols.append(col)
            ordered_item_labels.append(label)
    item_cov_df = pd.DataFrame()
    if ordered_item_cols:
        item_numeric = cfa_df[ordered_item_cols].apply(pd.to_numeric, errors="coerce")
        item_cov_df = item_numeric.cov()
        if len(item_cov_df.index) == len(ordered_item_labels):
            item_cov_df.index = ordered_item_labels
            item_cov_df.columns = ordered_item_labels

    return items_sheet, cov_df, item_cov_df, cr_reason


def _build_higherorder_report_tables(
    ho_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    stats_dict: Dict[str, Any],
    first_order_display_names: List[str],
    measuregroup_title: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    构建 Higher-order 模型下载表：
    - sheet1: 每题一行（含各一阶潜变量载荷、拟合指标、均值方差、alpha）
    - sheet2: 每个一阶潜变量一行（题号串/题目串、对高阶因子的载荷、高阶因子方差）
    - sheet3: 一阶潜变量方差-协方差矩阵
    - sheet4: 题目方差-协方差矩阵
    """
    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]
    factor_to_display = {f: first_order_display_names[i] for i, f in enumerate(factor_names)}

    safe_map = {c: _sanitize_name(c) for c in ho_df.columns}
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in ho_df.columns if str(c).startswith(prefix)]
        items = sorted(
            items,
            key=lambda x: (
                parse_item_col(_split_prefixed_col(x)[1])[1] is None,
                parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
                _split_prefixed_col(x)[1],
            ),
        )
        factor_items_orig[factor_names[i]] = items

    # 检测高阶因子名（兼容 =~ 与 ~ 两种方向）
    hof_name = "HigherOrderFactor"
    if estimates is not None and not estimates.empty and {"LHS", "RHS", "op"}.issubset(estimates.columns):
        cand = estimates[(estimates["op"] == "=~") & (estimates["RHS"].isin(factor_names))]
        if not cand.empty:
            vals = [str(x) for x in cand["LHS"].tolist() if str(x) not in factor_names]
            if vals:
                hof_name = max(set(vals), key=vals.count)
        else:
            cand2 = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names))]
            vals2 = [str(x) for x in cand2["RHS"].tolist() if str(x) not in factor_names]
            if vals2:
                hof_name = max(set(vals2), key=vals2.count)

    # 题目->一阶潜变量载荷映射（兼容 =~ 与 ~）
    load_unstd: Dict[Tuple[str, str], float] = {}
    load_std: Dict[Tuple[str, str], float] = {}
    if estimates is not None and not estimates.empty and {"LHS", "op", "RHS"}.issubset(estimates.columns):
        eq_rows = estimates[(estimates["op"] == "=~") & (estimates["LHS"].isin(factor_names))]
        for _, r in eq_rows.iterrows():
            f = str(r["LHS"])
            item_safe = str(r["RHS"])
            load_unstd[(f, item_safe)] = _to_num(r.get("Estimate", np.nan))
            load_std[(f, item_safe)] = _to_num(r.get("Std.all", np.nan))
        reg_rows = estimates[(estimates["op"] == "~") & (estimates["RHS"].isin(factor_names))]
        for _, r in reg_rows.iterrows():
            f = str(r["RHS"])
            item_safe = str(r["LHS"])
            if (f, item_safe) not in load_unstd:
                load_unstd[(f, item_safe)] = _to_num(r.get("Estimate", np.nan))
                load_std[(f, item_safe)] = _to_num(r.get("Std.all", np.nan))

    fit_vals = {
        "chi2_user_model": _get_any_stat(stats_dict, ["chi2", "Chi2"]),
        "df_user_model": _get_any_stat(stats_dict, ["DoF", "dof", "df"]),
        "p_value_user_model": _get_any_stat(stats_dict, ["chi2 p-value", "p-value", "pvalue", "p_value"]),
        "CFI": _get_any_stat(stats_dict, ["CFI"]),
        "TLI": _get_any_stat(stats_dict, ["TLI"]),
        "RMSEA": _get_any_stat(stats_dict, ["RMSEA"]),
        "SRMR": _get_any_stat(stats_dict, ["SRMR", "srmr"]),
        "GFI": _get_any_stat(stats_dict, ["GFI"]),
        "AGFI": _get_any_stat(stats_dict, ["AGFI"]),
        "NFI": _get_any_stat(stats_dict, ["NFI"]),
        "LogL": _get_any_stat(stats_dict, ["LogL", "logl", "LogLik", "loglik", "log_likelihood", "log-likelihood"]),
        "AIC": _get_any_stat(stats_dict, ["AIC"]),
        "BIC": _get_any_stat(stats_dict, ["BIC"]),
        "SABIC": _get_any_stat(stats_dict, ["SABIC"]),
    }

    # 高阶因子方差
    higher_var = np.nan
    if estimates is not None and not estimates.empty and {"LHS", "RHS", "op"}.issubset(estimates.columns):
        vv = estimates[(estimates["op"] == "~~") & (estimates["LHS"] == estimates["RHS"]) & (estimates["LHS"] == hof_name)]
        if not vv.empty:
            higher_var = _to_num(vv.iloc[0].get("Estimate", np.nan))

    # 一阶因子 -> 高阶因子载荷
    ho_unstd: Dict[str, float] = {f: np.nan for f in factor_names}
    ho_std: Dict[str, float] = {f: np.nan for f in factor_names}
    if estimates is not None and not estimates.empty and {"LHS", "RHS", "op"}.issubset(estimates.columns):
        a = estimates[(estimates["op"] == "=~") & (estimates["LHS"] == hof_name) & (estimates["RHS"].isin(factor_names))]
        for _, r in a.iterrows():
            f = str(r["RHS"])
            ho_unstd[f] = _to_num(r.get("Estimate", np.nan))
            ho_std[f] = _to_num(r.get("Std.all", np.nan))
        if a.empty:
            b = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names)) & (estimates["RHS"] == hof_name)]
            for _, r in b.iterrows():
                f = str(r["LHS"])
                ho_unstd[f] = _to_num(r.get("Estimate", np.nan))
                ho_std[f] = _to_num(r.get("Std.all", np.nan))

    # Composite Reliability (CR_H): 对 higher-order 目标 H 计算一次并复用到每题行
    cr_higher = np.nan
    cr_reason = ""
    try:
        ordered_item_cols: List[str] = []
        ordered_item_factor: List[str] = []
        for f in factor_names:
            for col in factor_items_orig.get(f, []):
                ordered_item_cols.append(col)
                ordered_item_factor.append(f)
        if not ordered_item_cols:
            cr_reason = "CR 未计算：未找到用于 CR 的题目列。"
        else:
            x_cr = ho_df[ordered_item_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
            if x_cr.empty:
                cr_reason = "CR 未计算：用于 CR 的有效样本为空（题目存在缺失/非数值）。"
            else:
                sigma = x_cr.cov().values
                s_vec = np.sqrt(np.diag(sigma))
                gamma = np.array([_to_num(ho_unstd.get(f, np.nan)) for f in factor_names], dtype=float).reshape(-1, 1)
                phi_h = _to_num(higher_var)
                if np.isnan(phi_h) or phi_h <= 0:
                    cr_reason = "CR 未计算：高阶因子方差 φ_H 缺失或非正数。"
                elif np.isnan(gamma).any():
                    cr_reason = "CR 未计算：一阶因子到高阶因子的载荷 γ 缺失。"
                elif (not np.all(np.isfinite(s_vec))) or np.any(s_vec <= 0):
                    cr_reason = "CR 未计算：题目标准差（来自协方差矩阵对角线）存在无效值。"
                else:
                    lambda_1 = np.zeros((len(ordered_item_cols), len(factor_names)), dtype=float)
                    f_idx = {f: i for i, f in enumerate(factor_names)}
                    miss_items = []
                    for i, (col, f) in enumerate(zip(ordered_item_cols, ordered_item_factor)):
                        safe_col = safe_map.get(col, _sanitize_name(col))
                        lv = _to_num(load_unstd.get((f, safe_col), np.nan))
                        if np.isnan(lv):
                            miss_items.append(col)
                            continue
                        lambda_1[i, f_idx[f]] = lv
                    if miss_items:
                        cr_reason = f"CR 未计算：以下题目缺少主因子非标准化载荷：{', '.join([str(x) for x in miss_items[:6]])}"
                    else:
                        lambda_h = (lambda_1 @ gamma).reshape(-1)
                        lambda_std_h = (lambda_h * np.sqrt(phi_h)) / s_vec
                        cr_higher = _cr_from_std_loadings(lambda_std_h)
                        if np.isnan(cr_higher):
                            cr_reason = "CR 未计算：标准化载荷生成后分母无效。"
    except Exception as cr_e:
        cr_higher = np.nan
        cr_reason = f"CR 未计算：计算过程异常（{cr_e}）。"

    # sheet1: 一题一行
    sheet1_rows = []
    for f in factor_names:
        items = factor_items_orig.get(f, [])
        fac_num = ho_df[items].apply(pd.to_numeric, errors="coerce") if items else pd.DataFrame()
        alpha = cronbach_alpha(fac_num) if not fac_num.empty else np.nan
        for col in items:
            _, base = _split_prefixed_col(col)
            pre, num, text = parse_item_col(base)
            item_num = num
            if item_num is None:
                m = re.search(r"(\d+)", base.split("_", 1)[0])
                item_num = int(m.group(1)) if m else np.nan
            item_text = text or base
            reverse = 1 if _is_reverse_method_item(base) else 0
            row = {
                "measuregroup_title": measuregroup_title,
                "measure_id": factor_to_display[f],
                "item_number": item_num,
                "item_text": item_text,
                "reverse": reverse,
                **fit_vals,
                "item_mean": _to_num(pd.to_numeric(ho_df[col], errors="coerce").mean()),
                "item_sd": _to_num(pd.to_numeric(ho_df[col], errors="coerce").std()),
                "cronbach_alpha": _to_num(alpha),
                "Composite Reliability (CR)": _to_num(cr_higher),
            }
            safe_col = safe_map.get(col, _sanitize_name(col))
            for fx in factor_names:
                disp = factor_to_display[fx]
                row[f"unstandardised_loading_{disp}"] = load_unstd.get((fx, safe_col), np.nan)
                row[f"standardised_loading_{disp}"] = load_std.get((fx, safe_col), np.nan)
            sheet1_rows.append(row)

    sheet1 = pd.DataFrame(sheet1_rows)
    if not sheet1.empty and "item_number" in sheet1.columns:
        sheet1["item_number"] = pd.to_numeric(sheet1["item_number"], errors="coerce")
        sheet1 = sheet1.sort_values(["item_number"], na_position="last").reset_index(drop=True)

    # sheet2: 一阶潜变量汇总（一因子一行）
    sheet2_rows = []
    for f in factor_names:
        items = factor_items_orig.get(f, [])
        nums = []
        texts = []
        for col in items:
            _, base = _split_prefixed_col(col)
            _, num, text = parse_item_col(base)
            if num is None:
                m = re.search(r"(\d+)", base.split("_", 1)[0])
                num = int(m.group(1)) if m else np.nan
            nums.append("" if np.isnan(num) else str(int(num)))
            texts.append((text or base).strip())
        sheet2_rows.append({
            "measuregroup_title": measuregroup_title,
            "first_order_latent": factor_to_display[f],
            "items_number": "%%".join([x for x in nums if x != ""]),
            "items_text": "%%".join(texts),
            "unstandardised_loading_on_higher_order": ho_unstd.get(f, np.nan),
            "standardised_loading_on_higher_order": ho_std.get(f, np.nan),
            "variance_higher_order": higher_var,
        })
    sheet2 = pd.DataFrame(sheet2_rows)

    # sheet3: 一阶潜变量方差-协方差矩阵
    # NOTE: 按需求暂时注释掉“潜变量方差-协方差矩阵”的生成逻辑，仅保留代码便于后续恢复。
    # sheet3 = pd.DataFrame(np.nan, index=first_order_display_names, columns=first_order_display_names)
    # if estimates is not None and not estimates.empty and {"LHS", "RHS", "op", "Estimate"}.issubset(estimates.columns):
    #     vv = estimates[estimates["op"] == "~~"].copy()
    #     for _, r in vv.iterrows():
    #         l = str(r["LHS"])
    #         rr = str(r["RHS"])
    #         if l in factor_names and rr in factor_names:
    #             dl, dr = factor_to_display[l], factor_to_display[rr]
    #             sheet3.loc[dl, dr] = _to_num(r.get("Estimate", np.nan))
    #             sheet3.loc[dr, dl] = _to_num(r.get("Estimate", np.nan))
    sheet3 = pd.DataFrame()

    # sheet4: 题目方差-协方差矩阵（按一阶潜变量顺序与题号升序）
    ordered_item_cols: List[str] = []
    ordered_item_labels: List[str] = []
    for f in factor_names:
        for col in factor_items_orig.get(f, []):
            _, base = _split_prefixed_col(col)
            _, num, text = parse_item_col(base)
            if num is None:
                m = re.search(r"(\d+)", base.split("_", 1)[0])
                num = int(m.group(1)) if m else np.nan
            label = f"{factor_to_display[f]}::{int(num)}_{(text or base)}" if not pd.isna(num) else f"{factor_to_display[f]}::{(text or base)}"
            ordered_item_cols.append(col)
            ordered_item_labels.append(label)
    sheet4 = pd.DataFrame()
    if ordered_item_cols:
        item_numeric = ho_df[ordered_item_cols].apply(pd.to_numeric, errors="coerce")
        sheet4 = item_numeric.cov()
        if len(sheet4.index) == len(ordered_item_labels):
            sheet4.index = ordered_item_labels
            sheet4.columns = ordered_item_labels

    return sheet1, sheet2, sheet3, sheet4, cr_reason


def _compute_higherorder_final_scores(
    ho_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    scale_min: int = 1,
    scale_max: int = 7,
) -> Tuple[pd.DataFrame, int]:
    """
    基于 Higher-order CFA（使用非标准化载荷）计算每个样本最终得分：
    1) λ_H = Λ * γ
    2) 解线性方程 Σw = λ_H * φ_H（不直接求逆）
    3) h_raw = w'(x-μ), z = h_raw / sqrt(w'Σw), FinalScore = 80 + 10*z
    缺失值策略：任一题缺失的样本，最终得分记为 -999。
    返回：(带“最终得分”列的数据, 缺失样本行数)
    """
    if ho_df is None or ho_df.empty:
        raise ValueError("无可用于计算最终得分的数据。")
    if not dataset_names_in_order:
        raise ValueError("一阶潜变量列表为空，无法计算最终得分。")
    if estimates is None or estimates.empty or not {"LHS", "op", "RHS", "Estimate"}.issubset(estimates.columns):
        raise ValueError("缺少 Higher-order CFA 参数估计结果（Estimate）。")

    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]

    # 1) 按模型因子顺序收集并排序题目，保证 μ/Σ/Λ/x 顺序一致
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in ho_df.columns if str(c).startswith(prefix)]
        items = sorted(
            items,
            key=lambda x: (
                parse_item_col(_split_prefixed_col(x)[1])[1] is None,
                parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
                _split_prefixed_col(x)[1],
            ),
        )
        if not items:
            raise ValueError(f"未找到潜变量 {factor_names[i]} 对应题目列（前缀 {prefix}）。")
        factor_items_orig[factor_names[i]] = items

    item_order_raw: List[str] = []
    for f in factor_names:
        item_order_raw.extend(factor_items_orig[f])
    if not item_order_raw:
        raise ValueError("未提取到用于计分的题目列。")

    # 2) 数值化并统计缺失
    x_num = ho_df[item_order_raw].apply(pd.to_numeric, errors="coerce")
    missing_mask = x_num.isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    x_valid = x_num.loc[~missing_mask].copy()
    if x_valid.empty:
        raise ValueError("所有样本均存在缺失，无法计算有效最终得分。")

    mu_vec = x_valid.mean(axis=0).values.reshape(-1, 1)
    sigma = x_valid.cov().values

    # 原列名 -> safe 列名（与 run_higherorder_cfa 一致）
    safe_map = {c: _sanitize_name(c) for c in item_order_raw}
    safe_items = [safe_map[c] for c in item_order_raw]
    safe_set = set(safe_items)
    item_idx = {c: i for i, c in enumerate(safe_items)}
    factor_idx = {f: i for i, f in enumerate(factor_names)}

    # 3) 题目 -> 一阶因子非标准化载荷 Λ
    # 仅对“该题所属的一阶因子”填值；其余交叉载荷按 0 处理
    load_map: Dict[Tuple[str, str], float] = {}
    load_rows = estimates[estimates["op"].isin(["=~", "~"])]
    for _, r in load_rows.iterrows():
        lhs, rhs = str(r["LHS"]), str(r["RHS"])
        est_val = _to_num(r.get("Estimate", np.nan))
        if np.isnan(est_val):
            continue
        if lhs in factor_names and rhs in safe_set:
            load_map[(lhs, rhs)] = est_val
        elif rhs in factor_names and lhs in safe_set:
            load_map[(rhs, lhs)] = est_val

    lambda_fs = np.zeros((len(item_order_raw), len(factor_names)), dtype=float)
    for f in factor_names:
        items_f = factor_items_orig[f]
        for j, raw_col in enumerate(items_f):
            safe_col = safe_map[raw_col]
            v = load_map.get((f, safe_col), np.nan)
            if np.isnan(v):
                # marker 项固定 1；部分输出不会显式给出该估计
                if j == 0:
                    v = 1.0
                else:
                    raise ValueError(f"未能提取载荷：{f} -> {raw_col}")
            lambda_fs[item_idx[safe_col], factor_idx[f]] = v

    # 4) 一阶因子 -> 高阶因子非标准化载荷 γ，与高阶因子方差 φ_H
    hof_name = "HigherOrderFactor"
    cand = estimates[(estimates["op"] == "=~") & (estimates["RHS"].isin(factor_names))]
    if not cand.empty:
        vals = [str(x) for x in cand["LHS"].tolist() if str(x) not in factor_names]
        if vals:
            hof_name = max(set(vals), key=vals.count)
    else:
        cand2 = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names))]
        vals2 = [str(x) for x in cand2["RHS"].tolist() if str(x) not in factor_names]
        if vals2:
            hof_name = max(set(vals2), key=vals2.count)

    gamma_map: Dict[str, float] = {}
    a = estimates[(estimates["op"] == "=~") & (estimates["LHS"] == hof_name) & (estimates["RHS"].isin(factor_names))]
    for _, r in a.iterrows():
        gamma_map[str(r["RHS"])] = _to_num(r.get("Estimate", np.nan))
    if a.empty:
        b = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names)) & (estimates["RHS"] == hof_name)]
        for _, r in b.iterrows():
            gamma_map[str(r["LHS"])] = _to_num(r.get("Estimate", np.nan))

    gamma = np.zeros((len(factor_names), 1), dtype=float)
    for i, f in enumerate(factor_names):
        g = _to_num(gamma_map.get(f, np.nan))
        if np.isnan(g):
            # 高阶因子 marker 路径固定为 1
            if i == 0:
                g = 1.0
            else:
                raise ValueError(f"未能提取一阶因子到高阶因子的载荷：{f} -> {hof_name}")
        gamma[i, 0] = g

    vv = estimates[(estimates["op"] == "~~") & (estimates["LHS"] == estimates["RHS"]) & (estimates["LHS"] == hof_name)]
    if vv.empty:
        raise ValueError("未能提取高阶因子方差（H ~~ H 的 Estimate）。")
    phi_h = _to_num(vv.iloc[0].get("Estimate", np.nan))
    if np.isnan(phi_h):
        raise ValueError("高阶因子方差无有效数值。")

    # 5) 回归权重 + 样本得分
    lambda_h = lambda_fs @ gamma
    rhs = lambda_h * phi_h
    try:
        w = np.linalg.solve(sigma, rhs)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(sigma.shape[0])
        w = np.linalg.solve(sigma + ridge, rhs)

    x_centered = x_valid.values - mu_vec.T
    h_raw = (x_centered @ w).flatten()
    sd_h = float(np.sqrt(np.maximum((w.T @ sigma @ w).item(), 1e-12)))
    z_h = h_raw / sd_h
    final_scores_valid = 80 + 10 * z_h

    # Absolute 0-100 final score（按 Higher-order_CFA_Absolute_0-100_Scoring_Instructions，复用 UI scale_min/scale_max）
    w_flat = w.flatten()
    l_abs = float(scale_min)
    u_abs = float(scale_max)
    lo_abs = np.where(w_flat >= 0, l_abs, u_abs)
    hi_abs = np.where(w_flat >= 0, u_abs, l_abs)
    s_min_abs = float(w_flat @ lo_abs)
    s_max_abs = float(w_flat @ hi_abs)
    den_abs = s_max_abs - s_min_abs
    if den_abs <= 1e-12:
        raise ValueError("absolute 最终得分计算失败：S_max 与 S_min 过近，无法线性映射。")
    scale_abs = 100.0 / den_abs
    beta_star_abs = scale_abs * w_flat
    intercept_star_abs = -scale_abs * s_min_abs
    absolute_final_scores_valid = (x_valid.values @ beta_star_abs) + intercept_star_abs

    scored_df = ho_df.copy()
    scored_df["relative_final_score"] = -999.0
    scored_df.loc[x_valid.index, "relative_final_score"] = final_scores_valid
    scored_df["absolute_final_score"] = -999.0
    scored_df.loc[x_valid.index, "absolute_final_score"] = absolute_final_scores_valid
    # 保证 "absolute_final_score" 紧接在 "relative_final_score" 右侧
    cols = [c for c in scored_df.columns if c != "absolute_final_score"]
    idx_after = cols.index("relative_final_score") + 1
    scored_df = scored_df[cols[:idx_after] + ["absolute_final_score"] + cols[idx_after:]]
    return scored_df, n_missing


def _compute_multifactor_final_scores(
    cfa_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    scale_min: int = 1,
    scale_max: int = 7,
) -> Tuple[pd.DataFrame, int]:
    """
    基于 Multi-factor Correlated Model（使用非标准化载荷）计算：
    - Output A：每个实质潜变量一列报告分（80 + 10*z_k）
    - Output B：Overall 最终得分一列（等权合成后重标准化，80 + 10*z_comp,std）
    缺失值策略：任一题缺失的样本，其所有得分列统一记为 -999。
    返回：(带得分列的数据, 缺失样本行数)
    """
    if cfa_df is None or cfa_df.empty:
        raise ValueError("无可用于计算最终得分的数据。")
    if not dataset_names_in_order:
        raise ValueError("潜变量列表为空，无法计算最终得分。")
    if estimates is None or estimates.empty or not {"LHS", "op", "RHS", "Estimate"}.issubset(estimates.columns):
        raise ValueError("缺少 Multi-factor CFA 参数估计结果（Estimate）。")

    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]

    # 按模型因子顺序收集题目，保证 μ/Σ/Λ/x 顺序一致
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in cfa_df.columns if str(c).startswith(prefix)]
        items = sorted(
            items,
            key=lambda x: (
                parse_item_col(_split_prefixed_col(x)[1])[1] is None,
                parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
                _split_prefixed_col(x)[1],
            ),
        )
        if not items:
            raise ValueError(f"未找到潜变量 {factor_names[i]} 对应题目列（前缀 {prefix}）。")
        factor_items_orig[factor_names[i]] = items

    item_order_raw: List[str] = []
    for f in factor_names:
        item_order_raw.extend(factor_items_orig[f])
    if not item_order_raw:
        raise ValueError("未提取到用于计分的题目列。")

    x_num = cfa_df[item_order_raw].apply(pd.to_numeric, errors="coerce")
    missing_mask = x_num.isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    x_valid = x_num.loc[~missing_mask].copy()
    if x_valid.empty:
        raise ValueError("所有样本均存在缺失，无法计算有效最终得分。")

    mu_vec = x_valid.mean(axis=0).values.reshape(-1, 1)
    sigma = x_valid.cov().values

    safe_map = {c: _sanitize_name(c) for c in item_order_raw}
    safe_items = [safe_map[c] for c in item_order_raw]
    safe_set = set(safe_items)
    item_idx = {c: i for i, c in enumerate(safe_items)}
    factor_idx = {f: i for i, f in enumerate(factor_names)}

    # 题目 -> 潜变量非标准化载荷 Λ（method 不参与）
    load_map: Dict[Tuple[str, str], float] = {}
    load_rows = estimates[estimates["op"].isin(["=~", "~"])]
    for _, r in load_rows.iterrows():
        lhs, rhs = str(r["LHS"]), str(r["RHS"])
        est_val = _to_num(r.get("Estimate", np.nan))
        if np.isnan(est_val):
            continue
        if lhs in factor_names and rhs in safe_set:
            load_map[(lhs, rhs)] = est_val
        elif rhs in factor_names and lhs in safe_set:
            load_map[(rhs, lhs)] = est_val

    lambda_mat = np.zeros((len(item_order_raw), len(factor_names)), dtype=float)
    for f in factor_names:
        items_f = factor_items_orig[f]
        for j, raw_col in enumerate(items_f):
            safe_col = safe_map[raw_col]
            v = load_map.get((f, safe_col), np.nan)
            if np.isnan(v):
                # marker 项固定为 1（部分输出不会显式给该参数）
                if j == 0:
                    v = 1.0
                else:
                    raise ValueError(f"未能提取载荷：{f} -> {raw_col}")
            lambda_mat[item_idx[safe_col], factor_idx[f]] = v

    # 潜变量协方差矩阵 Φ（仅实质潜变量）
    phi = np.full((len(factor_names), len(factor_names)), np.nan, dtype=float)
    vv = estimates[estimates["op"] == "~~"].copy()
    for _, r in vv.iterrows():
        l = str(r["LHS"])
        rr = str(r["RHS"])
        if l in factor_names and rr in factor_names:
            i = factor_idx[l]
            j = factor_idx[rr]
            v = _to_num(r.get("Estimate", np.nan))
            if not np.isnan(v):
                phi[i, j] = v
                phi[j, i] = v
    for i in range(len(factor_names)):
        if np.isnan(phi[i, i]):
            raise ValueError(f"未能提取潜变量方差：{factor_names[i]} ~~ {factor_names[i]}")
    # 若个别协方差缺失，按 0 处理（常见于约束或输出省略）
    phi = np.nan_to_num(phi, nan=0.0)

    # Step 2: W = Σ^{-1}ΛΦ（通过线性方程求解）
    rhs = lambda_mat @ phi
    try:
        w = np.linalg.solve(sigma, rhs)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(sigma.shape[0])
        w = np.linalg.solve(sigma + ridge, rhs)

    # Step 3: f_raw = W' * (x-μ)
    x_centered = x_valid.values - mu_vec.T
    f_raw = x_centered @ w  # (n, m)

    # Step 4: z_k 标准化
    sigma_f = w.T @ sigma @ w
    sd_vec = np.sqrt(np.maximum(np.diag(sigma_f), 1e-12)).reshape(1, -1)
    z = f_raw / sd_vec

    # Output A: 每因子报告分
    final_per_factor = 80 + 10 * z

    # Output B: overall（等权重 + 重标准化）
    m = len(factor_names)
    z_comp = z.mean(axis=1, keepdims=True)
    d = np.maximum(np.diag(sigma_f), 1e-12)
    denom = np.sqrt(np.outer(d, d))
    r_z = sigma_f / denom
    a = np.full((m, 1), 1.0 / m)
    sd_comp = float(np.sqrt(np.maximum((a.T @ r_z @ a).item(), 1e-12)))
    z_comp_std = z_comp.flatten() / sd_comp
    final_overall = 80 + 10 * z_comp_std

    # Output C: overall absolute final score（0-100，严格按 absolute 文档，复用 UI scale_min/scale_max）
    # 先得到整体等权重 item 权重向量 a_item = B^T c；此处 B^T = W = Σ^{-1}ΛΦ
    abs_item_weights = (w @ a).flatten()  # (p,)
    L, U = float(scale_min), float(scale_max)
    lo = np.where(abs_item_weights >= 0, L, U)
    hi = np.where(abs_item_weights >= 0, U, L)
    s_min = float(abs_item_weights @ lo)
    s_max = float(abs_item_weights @ hi)
    den_abs = s_max - s_min
    if den_abs <= 1e-12:
        raise ValueError("absolute final score 计算失败：S_max 与 S_min 过近，无法线性映射。")
    s_raw = (x_valid.values @ abs_item_weights).flatten()
    final_absolute = 100.0 * (s_raw - s_min) / den_abs
    final_absolute = np.clip(final_absolute, 0.0, 100.0)

    scored_df = cfa_df.copy()
    factor_score_cols = [f"relative_final_score_{name}" for name in dataset_names_in_order]
    for col in factor_score_cols:
        scored_df[col] = -999.0
    scored_df["relative_final_score"] = -999.0
    scored_df["absolute_final_score"] = -999.0
    for i, col in enumerate(factor_score_cols):
        scored_df.loc[x_valid.index, col] = final_per_factor[:, i]
    scored_df.loc[x_valid.index, "relative_final_score"] = final_overall
    scored_df.loc[x_valid.index, "absolute_final_score"] = final_absolute
    return scored_df, n_missing


def _extract_item_num_and_text_from_prefixed(col_name: str) -> Tuple[Any, str]:
    _, base = _split_prefixed_col(str(col_name))
    pre, num, text = parse_item_col(base)
    item_num = num
    if item_num is None:
        m = re.search(r"(\d+)", base.split("_", 1)[0])
        item_num = int(m.group(1)) if m else np.nan
    return item_num, (text or base)


def _resolve_n3_dataset_label() -> str:
    if st.session_state.get("n3_dual_dataset"):
        return str(st.session_state.get("n3_dual_dataset"))
    names = st.session_state.get("n3_merged_names", [])
    if names:
        return "sub_merge:" + "|".join([str(x) for x in names])
    return "n3_dataset"


def _build_multifactor_formula_table(
    cfa_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    measuregroup_title: str,
    measure_ids_in_order: List[str],
    scale_min: int,
    scale_max: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]
    factor_to_measure_id = {
        f: (
            str(measure_ids_in_order[i]).strip()
            if i < len(measure_ids_in_order) and str(measure_ids_in_order[i]).strip()
            else str(dataset_names_in_order[i])
        )
        for i, f in enumerate(factor_names)
    }
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in cfa_df.columns if str(c).startswith(prefix)]
        items = sorted(
            items,
            key=lambda x: (
                parse_item_col(_split_prefixed_col(x)[1])[1] is None,
                parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
                _split_prefixed_col(x)[1],
            ),
        )
        if not items:
            raise ValueError(f"未找到潜变量 {factor_names[i]} 对应题目列。")
        factor_items_orig[factor_names[i]] = items

    item_order_raw: List[str] = []
    item_measure_id_map: Dict[str, str] = {}
    for f in factor_names:
        for col in factor_items_orig[f]:
            if col not in item_measure_id_map:
                item_order_raw.append(col)
                item_measure_id_map[col] = factor_to_measure_id[f]

    x_all = cfa_df[item_order_raw].apply(pd.to_numeric, errors="coerce")
    x = x_all.dropna(axis=0, how="any")
    if x.empty:
        raise ValueError("有效样本为空，无法导出公式参数。")
    mu_vec = x.mean(axis=0).values.reshape(-1, 1)
    sigma = x.cov().values

    safe_map = {c: _sanitize_name(c) for c in item_order_raw}
    safe_items = [safe_map[c] for c in item_order_raw]
    safe_set = set(safe_items)
    item_idx = {c: i for i, c in enumerate(safe_items)}
    factor_idx = {f: i for i, f in enumerate(factor_names)}

    load_map: Dict[Tuple[str, str], float] = {}
    load_rows = estimates[estimates["op"].isin(["=~", "~"])]
    for _, r in load_rows.iterrows():
        lhs, rhs = str(r["LHS"]), str(r["RHS"])
        est_val = _to_num(r.get("Estimate", np.nan))
        if np.isnan(est_val):
            continue
        if lhs in factor_names and rhs in safe_set:
            load_map[(lhs, rhs)] = est_val
        elif rhs in factor_names and lhs in safe_set:
            load_map[(rhs, lhs)] = est_val

    lambda_mat = np.zeros((len(item_order_raw), len(factor_names)), dtype=float)
    for f in factor_names:
        items_f = factor_items_orig[f]
        for j, raw_col in enumerate(items_f):
            safe_col = safe_map[raw_col]
            v = load_map.get((f, safe_col), np.nan)
            if np.isnan(v):
                v = 1.0 if j == 0 else np.nan
            if np.isnan(v):
                raise ValueError(f"未能提取载荷：{f} -> {raw_col}")
            lambda_mat[item_idx[safe_col], factor_idx[f]] = v

    phi = np.full((len(factor_names), len(factor_names)), np.nan, dtype=float)
    vv = estimates[estimates["op"] == "~~"].copy()
    for _, r in vv.iterrows():
        l = str(r["LHS"])
        rr = str(r["RHS"])
        if l in factor_names and rr in factor_names:
            i, j = factor_idx[l], factor_idx[rr]
            v = _to_num(r.get("Estimate", np.nan))
            if not np.isnan(v):
                phi[i, j] = v
                phi[j, i] = v
    for i in range(len(factor_names)):
        if np.isnan(phi[i, i]):
            raise ValueError(f"未能提取潜变量方差：{factor_names[i]} ~~ {factor_names[i]}")
    phi = np.nan_to_num(phi, nan=0.0)

    rhs = lambda_mat @ phi
    try:
        w = np.linalg.solve(sigma, rhs)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(sigma.shape[0])
        w = np.linalg.solve(sigma + ridge, rhs)

    sigma_f = w.T @ sigma @ w
    d = np.maximum(np.diag(sigma_f), 1e-12)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    m = len(factor_names)
    a = np.full((m, 1), 1.0 / m)
    b = w @ d_inv_sqrt @ a
    r_z = d_inv_sqrt @ sigma_f @ d_inv_sqrt
    sd_comp = float(np.sqrt(np.maximum((a.T @ r_z @ a).item(), 1e-12)))

    reporting_mean = 80.0
    reporting_sd = 10.0
    beta_star = (reporting_sd / sd_comp) * b.flatten()
    intercept_star = float(reporting_mean - float(np.sum(beta_star * mu_vec.flatten())))

    # absolute 参数（0-100）：复用界面输入的 scale_min / scale_max 作为理论边界
    abs_item_weights = (w @ a).flatten()  # a_item = B^T c，B^T=W
    l_abs = float(scale_min)
    u_abs = float(scale_max)
    lo = np.where(abs_item_weights >= 0, l_abs, u_abs)
    hi = np.where(abs_item_weights >= 0, u_abs, l_abs)
    s_min = float(abs_item_weights @ lo)
    s_max = float(abs_item_weights @ hi)
    den_abs = s_max - s_min
    if den_abs <= 1e-12:
        raise ValueError("absolute 参数计算失败：S_max 与 S_min 过近，无法线性映射。")
    beta_abs = (100.0 / den_abs) * abs_item_weights
    intercept_abs = float(-(100.0 / den_abs) * s_min)

    dataset_label = _resolve_n3_dataset_label()
    measure_id_main = str(measure_ids_in_order[0]).strip() if measure_ids_in_order else str(dataset_names_in_order[0])
    created_date = date.today().strftime("%Y-%m-%d")
    formula_measure_df = pd.DataFrame([{
        "measure_id": measure_id_main,
        "measuregroup_title": str(measuregroup_title),
        "dataset": dataset_label,
        "CFA_model": "multi-factor",
        "intercept_rel": float(intercept_star),
        "intercept_abs": intercept_abs,
        "reporting_mean": reporting_mean,
        "reporting_sd": reporting_sd,
        "scale_min": int(scale_min),
        "scale_max": int(scale_max),
        "创建时间": created_date,
    }])
    formula_items_rows = []
    for i, col in enumerate(item_order_raw):
        item_num, item_text = _extract_item_num_and_text_from_prefixed(col)
        formula_items_rows.append({
            "measure_id": str(item_measure_id_map.get(col, "")),
            "item_col": str(col),
            "item_text": item_text,
            "item_num": item_num,
            "reverse": 1 if _is_reverse_method_item(col) else 0,
            "beta_rel": float(beta_star[i]),
            "beta_abs": float(beta_abs[i]),
            "sort_order": i + 1,
            "创建时间": created_date,
        })
    formula_items_df = pd.DataFrame(formula_items_rows)
    return formula_measure_df, formula_items_df


def _build_higherorder_formula_table(
    ho_df: pd.DataFrame,
    dataset_names_in_order: List[str],
    estimates: pd.DataFrame,
    measuregroup_title: str,
    measure_ids_in_order: List[str],
    scale_min: int,
    scale_max: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_names = [f"F{i+1}" for i in range(len(dataset_names_in_order))]
    factor_to_measure_id = {
        f: (
            str(measure_ids_in_order[i]).strip()
            if i < len(measure_ids_in_order) and str(measure_ids_in_order[i]).strip()
            else str(dataset_names_in_order[i])
        )
        for i, f in enumerate(factor_names)
    }
    factor_items_orig: Dict[str, List[str]] = {}
    for i, name in enumerate(dataset_names_in_order):
        prefix = _sanitize_name(name) + "__"
        items = [c for c in ho_df.columns if str(c).startswith(prefix)]
        items = sorted(
            items,
            key=lambda x: (
                parse_item_col(_split_prefixed_col(x)[1])[1] is None,
                parse_item_col(_split_prefixed_col(x)[1])[1] or 10**9,
                _split_prefixed_col(x)[1],
            ),
        )
        if not items:
            raise ValueError(f"未找到潜变量 {factor_names[i]} 对应题目列。")
        factor_items_orig[factor_names[i]] = items

    item_order_raw: List[str] = []
    item_measure_id_map: Dict[str, str] = {}
    for f in factor_names:
        for col in factor_items_orig[f]:
            if col not in item_measure_id_map:
                item_order_raw.append(col)
                item_measure_id_map[col] = factor_to_measure_id[f]

    x_all = ho_df[item_order_raw].apply(pd.to_numeric, errors="coerce")
    x = x_all.dropna(axis=0, how="any")
    if x.empty:
        raise ValueError("有效样本为空，无法导出公式参数。")
    mu_vec = x.mean(axis=0).values.reshape(-1, 1)
    sigma = x.cov().values

    safe_map = {c: _sanitize_name(c) for c in item_order_raw}
    safe_items = [safe_map[c] for c in item_order_raw]
    safe_set = set(safe_items)
    item_idx = {c: i for i, c in enumerate(safe_items)}
    factor_idx = {f: i for i, f in enumerate(factor_names)}

    load_map: Dict[Tuple[str, str], float] = {}
    load_rows = estimates[estimates["op"].isin(["=~", "~"])]
    for _, r in load_rows.iterrows():
        lhs, rhs = str(r["LHS"]), str(r["RHS"])
        est_val = _to_num(r.get("Estimate", np.nan))
        if np.isnan(est_val):
            continue
        if lhs in factor_names and rhs in safe_set:
            load_map[(lhs, rhs)] = est_val
        elif rhs in factor_names and lhs in safe_set:
            load_map[(rhs, lhs)] = est_val

    lambda_fs = np.zeros((len(item_order_raw), len(factor_names)), dtype=float)
    for f in factor_names:
        items_f = factor_items_orig[f]
        for j, raw_col in enumerate(items_f):
            safe_col = safe_map[raw_col]
            v = load_map.get((f, safe_col), np.nan)
            if np.isnan(v):
                v = 1.0 if j == 0 else np.nan
            if np.isnan(v):
                raise ValueError(f"未能提取载荷：{f} -> {raw_col}")
            lambda_fs[item_idx[safe_col], factor_idx[f]] = v

    hof_name = "HigherOrderFactor"
    cand = estimates[(estimates["op"] == "=~") & (estimates["RHS"].isin(factor_names))]
    if not cand.empty:
        vals = [str(x) for x in cand["LHS"].tolist() if str(x) not in factor_names]
        if vals:
            hof_name = max(set(vals), key=vals.count)
    else:
        cand2 = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names))]
        vals2 = [str(x) for x in cand2["RHS"].tolist() if str(x) not in factor_names]
        if vals2:
            hof_name = max(set(vals2), key=vals2.count)

    gamma_map: Dict[str, float] = {}
    a = estimates[(estimates["op"] == "=~") & (estimates["LHS"] == hof_name) & (estimates["RHS"].isin(factor_names))]
    for _, r in a.iterrows():
        gamma_map[str(r["RHS"])] = _to_num(r.get("Estimate", np.nan))
    if a.empty:
        b = estimates[(estimates["op"] == "~") & (estimates["LHS"].isin(factor_names)) & (estimates["RHS"] == hof_name)]
        for _, r in b.iterrows():
            gamma_map[str(r["LHS"])] = _to_num(r.get("Estimate", np.nan))

    gamma = np.zeros((len(factor_names), 1), dtype=float)
    for i, f in enumerate(factor_names):
        g = _to_num(gamma_map.get(f, np.nan))
        if np.isnan(g):
            g = 1.0 if i == 0 else np.nan
        if np.isnan(g):
            raise ValueError(f"未能提取高阶载荷：{f} -> {hof_name}")
        gamma[i, 0] = g

    vv = estimates[(estimates["op"] == "~~") & (estimates["LHS"] == estimates["RHS"]) & (estimates["LHS"] == hof_name)]
    if vv.empty:
        raise ValueError("未能提取高阶因子方差。")
    phi_h = _to_num(vv.iloc[0].get("Estimate", np.nan))
    if np.isnan(phi_h):
        raise ValueError("高阶因子方差无有效数值。")

    lambda_h = lambda_fs @ gamma
    rhs = lambda_h * phi_h
    try:
        w = np.linalg.solve(sigma, rhs)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(sigma.shape[0])
        w = np.linalg.solve(sigma + ridge, rhs)
    sd_h = float(np.sqrt(np.maximum((w.T @ sigma @ w).item(), 1e-12)))

    reporting_mean = 80.0
    reporting_sd = 10.0
    beta_star = (reporting_sd / sd_h) * w.flatten()
    intercept_star = float(reporting_mean - float(np.sum(beta_star * mu_vec.flatten())))

    # Absolute 0-100 参数（按 Higher-order_CFA_Absolute_0-100_Scoring_Instructions）
    w_flat = w.flatten()
    l_abs = float(scale_min)
    u_abs = float(scale_max)
    lo_abs = np.where(w_flat >= 0, l_abs, u_abs)
    hi_abs = np.where(w_flat >= 0, u_abs, l_abs)
    s_min_abs = float(w_flat @ lo_abs)
    s_max_abs = float(w_flat @ hi_abs)
    den_abs = s_max_abs - s_min_abs
    if den_abs <= 1e-12:
        raise ValueError("absolute 公式参数计算失败：S_max 与 S_min 过近，无法线性映射。")
    beta_abs = (100.0 / den_abs) * w_flat
    intercept_abs = float(-(100.0 / den_abs) * s_min_abs)

    dataset_label = _resolve_n3_dataset_label()
    measure_id_main = str(measure_ids_in_order[0]).strip() if measure_ids_in_order else str(dataset_names_in_order[0])
    created_date = date.today().strftime("%Y-%m-%d")
    formula_measure_df = pd.DataFrame([{
        "measure_id": measure_id_main,
        "measuregroup_title": str(measuregroup_title),
        "dataset": dataset_label,
        "CFA_model": "higher-order",
        "intercept_rel": float(intercept_star),
        "intercept_abs": intercept_abs,
        "reporting_mean": reporting_mean,
        "reporting_sd": reporting_sd,
        "scale_min": int(scale_min),
        "scale_max": int(scale_max),
        "创建时间": created_date,
    }])
    formula_items_rows = []
    for i, col in enumerate(item_order_raw):
        item_num, item_text = _extract_item_num_and_text_from_prefixed(col)
        formula_items_rows.append({
            "measure_id": str(item_measure_id_map.get(col, "")),
            "item_col": str(col),
            "item_text": item_text,
            "item_num": item_num,
            "reverse": 1 if _is_reverse_method_item(col) else 0,
            "beta_rel": float(beta_star[i]),
            "beta_abs": float(beta_abs[i]),
            "sort_order": i + 1,
            "创建时间": created_date,
        })
    formula_items_df = pd.DataFrame(formula_items_rows)
    return formula_measure_df, formula_items_df


# ==============================================================================
# 页面渲染
# ==============================================================================

def render_n3_multifactor_cfa():
    st.title("模块 4: Multi-Factor CFA")

    if "n3_merged_df" not in st.session_state:
        st.session_state.n3_merged_df = None
    if "n3_merged_source" not in st.session_state:
        st.session_state.n3_merged_source = ""
    if "n3_merged_names" not in st.session_state:
        st.session_state.n3_merged_names = []
    if "n3_saved_merges" not in st.session_state:
        st.session_state.n3_saved_merges = {}
    if "n3_cfa_merge_select" not in st.session_state:
        st.session_state.n3_cfa_merge_select = ""
    if "n3_ho_merge_select" not in st.session_state:
        st.session_state.n3_ho_merge_select = ""
    if "n3_model_tab" not in st.session_state:
        st.session_state.n3_model_tab = "Multi-factor Correlated Model"
    if "n3_model_tab_next" in st.session_state:
        st.session_state.n3_model_tab = st.session_state.n3_model_tab_next
        del st.session_state.n3_model_tab_next

    has_sub = "sub_datasets" in st.session_state and len(st.session_state.sub_datasets) > 0
    has_dual_data = (
        st.session_state.get("dc_merge_done")
        and st.session_state.get("dc_dataset_full")
        and st.session_state.get("dc_measures")
    )
    sub_datasets = st.session_state.sub_datasets if has_sub else {}

    # 数据来源：四数据集 / 子数据集合并 / 上传文件
    source_options = []
    if has_dual_data:
        source_options.append("四数据集 (Dataset → Measure)")
    if has_sub:
        source_options.append("子数据集合并 (Sub-datasets)")
    source_options.append("上传文件 (Upload files)")

    if len(source_options) == 1:
        data_source_choice = source_options[0]
    else:
        data_source_choice = st.radio("数据来源", source_options, key="n3_data_source")

    if has_dual_data and data_source_choice == "四数据集 (Dataset → Measure)":
        from .data_cleaning_dual import get_dual_mode_analysis_df
        dataset_names = ["Dataset1", "Dataset2", "Dataset4"]
        selected_dataset = st.selectbox("1. 选择数据集", dataset_names, key="n3_dual_dataset")
        measure_names = list(st.session_state.dc_measures.keys())
        if not measure_names:
            st.warning("请在数据清洗模块的「Measure 划分」中至少定义一个 Measure。")
        else:
            selected_measures = st.multiselect(
                "2. 选择 Measure（可多选，每个 Measure 对应一个潜变量）",
                measure_names,
                default=measure_names[:1] if measure_names else [],
                key="n3_dual_measures",
            )
            if selected_measures:
                df_dual = get_dual_mode_analysis_df(
                    selected_dataset,
                    selected_measures,
                    st.session_state.dc_dataset_full,
                    st.session_state.dc_measures,
                    item_columns_only=True,
                )
                if df_dual is not None and not df_dual.empty:
                    st.info(f"使用 **{selected_dataset}**，Measure: {', '.join(selected_measures)}（{len(df_dual)} 行 × {len(df_dual.columns)} 列）")

                    st.markdown("### 运行前配置（两种模型共用）")
                    st.caption("请确认每个潜变量的题目清单，以及方法因子题目。默认按题目文本包含“r”预选方法因子。")
                    measuregroup_title = st.text_input(
                        "measuregroup_title",
                        key="n3_measuregroup_title",
                        placeholder="例如：Workplace Wellbeing Battery",
                    )

                    selected_item_map: Dict[str, List[str]] = {}
                    all_selected_items: List[str] = []
                    for m in selected_measures:
                        m_items = [c for c in st.session_state.dc_measures.get(m, []) if c in df_dual.columns]
                        m_items = sort_item_cols_by_number(m_items)
                        pick = smart_multiselect(
                            options=m_items,
                            label=f"潜变量「{m}」题目选择",
                            key_suffix=f"n3_latent_items_{_sanitize_name(m)}",
                            default_selected=m_items,
                            show_selection_controls=True,
                        )
                        pick = [x for x in pick if x in m_items]
                        selected_item_map[m] = pick
                        all_selected_items.extend(pick)

                    all_selected_items = list(dict.fromkeys(all_selected_items))
                    method_key_suffix = "n3_method_items_shared"
                    method_sig_key = f"{method_key_suffix}_options_sig"
                    method_options_sig = tuple(all_selected_items)
                    # 选项集发生变化时，清理上次缓存，避免旧选择覆盖新的默认预选
                    if st.session_state.get(method_sig_key) != method_options_sig:
                        _reset_smart_multiselect_cache(method_key_suffix)
                        st.session_state[method_sig_key] = method_options_sig
                    default_method_items = [x for x in all_selected_items if _is_reverse_method_item(x)]
                    def _on_reset_method_dual():
                        _reset_smart_multiselect_cache(method_key_suffix)
                        st.session_state[method_sig_key] = method_options_sig
                        st.session_state[f"{method_key_suffix}_last_selected"] = default_method_items
                    st.button('按”末尾 r”规则重新预选方法因子题目', key="n3_btn_reset_method_defaults",
                              on_click=_on_reset_method_dual)
                    method_selected = smart_multiselect(
                        options=all_selected_items,
                        label="方法因子（method）题目选择（可多选）",
                        key_suffix=method_key_suffix,
                        default_selected=default_method_items,
                        show_selection_controls=True,
                    )

                    if st.button("应用上述选题到模型", key="n3_apply_dual_item_selection"):
                        if not measuregroup_title or not measuregroup_title.strip():
                            st.warning("请先填写 measuregroup_title。")
                        elif any(len(v) == 0 for v in selected_item_map.values()):
                            st.warning("每个潜变量至少保留 1 个题目后再应用。")
                        else:
                            # 为多因子 CFA 添加 Measure 前缀列名（与 run_multifactor_cfa 期望一致）
                            prefixed = {}
                            for m in selected_measures:
                                safe_m = _sanitize_name(m) + "__"
                                for c in selected_item_map.get(m, []):
                                    if c in df_dual.columns:
                                        prefixed[safe_m + c] = df_dual[c].values
                            if prefixed:
                                df_prefixed = pd.DataFrame(prefixed)
                                st.session_state.n3_merged_df = df_prefixed
                                st.session_state.n3_merged_names = list(selected_measures)
                                st.session_state.n3_merged_source = "four"
                                # 方法因子题目需转换到前缀列名空间
                                method_prefixed = []
                                for m in selected_measures:
                                    safe_m = _sanitize_name(m) + "__"
                                    m_items = set(selected_item_map.get(m, []))
                                    for c in method_selected:
                                        if c in m_items:
                                            method_prefixed.append(safe_m + c)
                                st.session_state.n3_method_items_shared = list(dict.fromkeys(method_prefixed))
                                st.success("已应用选题配置，可在下方 Model Specification 中运行模型。")
                    # 仅显示一次加载信息，避免重复提示

    if data_source_choice == "上传文件 (Upload files)":
        st.markdown("### 上传文件")
        st.caption("每个上传文件对应一个 measure 数据集。请为每个文件命名（作为潜变量名称），其余建模步骤保持原逻辑。")

        uploaded_files = st.file_uploader(
            "上传一个或多个 Excel/CSV 文件",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="n3_upload_measure_files",
        )

        upload_rows = []
        upload_datasets: Dict[str, pd.DataFrame] = {}
        name_seen = set()
        invalid_name_count = 0
        duplicate_name_count = 0
        parse_errors = []

        if uploaded_files:
            st.markdown("#### 上传文件命名（每个文件一个 measure）")
            for i, uf in enumerate(uploaded_files, start=1):
                default_name = _default_measure_name_from_filename(getattr(uf, "name", ""), i)
                input_key = f"n3_upload_measure_name_{i}_{_sanitize_name(str(getattr(uf, 'name', 'file')))}"
                measure_name = st.text_input(
                    f"{getattr(uf, 'name', f'file_{i}')}: measure 名称",
                    value=default_name,
                    key=input_key,
                ).strip()

                if not measure_name:
                    invalid_name_count += 1
                    continue
                if measure_name in name_seen:
                    duplicate_name_count += 1
                    continue

                try:
                    df_up = _read_uploaded_table(uf)
                except Exception as e:
                    parse_errors.append(f"{getattr(uf, 'name', f'file_{i}')}: {e}")
                    continue
                if df_up is None or df_up.empty:
                    parse_errors.append(f"{getattr(uf, 'name', f'file_{i}')}: 文件为空")
                    continue

                num_df = df_up.apply(pd.to_numeric, errors="coerce")
                valid_numeric_cols = [c for c in num_df.columns if not num_df[c].isna().all()]
                if not valid_numeric_cols:
                    parse_errors.append(f"{getattr(uf, 'name', f'file_{i}')}: 无可用数值列")
                    continue

                df_up_numeric = num_df[valid_numeric_cols].copy()
                upload_rows.append({
                    "measure_name": measure_name,
                    "file_name": getattr(uf, "name", f"file_{i}"),
                    "n_samples": int(df_up_numeric.shape[0]),
                    "n_items": int(df_up_numeric.shape[1]),
                })
                upload_datasets[measure_name] = df_up_numeric
                name_seen.add(measure_name)

            if upload_rows:
                st.dataframe(pd.DataFrame(upload_rows), use_container_width=True, hide_index=True)
            if invalid_name_count > 0:
                st.warning(f"有 {invalid_name_count} 个文件未填写 measure 名称，已暂不纳入。")
            if duplicate_name_count > 0:
                st.warning(f"有 {duplicate_name_count} 个文件的 measure 名称重复，已暂不纳入。请确保名称唯一。")
            if parse_errors:
                st.warning("以下文件未能纳入：" + "； ".join(parse_errors[:6]))

            if st.button("确认使用上述数据库", key="n3_apply_upload_files"):
                if len(upload_datasets) < 1:
                    st.warning("请至少上传并命名 1 个可用文件。")
                else:
                    merged_upload = merge_dataframes(list(upload_datasets.keys()), upload_datasets)
                    if merged_upload is None or merged_upload.empty:
                        st.error("上传文件合并失败（可能样本行不重叠）。请检查上传数据后重试。")
                    else:
                        st.session_state.n3_merged_df = merged_upload
                        st.session_state.n3_merged_names = list(upload_datasets.keys())
                        st.session_state.n3_merged_source = "upload"
                        st.success(
                            f"已应用上传数据：{len(upload_datasets)} 个 measure，"
                            f"{merged_upload.shape[0]} 行 × {merged_upload.shape[1]} 列。"
                        )
        else:
            st.info("请先上传文件，再为每个文件填写 measure 名称并点击“确认使用上述数据库”。")

        if (
            st.session_state.get("n3_merged_source") == "upload"
            and st.session_state.get("n3_merged_df") is not None
            and st.session_state.get("n3_merged_names")
        ):
            st.markdown("### 运行前配置（两种模型共用）")
            st.caption("请确认每个潜变量的题目清单，以及方法因子题目。默认按题目文本末尾“r”预选方法因子。")
            df_upload_merge = st.session_state.get("n3_merged_df")
            upload_names = st.session_state.get("n3_merged_names", [])

            measuregroup_title = st.text_input(
                "measuregroup_title",
                key="n3_measuregroup_title",
                placeholder="例如：Workplace Wellbeing Battery",
            )

            selected_item_map_upload: Dict[str, List[str]] = {}
            all_selected_items_upload: List[str] = []
            for m in upload_names:
                prefix = f"{m}__"
                m_items = [c for c in df_upload_merge.columns if str(c).startswith(prefix)]
                m_items = sort_item_cols_by_number(m_items)
                pick = smart_multiselect(
                    options=m_items,
                    label=f"潜变量「{m}」题目选择",
                    key_suffix=f"n3_upload_latent_items_{_sanitize_name(m)}",
                    default_selected=m_items,
                    show_selection_controls=True,
                )
                pick = [x for x in pick if x in m_items]
                selected_item_map_upload[m] = pick
                all_selected_items_upload.extend(pick)

            all_selected_items_upload = list(dict.fromkeys(all_selected_items_upload))
            method_key_suffix = "n3_method_items_shared"
            method_sig_key = f"{method_key_suffix}_options_sig"
            method_options_sig = tuple(all_selected_items_upload)
            if st.session_state.get(method_sig_key) != method_options_sig:
                _reset_smart_multiselect_cache(method_key_suffix)
                st.session_state[method_sig_key] = method_options_sig
            default_method_items_upload = [x for x in all_selected_items_upload if _is_reverse_method_item(x)]
            def _on_reset_method_upload():
                _reset_smart_multiselect_cache(method_key_suffix)
                st.session_state[method_sig_key] = method_options_sig
                st.session_state[f"{method_key_suffix}_last_selected"] = default_method_items_upload
            st.button('按”末尾 r”规则重新预选方法因子题目', key="n3_btn_reset_method_defaults_upload",
                      on_click=_on_reset_method_upload)
            method_selected_upload = smart_multiselect(
                options=all_selected_items_upload,
                label="方法因子（method）题目选择（可多选）",
                key_suffix=method_key_suffix,
                default_selected=default_method_items_upload,
                show_selection_controls=True,
            )

            if st.button("应用上述选题到模型", key="n3_apply_upload_item_selection"):
                if not measuregroup_title or not measuregroup_title.strip():
                    st.warning("请先填写 measuregroup_title。")
                elif any(len(v) == 0 for v in selected_item_map_upload.values()):
                    st.warning("每个潜变量至少保留 1 个题目后再应用。")
                else:
                    selected_cols_upload: List[str] = []
                    for m in upload_names:
                        selected_cols_upload.extend(selected_item_map_upload.get(m, []))
                    selected_cols_upload = list(dict.fromkeys([c for c in selected_cols_upload if c in df_upload_merge.columns]))
                    if selected_cols_upload:
                        st.session_state.n3_merged_df = df_upload_merge[selected_cols_upload].copy()
                        st.session_state.n3_merged_names = list(upload_names)
                        st.session_state.n3_merged_source = "upload"
                        st.session_state.n3_method_items_shared = [
                            c for c in dict.fromkeys(method_selected_upload) if c in selected_cols_upload
                        ]
                        st.success("已应用选题配置，可在下方 Model Specification 中运行模型。")

    if has_sub and data_source_choice == "子数据集合并 (Sub-datasets)":
        source = "数据清洗"
        rows = []
        for name, df in sub_datasets.items():
            n_items = df.shape[1]
            n_samples = df.shape[0]
            status = get_dataframe_status(df)
            created_date = get_dataframe_date(name)
            rows.append({"name": name, "n_items": n_items, "n_samples": n_samples, "status": status, "date": created_date})

        st.markdown("### 数据集合并")
        st.caption("从下方选择需要合并的单因子/子数据集，预览或执行合并后可在右侧查看合并结果。")

        search = st.text_input("Search dataframes...", key="n3_search_df", placeholder="搜索数据集名称...")
        st.selectbox("All Sources", [source], key="n3_source_filter")

        if search and search.strip():
            q = search.strip().lower()
            rows = [r for r in rows if q in r["name"].lower()]

        if "n3_selected_names" not in st.session_state:
            st.session_state.n3_selected_names = []

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Available Single-Factor Dataframes")
            st.caption("Selected: " + str(len(st.session_state.n3_selected_names)))
            selected_now = []
            if not rows:
                st.caption("（无匹配的数据集，请调整搜索条件或先在数据清洗中创建子数据集）")
            else:
                for r in rows:
                    name = r["name"]
                    n_items, n_samples, status, created_date = r["n_items"], r["n_samples"], r["status"], r["date"]
                    default = name in st.session_state.n3_selected_names
                    label = f"{name}  ·  {n_items} items  ·  {n_samples} samples  ·  {status}  ·  {created_date}"
                    checked = st.checkbox(label, value=default, key=f"n3_cb_{name}")
                    if checked:
                        selected_now.append(name)
            st.session_state.n3_selected_names = selected_now
            st.markdown("")
            prev_btn = st.button("Preview selected", type="secondary", key="n3_btn_preview")
            merge_btn = st.button("Merge selected", type="primary", key="n3_btn_merge")

        with col_right:
            st.markdown("#### Merged Dataframe Preview: DF_merge")
            st.caption("Select dataframes on the left and click **Preview selected** to view the merged data. Merge uses inner join by row index; column names are prefixed with dataset name to avoid duplicates.")
            if prev_btn or merge_btn:
                if not selected_now:
                    st.warning("请至少勾选一个数据集。")
                else:
                    merged = merge_dataframes(selected_now, sub_datasets)
                    if merged is None:
                        st.error("合并失败，请检查所选数据集。")
                    else:
                        if merge_btn:
                            st.session_state.n3_merged_df = merged
                            st.session_state.n3_merged_df_full = merged.copy()  # 保留完整合并数据，供选项列表使用
                            st.session_state.n3_merged_names = selected_now
                            st.session_state.n3_merged_source = "sub"
                            st.success(f"已合并 {len(selected_now)} 个数据集，共 {merged.shape[0]} 行、{merged.shape[1]} 列。")
                        st.dataframe(merged.head(20), use_container_width=True)
                        st.caption(f"Preview: shape = {merged.shape[0]} rows × {merged.shape[1]} columns. Showing first 20 rows.")
            if st.session_state.n3_merged_df is not None and not (prev_btn or merge_btn):
                df_merge = st.session_state.n3_merged_df
                st.success(f"当前合并结果：{len(st.session_state.n3_merged_names)} 个数据集 → {df_merge.shape[0]} 行 × {df_merge.shape[1]} 列")
                st.dataframe(df_merge.head(20), use_container_width=True)
                st.caption(f"Shape: {df_merge.shape[0]} rows × {df_merge.shape[1]} columns. Showing first 20 rows.")

        st.markdown("---")
        if st.session_state.n3_merged_df is not None:
            df_merge = st.session_state.n3_merged_df
            buf = df_merge.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="📥 下载合并后的数据集 (DF_merge.csv)",
                data=buf,
                file_name="DF_merge.csv",
                mime="text/csv",
                key="n3_download_merge",
            )
            save_name = st.text_input("将当前合并另存为（名称）", key="n3_save_merge_name", placeholder="例如：三量表合并_202301")
            if st.button("保存当前合并", key="n3_btn_save_merge"):
                if save_name and save_name.strip():
                    st.session_state.n3_saved_merges[save_name.strip()] = {
                        "df": st.session_state.n3_merged_df.copy(),
                        "names": list(st.session_state.n3_merged_names),
                    }
                    st.success(f"已保存为「{save_name.strip()}」。")
                else:
                    st.warning("请输入保存名称。")
            if st.session_state.n3_saved_merges:
                st.caption("已保存的合并：" + "、".join(st.session_state.n3_saved_merges.keys()))

        if st.session_state.get("n3_merged_df") is not None and st.session_state.get("n3_merged_names"):
            st.markdown("### 运行前配置（两种模型共用）")
            st.caption("请确认每个潜变量的题目清单，以及方法因子题目。默认按题目文本末尾“r”预选方法因子。")
            # 构建选项列表用完整数据（n3_merged_df_full），避免"应用选题"后选项被截断
            df_sub_merge = st.session_state.get("n3_merged_df_full", st.session_state.get("n3_merged_df"))
            sub_names = st.session_state.get("n3_merged_names", [])

            measuregroup_title = st.text_input(
                "measuregroup_title",
                key="n3_measuregroup_title",
                placeholder="例如：Workplace Wellbeing Battery",
            )

            selected_item_map_sub: Dict[str, List[str]] = {}
            all_selected_items_sub: List[str] = []
            for m in sub_names:
                prefix = f"{m}__"
                m_items = [c for c in df_sub_merge.columns if str(c).startswith(prefix)]
                m_items = sort_item_cols_by_number(m_items)
                pick = smart_multiselect(
                    options=m_items,
                    label=f"潜变量「{m}」题目选择",
                    key_suffix=f"n3_sub_latent_items_{_sanitize_name(m)}",
                    default_selected=m_items,
                    show_selection_controls=True,
                )
                pick = [x for x in pick if x in m_items]
                selected_item_map_sub[m] = pick
                all_selected_items_sub.extend(pick)

            all_selected_items_sub = list(dict.fromkeys(all_selected_items_sub))
            method_key_suffix = "n3_method_items_shared"
            method_sig_key = f"{method_key_suffix}_options_sig"
            method_options_sig = tuple(all_selected_items_sub)
            if st.session_state.get(method_sig_key) != method_options_sig:
                _reset_smart_multiselect_cache(method_key_suffix)
                st.session_state[method_sig_key] = method_options_sig
            default_method_items_sub = [x for x in all_selected_items_sub if _is_reverse_method_item(x)]
            def _on_reset_method_sub():
                _reset_smart_multiselect_cache(method_key_suffix)
                st.session_state[method_sig_key] = method_options_sig
                st.session_state[f"{method_key_suffix}_last_selected"] = default_method_items_sub
            st.button('按”末尾 r”规则重新预选方法因子题目', key="n3_btn_reset_method_defaults_sub",
                      on_click=_on_reset_method_sub)
            method_selected_sub = smart_multiselect(
                options=all_selected_items_sub,
                label="方法因子（method）题目选择（可多选）",
                key_suffix=method_key_suffix,
                default_selected=default_method_items_sub,
                show_selection_controls=True,
            )

            if st.button("应用上述选题到模型", key="n3_apply_sub_item_selection"):
                if not measuregroup_title or not measuregroup_title.strip():
                    st.warning("请先填写 measuregroup_title。")
                elif any(len(v) == 0 for v in selected_item_map_sub.values()):
                    st.warning("每个潜变量至少保留 1 个题目后再应用。")
                else:
                    selected_cols_sub: List[str] = []
                    for m in sub_names:
                        selected_cols_sub.extend(selected_item_map_sub.get(m, []))
                    selected_cols_sub = list(dict.fromkeys([c for c in selected_cols_sub if c in df_sub_merge.columns]))
                    if selected_cols_sub:
                        # 确保完整数据已保存（首次应用或数据源切换时）
                        if "n3_merged_df_full" not in st.session_state or st.session_state.get("n3_merged_source") != "sub":
                            st.session_state.n3_merged_df_full = df_sub_merge.copy()
                        # 模型用的数据只包含选中的列（过滤掉不在完整数据中的列名，防止数据源切换后 KeyError）
                        valid_cols = [c for c in selected_cols_sub if c in st.session_state.n3_merged_df_full.columns]
                        st.session_state.n3_merged_df = st.session_state.n3_merged_df_full[valid_cols].copy() if valid_cols else pd.DataFrame()
                        st.session_state.n3_merged_names = list(sub_names)
                        st.session_state.n3_merged_source = "sub"
                        st.session_state.n3_method_items_shared = [
                            c for c in dict.fromkeys(method_selected_sub) if c in selected_cols_sub
                        ]
                        st.success("已应用选题配置，可在下方 Model Specification 中运行模型。")

    # ========== Model Specification（可切换 Tab）==========
    st.markdown("---")
    st.markdown("### Run Models")

    model_tab_choice = st.radio(
        "模型分页",
        ["Multi-factor Correlated Model", "Higher-order Model"],
        key="n3_model_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    if model_tab_choice == "Multi-factor Correlated Model":
        st.markdown("#### 多因子相关模型")
        st.caption("多因子相关模型：潜变量之间自由相关；每个潜变量首题载荷固定为 1 且方差自由估计；method 与潜变量正交且方差固定为 1。")
        cfa_df = st.session_state.get("n3_merged_df")
        cfa_names = st.session_state.get("n3_merged_names", [])
        if cfa_df is None or not cfa_names:
            st.info("请先在上方“运行前配置”中应用题目选择，生成共享建模数据。")
        else:
            # 方案A：共享配置中的 method 题目用“原始列名”存储，此处按原始列名过滤
            method_selected = [c for c in st.session_state.get("n3_method_items_shared", []) if c in cfa_df.columns]
            if not method_selected:
                st.warning("当前未设置方法因子题目。请先回到上方“运行前配置”选择至少 1 个方法因子题目。")

            mapping = "  |  ".join([f"F{i+1} = {name}" for i, name in enumerate(cfa_names)])
            st.caption("当前共享配置（潜变量对应）： " + mapping)

            run_cfa_btn = st.button("运行 Multi-Factor CFA", type="primary", key="n3_btn_run_cfa")
            if run_cfa_btn:
                if not method_selected:
                    st.error("请先在“运行前配置”中至少选择 1 个方法因子题目。")
                    st.stop()
                with st.spinner("正在拟合多因子 CFA..."):
                    model, estimates, stats_dict, err, syntax = run_multifactor_cfa(
                        cfa_df, cfa_names, method_selected
                    )
                if err:
                    st.error(err)
                    if syntax:
                        st.code(syntax, language="text")
                else:
                    st.session_state.n3_cfa_model = model
                    st.session_state.n3_cfa_estimates = estimates
                    st.session_state.n3_cfa_stats = stats_dict
                    st.session_state.n3_cfa_syntax = syntax
                    st.session_state.n3_cfa_dataset_names = cfa_names
                    st.session_state.n3_cfa_method_items = method_selected
                    st.success("Multi-Factor CFA 拟合成功。")

            # Multi-Factor CFA 结果（仅在本分页内显示）
            if "n3_cfa_stats" in st.session_state and st.session_state.n3_cfa_stats is not None:
                st.markdown("---")
                st.subheader("Multi-Factor CFA 结果")
                stats_dict = st.session_state.n3_cfa_stats
                syntax = st.session_state.get("n3_cfa_syntax", "")

                chi2_val = _get_any_stat(stats_dict, ["chi2", "Chi2"])
                dof_val = _get_any_stat(stats_dict, ["DoF", "dof", "df"])
                p_val = _get_any_stat(stats_dict, ["chi2 p-value", "p-value", "pvalue", "p_value"])
                st.markdown("#### 模型检验 (Model Test User Model)")
                st.table(pd.DataFrame({
                    "Statistic": ["Chi-square", "Degrees of freedom", "P-value"],
                    "Value": [
                        f"{chi2_val:.3f}" if not np.isnan(chi2_val) else "N/A",
                        f"{int(dof_val)}" if not np.isnan(dof_val) else "N/A",
                        f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
                    ],
                }))

                st.markdown("#### 拟合指数 (Fit Indices)")
                fit_keys = ["CFI", "TLI", "RMSEA", "SRMR", "GFI", "AGFI", "NFI", "AIC", "BIC", "SABIC", "LogL"]
                fit_key_aliases = {
                    "CFI": ["CFI"],
                    "TLI": ["TLI"],
                    "RMSEA": ["RMSEA"],
                    "SRMR": ["SRMR", "srmr"],
                    "GFI": ["GFI"],
                    "AGFI": ["AGFI"],
                    "NFI": ["NFI"],
                    "AIC": ["AIC"],
                    "BIC": ["BIC"],
                    "SABIC": ["SABIC"],
                    "LogL": ["LogL", "logl", "LogLik", "loglik", "log_likelihood", "log-likelihood"],
                }
                fit_rows = []
                for k in fit_keys:
                    val = _get_any_stat(stats_dict, fit_key_aliases.get(k, [k]))
                    if np.isnan(val):
                        disp = "N/A"
                    elif k in ("AIC", "BIC", "SABIC", "LogL") or "chi" in k.lower():
                        disp = f"{val:.2f}"
                    else:
                        disp = f"{val:.4f}"
                    fit_rows.append({"Index": k, "Value": disp})
                st.dataframe(pd.DataFrame(fit_rows), use_container_width=True, hide_index=True)

                st.markdown("#### 标准化因子载荷 (Standardized Factor Loadings)")
                est = st.session_state.get("n3_cfa_estimates")
                if est is not None and not est.empty:
                    loadings = est[(est["op"] == "=~") | (est["op"] == "~")].copy()
                    if "Std.all" in loadings.columns:
                        display_cols = ["LHS", "op", "RHS", "Estimate", "Std.Err", "Std.all", "P(>|z|)"]
                    else:
                        display_cols = [c for c in ["LHS", "op", "RHS", "Estimate", "Std.Err", "P(>|z|)"] if c in loadings.columns]
                    show_df = loadings[[c for c in display_cols if c in loadings.columns]]
                    num_cols = [c for c in show_df.columns if show_df[c].dtype in (np.floating, np.int64)]
                    if num_cols:
                        st.dataframe(show_df.style.format({c: "{:.3f}" for c in num_cols}))
                    else:
                        st.dataframe(show_df)
                    csv_est = loadings.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("📥 下载因子载荷表", data=csv_est, file_name="multifactor_cfa_loadings.csv", mime="text/csv", key="n3_dl_loadings")

                st.markdown("#### 潜变量间相关 (Factor Correlations)")
                if est is not None and not est.empty:
                    cov_rows = est[(est["op"] == "~~") & (est["LHS"] != est["RHS"])]
                    factor_names_cfa = [f"F{i+1}" for i in range(len(st.session_state.get("n3_cfa_dataset_names", [])))]
                    corr_rows = cov_rows[cov_rows["LHS"].isin(factor_names_cfa) & cov_rows["RHS"].isin(factor_names_cfa)]
                    if not corr_rows.empty:
                        cols = [c for c in ["LHS", "RHS", "Estimate", "Std.all"] if c in corr_rows.columns]
                        sub = [c for c in cols if corr_rows[c].dtype in (np.floating,)]
                        if sub:
                            st.dataframe(corr_rows[cols].style.format({c: "{:.3f}" for c in sub}))
                        else:
                            st.dataframe(corr_rows[cols])
                    else:
                        st.caption("（潜变量间相关见上方参数估计表中 LHS ~~ RHS 且 LHS、RHS 均为 F1, F2, … 的行。）")

                st.markdown("#### 模型语法 (Model Syntax)")
                st.code(syntax, language="text")

                st.markdown("#### 📥 生成模型下载表")
                cfa_df_for_report = st.session_state.get("n3_merged_df")
                cfa_names_for_report = st.session_state.get("n3_merged_names", [])
                if cfa_df_for_report is None or not cfa_names_for_report:
                    st.info("请先在上方“运行前配置”中应用题目选择。")
                else:
                    st.caption("请为每个潜变量命名（通常对应各 Measure 名称）。")
                    latent_display_names = []
                    for i, n in enumerate(cfa_names_for_report):
                        latent_display_names.append(
                            st.text_input(
                                f"潜变量 F{i+1} 名称",
                                value=str(n),
                                key=f"n3_cfa_latent_name_{i}",
                            ).strip() or f"F{i+1}"
                        )
                    measuregroup_title = (st.session_state.get("n3_measuregroup_title") or "").strip()
                    if not measuregroup_title:
                        measuregroup_title = st.text_input(
                            "measuregroup_title（用于 sheet 命名）",
                            value="measuregroup",
                            key="n3_cfa_measuregroup_title_fallback",
                        ).strip() or "measuregroup"
                    if st.button("生成下载表（题目表 + 潜变量方差协方差 + 题目协方差）", key="n3_btn_build_cfa_report"):
                        try:
                            est_r = st.session_state.get("n3_cfa_estimates")
                            stats_r = st.session_state.get("n3_cfa_stats") or {}
                            items_sheet, latent_cov, item_cov, cr_warning_msg = _build_multifactor_report_tables(
                                cfa_df_for_report,
                                cfa_names_for_report,
                                est_r,
                                stats_r,
                                latent_display_names,
                                measuregroup_title,
                            )
                            xbuf = io.BytesIO()
                            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as w:
                                sh1 = (measuregroup_title[:31] if len(measuregroup_title) > 31 else measuregroup_title) or "Model"
                                sh1 = "".join(c for c in sh1 if c not in '[]:*?/\\')
                                sh2_base = (measuregroup_title + "_cov")
                                sh2 = (sh2_base[:31] if len(sh2_base) > 31 else sh2_base) or "Model_cov"
                                sh2 = "".join(c for c in sh2 if c not in '[]:*?/\\')
                                sh3_base = (measuregroup_title + "_item_cov")
                                sh3 = (sh3_base[:31] if len(sh3_base) > 31 else sh3_base) or "Model_item_cov"
                                sh3 = "".join(c for c in sh3 if c not in '[]:*?/\\')
                                items_sheet.to_excel(w, sheet_name=sh1, index=False)
                                latent_cov.to_excel(w, sheet_name=sh2, index=True)
                                item_cov.to_excel(w, sheet_name=sh3, index=True)
                            xbuf.seek(0)
                            st.session_state.n3_cfa_report_xlsx = xbuf.getvalue()
                            st.session_state.n3_cfa_report_items_preview = items_sheet.copy()
                            st.session_state.n3_cfa_report_cov_preview = latent_cov.copy()
                            st.session_state.n3_cfa_report_item_cov_preview = item_cov.copy()
                            st.session_state.n3_cfa_cr_warning = cr_warning_msg or ""
                            measuregroup_id = _safe_filename_part(measuregroup_title, "measuregroup")
                            user_name = st.session_state.get("user_name", "unknown_user")
                            safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name)).strip() or "unknown_user"
                            today = date.today().strftime("%Y-%m-%d")
                            st.session_state.n3_cfa_report_filename = f"{measuregroup_id}_multi_cfa_report_{today}_{safe_user}.xlsx"
                            st.success("已生成下载表。")
                        except Exception as e:
                            st.error(f"生成下载表失败: {e}")
                    if st.session_state.get("n3_cfa_report_items_preview") is not None:
                        st.markdown("##### 预览：题目明细表（前20行）")
                        st.dataframe(
                            st.session_state.n3_cfa_report_items_preview.head(20),
                            use_container_width=True,
                        )
                    if st.session_state.get("n3_cfa_report_cov_preview") is not None:
                        st.markdown("##### 预览：潜变量方差-协方差矩阵（前20行）")
                        st.dataframe(
                            st.session_state.n3_cfa_report_cov_preview.head(20),
                            use_container_width=True,
                        )
                    if st.session_state.get("n3_cfa_report_item_cov_preview") is not None:
                        st.markdown("##### 预览：题目方差-协方差矩阵（前20行）")
                        st.dataframe(
                            st.session_state.n3_cfa_report_item_cov_preview.head(20),
                            use_container_width=True,
                        )
                    if st.session_state.get("n3_cfa_cr_warning"):
                        st.warning(st.session_state.n3_cfa_cr_warning)
                    if st.session_state.get("n3_cfa_report_xlsx"):
                        st.download_button(
                            "⬇️ 下载 Multi-factor CFA 模型表（Excel）",
                            data=st.session_state.n3_cfa_report_xlsx,
                            file_name=st.session_state.get("n3_cfa_report_filename", "measuregroup+measure_multi_cfa_report.xlsx"),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="n3_download_cfa_report_xlsx",
                        )

                st.markdown("#### 🧮 最终得分计算（基于 Multi-factor Correlated Model）")
                st.caption("输出每个潜变量报告分（Output A）以及 Overall relative final score（Output B）；统一使用非标准化载荷（Estimate）。absolute final score 与下方导出公式参数表将复用此处填写的 scale_min / scale_max。")
                cfa_s1, cfa_s2 = st.columns(2)
                with cfa_s1:
                    n3_cfa_scale_min = st.number_input("scale_min", min_value=0, value=1, step=1, key="n3_cfa_scale_min")
                with cfa_s2:
                    n3_cfa_scale_max = st.number_input("scale_max", min_value=1, value=7, step=1, key="n3_cfa_scale_max")
                is_four_dataset = (data_source_choice == "四数据集 (Dataset → Measure)")
                if not is_four_dataset:
                    st.number_input("当前是 dataset 1-4中的哪一个？（用于下载文件命名）", min_value=1, value=1, step=1, key="n3_cfa_scored_dataset_n")
                if int(n3_cfa_scale_min) >= int(n3_cfa_scale_max):
                    st.error("scale_min 必须小于 scale_max。")
                elif (int(n3_cfa_scale_min), int(n3_cfa_scale_max)) != (1, 7):
                    st.warning("⚠️ 当前 scale_min / scale_max 非默认 [1, 7]。请确认量表边界与问卷设计一致，否则 absolute 0–100 分数解释可能产生偏差。")
                if st.button("计算最终得分并生成可下载数据集", key="n3_btn_cfa_final_score"):
                    try:
                        cfa_df_score = st.session_state.get("n3_merged_df")
                        cfa_names_score = st.session_state.get("n3_cfa_dataset_names", [])
                        est_score = st.session_state.get("n3_cfa_estimates")
                        scored_df, n_missing_rows = _compute_multifactor_final_scores(
                            cfa_df_score,
                            cfa_names_score,
                            est_score,
                            scale_min=int(st.session_state.get("n3_cfa_scale_min", 1)),
                            scale_max=int(st.session_state.get("n3_cfa_scale_max", 7)),
                        )
                        st.session_state.n3_cfa_scored_df = scored_df

                        csv_bytes = scored_df.to_csv(index=False).encode("utf-8-sig")
                        st.session_state.n3_cfa_scored_csv_bytes = csv_bytes

                        xlsx_buf = io.BytesIO()
                        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w_x:
                            scored_df.to_excel(w_x, sheet_name="ScoredData", index=False)
                        xlsx_buf.seek(0)
                        st.session_state.n3_cfa_scored_xlsx_bytes = xlsx_buf.getvalue()

                        if n_missing_rows > 0:
                            st.warning(f"检测到 {n_missing_rows} 行样本存在缺失值，这些样本的得分列已记为 -999。")
                        else:
                            st.info("检测到 0 行样本存在缺失值。")
                        st.success(f"最终得分计算完成：共 {len(scored_df)} 行样本。")
                    except Exception as e:
                        st.error(f"计算 Multi-factor 最终得分失败: {e}")

                if st.session_state.get("n3_cfa_scored_df") is not None:
                    cfa_names_map = st.session_state.get("n3_cfa_dataset_names", [])
                    if cfa_names_map:
                        mapping_lines = [f"F{i+1} → {name}" for i, name in enumerate(cfa_names_map)]
                        st.markdown("##### 得分列名映射（F → Measure）")
                        st.caption(" | ".join(mapping_lines))
                    st.markdown("##### 数据预览（含因子得分与最终得分，前20行）")
                    preview_scored = st.session_state.n3_cfa_scored_df.head(20).copy()
                    score_cols = [c for c in preview_scored.columns if str(c).startswith("relative_final_score")]
                    for c in score_cols:
                        preview_scored[c] = pd.to_numeric(preview_scored[c], errors="coerce").round(3)
                    if "absolute_final_score" in preview_scored.columns:
                        preview_scored["absolute_final_score"] = pd.to_numeric(
                            preview_scored["absolute_final_score"], errors="coerce"
                        ).round(3)
                    st.dataframe(preview_scored, use_container_width=True)

                    dl_col1, dl_col2 = st.columns(2)
                    measuregroup_title_dl = (st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup"
                    mgid_dl = _safe_filename_part(measuregroup_title_dl, "measuregroup")
                    cfa_user = st.session_state.get("user_name", "unknown_user")
                    cfa_safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(cfa_user)).strip() or "unknown_user"
                    cfa_today = date.today().strftime("%Y-%m-%d")
                    if data_source_choice == "四数据集 (Dataset → Measure)":
                        ds_name = st.session_state.get("n3_dual_dataset", "Dataset4")
                        m = re.search(r"(\d+)", str(ds_name))
                        cfa_dataset_n = int(m.group(1)) if m else 4
                    else:
                        cfa_dataset_n = int(st.session_state.get("n3_cfa_scored_dataset_n", 1))
                    cfa_scored_base = f"{mgid_dl}_multi_cfa_dataset{cfa_dataset_n}_scored_{cfa_today}_{cfa_safe_user}"
                    with dl_col1:
                        if st.session_state.get("n3_cfa_scored_csv_bytes"):
                            st.download_button(
                                "⬇️ 下载含得分数据（CSV）",
                                data=st.session_state.n3_cfa_scored_csv_bytes,
                                file_name=f"{cfa_scored_base}.csv",
                                mime="text/csv",
                                key="n3_download_cfa_scored_csv",
                            )
                    with dl_col2:
                        if st.session_state.get("n3_cfa_scored_xlsx_bytes"):
                            st.download_button(
                                "⬇️ 下载含得分数据（Excel）",
                                data=st.session_state.n3_cfa_scored_xlsx_bytes,
                                file_name=f"{cfa_scored_base}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="n3_download_cfa_scored_xlsx",
                            )

                st.markdown("#### 📤 导出最终得分公式参数表")
                st.caption("FinalScore = intercept + Σ(beta * item)。")
                _cfa_smin = int(st.session_state.get("n3_cfa_scale_min", 1))
                _cfa_smax = int(st.session_state.get("n3_cfa_scale_max", 7))
                st.caption(f"公式参数将使用**上方** scale_min / scale_max（当前为 [{_cfa_smin}, {_cfa_smax}]）。若已修改 scale_min / scale_max，建议重新执行「计算最终得分」以保持数据与公式一致。若非 1–7 量表，请先在上方修改后再生成。")
                #st.caption("measure_id 将自动使用上方“生成模型下载表”中填写的潜变量名称（无需重复填写）。")
                cfa_names_for_hint = st.session_state.get("n3_cfa_dataset_names", [])
                formula_measure_ids_hint = [
                    (st.session_state.get(f"n3_cfa_latent_name_{i}") or str(n)).strip() or f"F{i+1}"
                    for i, n in enumerate(cfa_names_for_hint)
                ]
                if formula_measure_ids_hint:
                    st.info("本次将使用的 measure_id： " + " | ".join(formula_measure_ids_hint))

                if st.button("生成公式参数表", key="n3_btn_build_cfa_formula"):
                    try:
                        formula_scale_min = int(st.session_state.get("n3_cfa_scale_min", 1))
                        formula_scale_max = int(st.session_state.get("n3_cfa_scale_max", 7))
                        if formula_scale_min >= formula_scale_max:
                            st.error("scale_min 必须小于 scale_max，请在上方修正后再生成公式参数表。")
                            st.stop()
                        cfa_df_formula = st.session_state.get("n3_merged_df")
                        cfa_names_formula = st.session_state.get("n3_cfa_dataset_names", [])
                        est_formula = st.session_state.get("n3_cfa_estimates")
                        formula_measuregroup_title = (st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup"
                        formula_measure_ids = [
                            (st.session_state.get(f"n3_cfa_latent_name_{i}") or str(n)).strip() or f"F{i+1}"
                            for i, n in enumerate(cfa_names_formula)
                        ]
                        if cfa_df_formula is None or not cfa_names_formula or est_formula is None:
                            st.error("请先完成 Multi-factor 模型拟合后再导出公式参数。")
                            st.stop()
                        formula_measure_df, formula_items_df = _build_multifactor_formula_table(
                            cfa_df_formula,
                            cfa_names_formula,
                            est_formula,
                            formula_measuregroup_title,
                            formula_measure_ids,
                            formula_scale_min,
                            formula_scale_max,
                        )
                        st.session_state.n3_cfa_formula_measure_df = formula_measure_df
                        st.session_state.n3_cfa_formula_items_df = formula_items_df
                        export_flat = formula_items_df.copy()
                        for col, val in formula_measure_df.iloc[0].items():
                            if col not in export_flat.columns:
                                export_flat[col] = val
                        cols_order = ["measure_id", "measuregroup_title", "dataset", "CFA_model", "intercept_rel", "intercept_abs",
                                      "reporting_mean", "reporting_sd", "scale_min", "scale_max", "创建时间",
                                      "item_col", "item_text", "item_num", "reverse", "beta_rel", "beta_abs", "sort_order"]
                        export_flat = export_flat[[c for c in cols_order if c in export_flat.columns]]
                        st.session_state.n3_cfa_formula_csv_bytes = export_flat.to_csv(index=False).encode("utf-8-sig")
                        fbuf = io.BytesIO()
                        with pd.ExcelWriter(fbuf, engine="xlsxwriter") as w_f:
                            export_flat.to_excel(w_f, sheet_name="formula", index=False)
                        fbuf.seek(0)
                        st.session_state.n3_cfa_formula_xlsx_bytes = fbuf.getvalue()
                        st.success("已生成公式参数表。")
                    except Exception as e:
                        st.error(f"生成公式参数表失败: {e}")

                if st.session_state.get("n3_cfa_formula_items_df") is not None:
                    st.markdown("##### 公式参数表预览（与 Excel/CSV 导出格式一致）")
                    m_df = st.session_state.get("n3_cfa_formula_measure_df")
                    i_df = st.session_state.get("n3_cfa_formula_items_df")
                    if m_df is not None and i_df is not None:
                        preview_flat = i_df.copy()
                        for col, val in m_df.iloc[0].items():
                            if col not in preview_flat.columns:
                                preview_flat[col] = val
                        st.dataframe(preview_flat.head(20), use_container_width=True)
                    cf_dl1, cf_dl2, cf_dl3, cf_dl4 = st.columns(4)
                    cfa_formula_mgid = _safe_filename_part((st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup", "measuregroup")
                    cfa_formula_user = st.session_state.get("user_name", "unknown_user")
                    cfa_formula_safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(cfa_formula_user)).strip() or "unknown_user"
                    cfa_formula_today = date.today().strftime("%Y-%m-%d")
                    cfa_formula_base = f"{cfa_formula_mgid}_multi_cfa_final_score_formula_{cfa_formula_today}_{cfa_formula_safe_user}"
                    with cf_dl1:
                        if st.session_state.get("n3_cfa_formula_csv_bytes"):
                            st.download_button(
                                "⬇️ 下载公式参数表（CSV）",
                                data=st.session_state.n3_cfa_formula_csv_bytes,
                                file_name=f"{cfa_formula_base}.csv",
                                mime="text/csv",
                                key="n3_download_cfa_formula_csv",
                            )
                    with cf_dl2:
                        if st.session_state.get("n3_cfa_formula_xlsx_bytes"):
                            st.download_button(
                                "⬇️ 下载公式参数表（Excel）",
                                data=st.session_state.n3_cfa_formula_xlsx_bytes,
                                file_name=f"{cfa_formula_base}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="n3_download_cfa_formula_xlsx",
                            )
                    with cf_dl3:
                        if _DB_SAVE_AVAILABLE and save_formula_params and st.button("☁️ 保存公式参数到云端", key="n3_save_cfa_formula_cloud"):
                            m_df = st.session_state.get("n3_cfa_formula_measure_df")
                            i_df = st.session_state.get("n3_cfa_formula_items_df")
                            if m_df is not None and i_df is not None:
                                ok, msg = save_formula_params(m_df, i_df)
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(f"保存失败: {msg}")
                            else:
                                st.error("公式表未就绪，请先生成公式参数表。")
                    with cf_dl4:
                        m_df = st.session_state.get("n3_cfa_formula_measure_df")
                        i_df = st.session_state.get("n3_cfa_formula_items_df")
                        if m_df is not None and i_df is not None:
                            if build_formula_params_json:
                                json_bytes = build_formula_params_json(m_df, i_df).encode("utf-8")
                            else:
                                import json
                                def _native(v):
                                    if v is None or isinstance(v, (str, int, float, bool)):
                                        return v
                                    if hasattr(v, "item"):
                                        v = float(v) if hasattr(v, "__float__") else v
                                        return None if (v != v or abs(v) == float("inf")) else v
                                    return str(v) if hasattr(v, "isoformat") else v
                                m_dict = {k: _native(v) for k, v in m_df.iloc[0].to_dict().items()}
                                items_list = [{k: _native(v) for k, v in r.items()} for r in i_df.to_dict(orient="records")]
                                json_bytes = json.dumps({"schema_version": 1, "measure": m_dict, "items": items_list}, ensure_ascii=False, indent=2).encode("utf-8")
                            json_fn = f"{cfa_formula_base}.json"
                            st.download_button(
                                "⬇️ 下载公式参数表（JSON）",
                                data=json_bytes,
                                file_name=json_fn,
                                mime="application/json",
                                key="n3_download_cfa_formula_json",
                            )

    if model_tab_choice == "Higher-order Model":
        st.markdown("#### 高阶模型 (Higher-order Model)")
        st.caption(
            "一阶因子（F1, F2, F3…）由各子数据集的题目定义，每个一阶因子的首题载荷固定为 1；"
            "高阶因子（HigherOrderFactor）由各一阶因子定义，对 F1 的载荷固定为 1，高阶因子方差自由估计；"
            "method 因子与各一阶因子及高阶因子正交，方差固定为 1。"
        )
        ho_df = st.session_state.get("n3_merged_df")
        ho_names = st.session_state.get("n3_merged_names", [])
        if ho_df is None or not ho_names:
            st.info("请先在上方“运行前配置”中应用题目选择，生成共享建模数据。")
        else:
            # 方案A：共享配置中的 method 题目用“原始列名”存储，此处按原始列名过滤
            ho_method_selected = [c for c in st.session_state.get("n3_method_items_shared", []) if c in ho_df.columns]
            if not ho_method_selected:
                st.warning("当前未设置方法因子题目。请先回到上方“运行前配置”选择至少 1 个方法因子题目。")

            mapping_ho = "  |  ".join([f"F{i+1} = {name}" for i, name in enumerate(ho_names)])
            st.caption("当前共享配置（一阶因子对应）： " + mapping_ho + "；高阶因子 =~ F1 + F2 + …")

            run_ho_btn = st.button("运行 Higher-order CFA", type="primary", key="n3_btn_run_ho_cfa")
            if run_ho_btn:
                if not ho_method_selected:
                    st.error("请先在“运行前配置”中至少选择 1 个方法因子题目。")
                    st.stop()
                with st.spinner("正在拟合 Higher-order CFA..."):
                    model_ho, estimates_ho, stats_ho, err_ho, syntax_ho = run_higherorder_cfa(
                        ho_df, ho_names, ho_method_selected
                    )
                if err_ho:
                    st.error(err_ho)
                    if syntax_ho:
                        st.code(syntax_ho, language="text")
                else:
                    st.session_state.n3_ho_cfa_model = model_ho
                    st.session_state.n3_ho_cfa_estimates = estimates_ho
                    st.session_state.n3_ho_cfa_stats = stats_ho
                    st.session_state.n3_ho_cfa_syntax = syntax_ho
                    st.session_state.n3_ho_cfa_dataset_names = ho_names
                    st.session_state.n3_ho_method_items = ho_method_selected
                    st.success("Higher-order CFA 拟合成功。")

            # Higher-order CFA 结果（仅在本分页内显示）
            if "n3_ho_cfa_stats" in st.session_state and st.session_state.n3_ho_cfa_stats is not None:
                st.markdown("---")
                st.subheader("Higher-order CFA 结果")
                stats_ho = st.session_state.n3_ho_cfa_stats
                syntax_ho = st.session_state.get("n3_ho_cfa_syntax", "")
                est_ho = st.session_state.get("n3_ho_cfa_estimates")
                ho_names = st.session_state.get("n3_ho_cfa_dataset_names", [])
                factor_names_ho = [f"F{i+1}" for i in range(len(ho_names))]
                hof_name = "HigherOrderFactor"
                hof_names_possible = [hof_name, "HOF", "HigherOrderFactor"]

                st.markdown("#### 模型检验 (Model Test User Model)")
                chi2_ho = _get_any_stat(stats_ho, ["chi2", "Chi2"])
                dof_ho = _get_any_stat(stats_ho, ["DoF", "dof", "df"])
                p_ho = _get_any_stat(stats_ho, ["chi2 p-value", "p-value", "pvalue", "p_value"])
                st.table(pd.DataFrame({
                    "Statistic": ["Test statistic", "Degrees of freedom", "P-value (Chi-square)"],
                    "Value": [
                        f"{chi2_ho:.3f}" if not np.isnan(chi2_ho) else "N/A",
                        f"{int(dof_ho)}" if not np.isnan(dof_ho) else "N/A",
                        f"{p_ho:.4f}" if not np.isnan(p_ho) else "N/A",
                    ],
                }))

                st.markdown("#### 拟合指数 (Fit Indices)")
                fit_keys_ho = ["CFI", "TLI", "RMSEA", "SRMR", "GFI", "AGFI", "NFI", "AIC", "BIC", "SABIC", "LogL"]
                fit_key_aliases_ho = {
                    "CFI": ["CFI"],
                    "TLI": ["TLI"],
                    "RMSEA": ["RMSEA"],
                    "SRMR": ["SRMR", "srmr"],
                    "GFI": ["GFI"],
                    "AGFI": ["AGFI"],
                    "NFI": ["NFI"],
                    "AIC": ["AIC"],
                    "BIC": ["BIC"],
                    "SABIC": ["SABIC"],
                    "LogL": ["LogL", "logl", "LogLik", "loglik", "log_likelihood", "log-likelihood"],
                }
                fit_rows_ho = []
                for k in fit_keys_ho:
                    val = _get_any_stat(stats_ho, fit_key_aliases_ho.get(k, [k]))
                    if np.isnan(val):
                        disp = "N/A"
                    elif k in ("AIC", "BIC", "SABIC", "LogL") or "chi" in k.lower():
                        disp = f"{val:.2f}"
                    else:
                        disp = f"{val:.4f}"
                    fit_rows_ho.append({"Index": k, "Value": disp})
                st.dataframe(pd.DataFrame(fit_rows_ho), use_container_width=True, hide_index=True)

                st.markdown("#### 标准化高阶因子载荷 (Standardised Higher-order Loadings)")
                if est_ho is not None and not est_ho.empty:
                    load_ho = est_ho[(est_ho["op"] == "=~") | (est_ho["op"] == "~")].copy()
                    # semopy 可能将 HOF=~F1 输出为 F1~HOF（回归形式），故同时接受两种方向
                    hof_load = load_ho[
                        (load_ho["LHS"].isin(hof_names_possible) & load_ho["RHS"].isin(factor_names_ho))
                        | (load_ho["LHS"].isin(factor_names_ho) & load_ho["RHS"].isin(hof_names_possible))
                    ].copy()
                    if not hof_load.empty:
                        # 统一显示为「高阶因子 -> 一阶因子」：若当前是 F~HOF，则交换 LHS/RHS 列用于展示
                        show = hof_load.copy()
                        swap = show["RHS"].isin(hof_names_possible)
                        if swap.any():
                            lhs_vals = show.loc[swap, "RHS"].values
                            rhs_vals = show.loc[swap, "LHS"].values
                            show.loc[swap, "LHS"] = lhs_vals
                            show.loc[swap, "RHS"] = rhs_vals
                        disp_cols = [c for c in ["LHS", "RHS", "Estimate", "Std.Err", "Std.all", "P(>|z|)"] if c in show.columns]
                        num_c = [c for c in disp_cols if c in show.columns and show[c].dtype in (np.floating, np.int64)]
                        st.dataframe(show[disp_cols].style.format({c: "{:.3f}" for c in num_c}))
                    else:
                        st.caption("（高阶因子对 F1、F2、F3 的载荷见参数估计表中 LHS 为 HigherOrderFactor 的 =~ 行或 F1/F2/F3 ~ HigherOrderFactor 的 ~ 行。）")

                st.markdown("#### 题目在一阶因子上的载荷 (Factor Loadings of Items)")
                if est_ho is not None and not est_ho.empty:
                    load_ho = est_ho[(est_ho["op"] == "=~") | (est_ho["op"] == "~")].copy()
                    # 一阶因子对题目的载荷：LHS 为 F1/F2/F3，RHS 为题目名；排除潜变量对潜变量（HOF~F 或 F~F）
                    first_order_load = load_ho[load_ho["LHS"].isin(factor_names_ho) & ~load_ho["RHS"].isin(factor_names_ho + hof_names_possible)]
                    # 若 semopy 以「题目 ~ 因子」形式输出，则取 RHS=因子、LHS=题目，再统一为「因子=~题目」列顺序展示
                    if first_order_load.empty:
                        rev = load_ho[load_ho["RHS"].isin(factor_names_ho) & ~load_ho["LHS"].isin(factor_names_ho + hof_names_possible)]
                        if not rev.empty:
                            first_order_load = rev.rename(columns={"LHS": "RHS", "RHS": "LHS"})[load_ho.columns]
                    if not first_order_load.empty:
                        cols_show = [c for c in ["LHS", "op", "RHS", "Estimate", "Std.Err", "Std.all", "P(>|z|)"] if c in first_order_load.columns]
                        num_c = [c for c in cols_show if first_order_load[c].dtype in (np.floating, np.int64)]
                        st.dataframe(first_order_load[cols_show].style.format({c: "{:.3f}" for c in num_c}))
                    csv_ho = first_order_load.to_csv(index=False).encode("utf-8-sig") if not first_order_load.empty else b""
                    if csv_ho:
                        st.download_button("📥 下载一阶因子载荷表", data=csv_ho, file_name="higherorder_cfa_loadings.csv", mime="text/csv", key="n3_dl_ho_loadings")

                st.markdown("#### 高阶因子方差 (Variance of Higher-order Factor)")
                if est_ho is not None and not est_ho.empty:
                    var_rows = est_ho[(est_ho["op"] == "~~") & (est_ho["LHS"] == est_ho["RHS"])]
                    hof_var = var_rows[var_rows["LHS"].isin(hof_names_possible)]
                    if not hof_var.empty:
                        cols_v = [c for c in ["LHS", "RHS", "Estimate", "Std.Err", "Std.all", "P(>|z|)"] if c in hof_var.columns]
                        st.table(hof_var[cols_v].style.format({c: "{:.3f}" for c in cols_v if hof_var[c].dtype in (np.floating,)}))
                    else:
                        st.caption("（高阶因子方差见参数估计表中 LHS ~~ RHS 且 LHS 为高阶因子名的行。）")

                st.markdown("#### 模型语法 (Model Syntax)")
                st.code(syntax_ho, language="text")

                st.markdown("#### 📥 生成 Higher-order 模型下载表（3 sheets）")
                ho_df_for_report = st.session_state.get("n3_merged_df")
                ho_names_for_report = st.session_state.get("n3_merged_names", [])
                if ho_df_for_report is None or not ho_names_for_report:
                    st.info("请先在上方“运行前配置”中应用题目选择。")
                else:
                    st.caption("请为每个一阶潜变量命名（通常对应各 Measure 名称）。")
                    fo_display_names = []
                    for i, n in enumerate(ho_names_for_report):
                        fo_display_names.append(
                            st.text_input(
                                f"一阶潜变量 F{i+1} 名称",
                                value=str(n),
                                key=f"n3_ho_latent_name_{i}",
                            ).strip() or f"F{i+1}"
                        )
                    measuregroup_title = (st.session_state.get("n3_measuregroup_title") or "").strip()
                    if not measuregroup_title:
                        measuregroup_title = st.text_input(
                            "measuregroup_title（用于 sheet 命名）",
                            value="measuregroup",
                            key="n3_ho_measuregroup_title_fallback",
                        ).strip() or "measuregroup"

                    if st.button("生成下载表（sheet1+sheet2+题目协方差矩阵）", key="n3_btn_build_ho_report"):
                        try:
                            est_r = st.session_state.get("n3_ho_cfa_estimates")
                            stats_r = st.session_state.get("n3_ho_cfa_stats") or {}
                            sheet1, sheet2, sheet3, sheet4, cr_warning_msg = _build_higherorder_report_tables(
                                ho_df_for_report,
                                ho_names_for_report,
                                est_r,
                                stats_r,
                                fo_display_names,
                                measuregroup_title,
                            )
                            xbuf = io.BytesIO()
                            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as w:
                                sh1 = (measuregroup_title[:31] if len(measuregroup_title) > 31 else measuregroup_title) or "Model"
                                sh1 = "".join(c for c in sh1 if c not in '[]:*?/\\')
                                sh2_base = (measuregroup_title + "_fo")
                                sh2 = (sh2_base[:31] if len(sh2_base) > 31 else sh2_base) or "Model_fo"
                                sh2 = "".join(c for c in sh2 if c not in '[]:*?/\\')
                                sh3_base = (measuregroup_title + "_cov")
                                sh3 = (sh3_base[:31] if len(sh3_base) > 31 else sh3_base) or "Model_cov"
                                sh3 = "".join(c for c in sh3 if c not in '[]:*?/\\')
                                sh4_base = (measuregroup_title + "_item_cov")
                                sh4 = (sh4_base[:31] if len(sh4_base) > 31 else sh4_base) or "Model_item_cov"
                                sh4 = "".join(c for c in sh4 if c not in '[]:*?/\\')
                                sheet1.to_excel(w, sheet_name=sh1, index=False)
                                sheet2.to_excel(w, sheet_name=sh2, index=False)
                                # NOTE: 按需求暂时注释掉“潜变量方差-协方差矩阵”导出逻辑。
                                # sheet3.to_excel(w, sheet_name=sh3, index=True)
                                sheet4.to_excel(w, sheet_name=sh4, index=True)
                            xbuf.seek(0)
                            st.session_state.n3_ho_report_xlsx = xbuf.getvalue()
                            st.session_state.n3_ho_report_sheet1_preview = sheet1.copy()
                            st.session_state.n3_ho_report_sheet2_preview = sheet2.copy()
                            # NOTE: 按需求暂时注释掉“潜变量方差-协方差矩阵”预览缓存逻辑。
                            # st.session_state.n3_ho_report_sheet3_preview = sheet3.copy()
                            st.session_state.pop("n3_ho_report_sheet3_preview", None)
                            st.session_state.n3_ho_report_sheet4_preview = sheet4.copy()
                            st.session_state.n3_ho_cr_warning = cr_warning_msg or ""
                            measuregroup_id = _safe_filename_part(measuregroup_title, "measuregroup")
                            ho_user = st.session_state.get("user_name", "unknown_user")
                            ho_safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(ho_user)).strip() or "unknown_user"
                            ho_today = date.today().strftime("%Y-%m-%d")
                            st.session_state.n3_ho_report_filename = f"{measuregroup_id}_higher_cfa_report_{ho_today}_{ho_safe_user}.xlsx"
                            st.success("已生成 Higher-order 下载表。")
                        except Exception as e:
                            st.error(f"生成 Higher-order 下载表失败: {e}")

                    if st.session_state.get("n3_ho_report_sheet1_preview") is not None:
                        st.markdown("##### 预览：Model info sheet_1（前20行）")
                        st.dataframe(
                            st.session_state.n3_ho_report_sheet1_preview.head(20),
                            use_container_width=True,
                        )
                    if st.session_state.get("n3_ho_report_sheet2_preview") is not None:
                        st.markdown("##### 预览：Model info sheet_2（前20行）")
                        st.dataframe(
                            st.session_state.n3_ho_report_sheet2_preview.head(20),
                            use_container_width=True,
                        )
                    # NOTE: 按需求暂时注释掉“潜变量方差-协方差矩阵”预览展示逻辑。
                    # if st.session_state.get("n3_ho_report_sheet3_preview") is not None:
                    #     st.markdown("##### 预览：一阶潜变量方差-协方差矩阵（前20行）")
                    #     st.dataframe(
                    #         st.session_state.n3_ho_report_sheet3_preview.head(20),
                    #         use_container_width=True,
                    #     )
                    if st.session_state.get("n3_ho_report_sheet4_preview") is not None:
                        st.markdown("##### 预览：题目方差-协方差矩阵（前20行）")
                        st.dataframe(
                            st.session_state.n3_ho_report_sheet4_preview.head(20),
                            use_container_width=True,
                        )
                    if st.session_state.get("n3_ho_cr_warning"):
                        st.warning(st.session_state.n3_ho_cr_warning)

                    if st.session_state.get("n3_ho_report_xlsx"):
                        st.download_button(
                            "⬇️ 下载 Higher-order 模型表（Excel）",
                            data=st.session_state.n3_ho_report_xlsx,
                            file_name=st.session_state.get("n3_ho_report_filename", "measuregroup+measure_higher_cfa_report.xlsx"),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="n3_download_ho_report_xlsx",
                        )

                st.markdown("#### 🧮 最终得分计算（基于 Higher-order CFA）")
                st.caption("若样本存在缺失值，则该样本最终得分与 absolute_final_score 记为 -999。将同时输出 relative 最终得分（80+10z）与 absolute_final_score（0-100 绝对分）。absolute 与下方导出公式参数表将复用此处填写的 scale_min / scale_max。")
                ho_s1, ho_s2 = st.columns(2)
                with ho_s1:
                    n3_ho_scale_min = st.number_input("scale_min", min_value=0, value=1, step=1, key="n3_ho_scale_min")
                with ho_s2:
                    n3_ho_scale_max = st.number_input("scale_max", min_value=1, value=7, step=1, key="n3_ho_scale_max")
                if not (data_source_choice == "四数据集 (Dataset → Measure)"):
                    st.number_input("当前是 dataset 中的哪一个？（用于下载文件命名）", min_value=1, value=1, step=1, key="n3_ho_scored_dataset_n")
                if int(n3_ho_scale_min) >= int(n3_ho_scale_max):
                    st.error("scale_min 必须小于 scale_max。")
                elif (int(n3_ho_scale_min), int(n3_ho_scale_max)) != (1, 7):
                    st.warning("⚠️ 当前 scale_min / scale_max 非默认 [1, 7]。请确认量表边界与问卷设计一致，否则 absolute 0–100 分数解释可能产生偏差。")
                if st.button("计算最终得分并生成可下载数据集", key="n3_btn_ho_final_score"):
                    try:
                        ho_df_score = st.session_state.get("n3_merged_df")
                        ho_names_score = st.session_state.get("n3_ho_cfa_dataset_names", [])
                        est_score = st.session_state.get("n3_ho_cfa_estimates")
                        scored_df, n_missing_rows = _compute_higherorder_final_scores(
                            ho_df_score,
                            ho_names_score,
                            est_score,
                            scale_min=int(st.session_state.get("n3_ho_scale_min", 1)),
                            scale_max=int(st.session_state.get("n3_ho_scale_max", 7)),
                        )
                        st.session_state.n3_ho_scored_df = scored_df

                        csv_bytes = scored_df.to_csv(index=False).encode("utf-8-sig")
                        st.session_state.n3_ho_scored_csv_bytes = csv_bytes

                        xlsx_buf = io.BytesIO()
                        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w_x:
                            scored_df.to_excel(w_x, sheet_name="ScoredData", index=False)
                        xlsx_buf.seek(0)
                        st.session_state.n3_ho_scored_xlsx_bytes = xlsx_buf.getvalue()

                        if n_missing_rows > 0:
                            st.warning(f"检测到 {n_missing_rows} 行样本存在缺失值，这些样本的“最终得分”已记为 -999。")
                        else:
                            st.info("检测到 0 行样本存在缺失值。")
                        st.success(f"最终得分计算完成：共 {len(scored_df)} 行样本。")
                    except Exception as e:
                        st.error(f"计算 Higher-order 最终得分失败: {e}")

                if st.session_state.get("n3_ho_scored_df") is not None:
                    st.markdown("##### 数据预览（含最终得分，前20行）")
                    preview_scored = st.session_state.n3_ho_scored_df.head(20).copy()
                    if "relative_final_score" in preview_scored.columns:
                        preview_scored["relative_final_score"] = pd.to_numeric(preview_scored["relative_final_score"], errors="coerce").round(3)
                    if "absolute_final_score" in preview_scored.columns:
                        preview_scored["absolute_final_score"] = pd.to_numeric(preview_scored["absolute_final_score"], errors="coerce").round(3)
                    st.dataframe(preview_scored, use_container_width=True)

                    dl_col1, dl_col2 = st.columns(2)
                    measuregroup_title_dl = (st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup"
                    ho_mgid_dl = _safe_filename_part(measuregroup_title_dl, "measuregroup")
                    ho_user = st.session_state.get("user_name", "unknown_user")
                    ho_safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(ho_user)).strip() or "unknown_user"
                    ho_today = date.today().strftime("%Y-%m-%d")
                    if data_source_choice == "四数据集 (Dataset → Measure)":
                        ds_name_ho = st.session_state.get("n3_dual_dataset", "Dataset4")
                        m_ho = re.search(r"(\d+)", str(ds_name_ho))
                        ho_dataset_n = int(m_ho.group(1)) if m_ho else 4
                    else:
                        ho_dataset_n = int(st.session_state.get("n3_ho_scored_dataset_n", 1))
                    ho_scored_base = f"{ho_mgid_dl}_higher_cfa_dataset{ho_dataset_n}_scored_{ho_today}_{ho_safe_user}"
                    with dl_col1:
                        if st.session_state.get("n3_ho_scored_csv_bytes"):
                            st.download_button(
                                "⬇️ 下载含最终得分数据（CSV）",
                                data=st.session_state.n3_ho_scored_csv_bytes,
                                file_name=f"{ho_scored_base}.csv",
                                mime="text/csv",
                                key="n3_download_ho_scored_csv",
                            )
                    with dl_col2:
                        if st.session_state.get("n3_ho_scored_xlsx_bytes"):
                            st.download_button(
                                "⬇️ 下载含最终得分数据（Excel）",
                                data=st.session_state.n3_ho_scored_xlsx_bytes,
                                file_name=f"{ho_scored_base}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="n3_download_ho_scored_xlsx",
                            )

                st.markdown("#### 📤 导出最终得分公式参数表")
                st.caption("导出部署时使用的线性公式参数：FinalScore = intercept + Σ(beta * item)。")
                _ho_smin = int(st.session_state.get("n3_ho_scale_min", 1))
                _ho_smax = int(st.session_state.get("n3_ho_scale_max", 7))
                st.caption(f"公式参数将使用**上方** scale_min / scale_max（当前为 [{_ho_smin}, {_ho_smax}]）。若已修改 scale_min / scale_max，建议重新执行「计算最终得分」以保持数据与公式一致。若非 1–7 量表，请先在上方修改后再生成。")
                #st.caption("measure_id 将自动使用上方“生成 Higher-order 模型下载表”中填写的一阶潜变量名称（无需重复填写）。")
                ho_names_for_hint = st.session_state.get("n3_ho_cfa_dataset_names", [])
                formula_measure_ids_hint_ho = [
                    (st.session_state.get(f"n3_ho_latent_name_{i}") or str(n)).strip() or f"F{i+1}"
                    for i, n in enumerate(ho_names_for_hint)
                ]
                if formula_measure_ids_hint_ho:
                    st.info("本次将使用的 measure_id： " + " | ".join(formula_measure_ids_hint_ho))

                if st.button("生成公式参数表", key="n3_btn_build_ho_formula"):
                    try:
                        formula_scale_min_ho = int(st.session_state.get("n3_ho_scale_min", 1))
                        formula_scale_max_ho = int(st.session_state.get("n3_ho_scale_max", 7))
                        if formula_scale_min_ho >= formula_scale_max_ho:
                            st.error("scale_min 必须小于 scale_max，请在上方修正后再生成公式参数表。")
                            st.stop()
                        ho_df_formula = st.session_state.get("n3_merged_df")
                        ho_names_formula = st.session_state.get("n3_ho_cfa_dataset_names", [])
                        est_formula_ho = st.session_state.get("n3_ho_cfa_estimates")
                        formula_measuregroup_title_ho = (st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup"
                        formula_measure_ids_ho = [
                            (st.session_state.get(f"n3_ho_latent_name_{i}") or str(n)).strip() or f"F{i+1}"
                            for i, n in enumerate(ho_names_formula)
                        ]
                        if ho_df_formula is None or not ho_names_formula or est_formula_ho is None:
                            st.error("请先完成 Higher-order 模型拟合后再导出公式参数。")
                            st.stop()
                        formula_measure_df_ho, formula_items_df_ho = _build_higherorder_formula_table(
                            ho_df_formula,
                            ho_names_formula,
                            est_formula_ho,
                            formula_measuregroup_title_ho,
                            formula_measure_ids_ho,
                            formula_scale_min_ho,
                            formula_scale_max_ho,
                        )
                        st.session_state.n3_ho_formula_measure_df = formula_measure_df_ho
                        st.session_state.n3_ho_formula_items_df = formula_items_df_ho
                        export_flat_ho = formula_items_df_ho.copy()
                        for col, val in formula_measure_df_ho.iloc[0].items():
                            if col not in export_flat_ho.columns:
                                export_flat_ho[col] = val
                        cols_order_ho = ["measure_id", "measuregroup_title", "dataset", "CFA_model", "intercept_rel", "intercept_abs",
                                        "reporting_mean", "reporting_sd", "scale_min", "scale_max", "创建时间",
                                        "item_col", "item_text", "item_num", "reverse", "beta_rel", "beta_abs", "sort_order"]
                        export_flat_ho = export_flat_ho[[c for c in cols_order_ho if c in export_flat_ho.columns]]
                        st.session_state.n3_ho_formula_csv_bytes = export_flat_ho.to_csv(index=False).encode("utf-8-sig")
                        fbuf_ho = io.BytesIO()
                        with pd.ExcelWriter(fbuf_ho, engine="xlsxwriter") as w_f:
                            export_flat_ho.to_excel(w_f, sheet_name="formula", index=False)
                        fbuf_ho.seek(0)
                        st.session_state.n3_ho_formula_xlsx_bytes = fbuf_ho.getvalue()
                        st.success("已生成公式参数表。")
                    except Exception as e:
                        st.error(f"生成公式参数表失败: {e}")

                if st.session_state.get("n3_ho_formula_items_df") is not None:
                    st.markdown("##### 公式参数表预览（与 Excel/CSV 导出格式一致）")
                    m_df_ho = st.session_state.get("n3_ho_formula_measure_df")
                    i_df_ho = st.session_state.get("n3_ho_formula_items_df")
                    if m_df_ho is not None and i_df_ho is not None:
                        preview_flat_ho = i_df_ho.copy()
                        for col, val in m_df_ho.iloc[0].items():
                            if col not in preview_flat_ho.columns:
                                preview_flat_ho[col] = val
                        st.dataframe(preview_flat_ho.head(20), use_container_width=True)
                    hf_dl1, hf_dl2, hf_dl3, hf_dl4 = st.columns(4)
                    ho_formula_mgid = _safe_filename_part((st.session_state.get("n3_measuregroup_title") or "").strip() or "measuregroup", "measuregroup")
                    ho_formula_user = st.session_state.get("user_name", "unknown_user")
                    ho_formula_safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(ho_formula_user)).strip() or "unknown_user"
                    ho_formula_today = date.today().strftime("%Y-%m-%d")
                    ho_formula_base = f"{ho_formula_mgid}_higher_cfa_final_score_formula_{ho_formula_today}_{ho_formula_safe_user}"
                    with hf_dl1:
                        if st.session_state.get("n3_ho_formula_csv_bytes"):
                            st.download_button(
                                "⬇️ 下载公式参数表（CSV）",
                                data=st.session_state.n3_ho_formula_csv_bytes,
                                file_name=f"{ho_formula_base}.csv",
                                mime="text/csv",
                                key="n3_download_ho_formula_csv",
                            )
                    with hf_dl2:
                        if st.session_state.get("n3_ho_formula_xlsx_bytes"):
                            st.download_button(
                                "⬇️ 下载公式参数表（Excel）",
                                data=st.session_state.n3_ho_formula_xlsx_bytes,
                                file_name=f"{ho_formula_base}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="n3_download_ho_formula_xlsx",
                            )
                    with hf_dl3:
                        if _DB_SAVE_AVAILABLE and save_formula_params and st.button("☁️ 保存公式参数到云端", key="n3_save_ho_formula_cloud"):
                            m_df_ho = st.session_state.get("n3_ho_formula_measure_df")
                            i_df_ho = st.session_state.get("n3_ho_formula_items_df")
                            if m_df_ho is not None and i_df_ho is not None:
                                ok, msg = save_formula_params(m_df_ho, i_df_ho)
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(f"保存失败: {msg}")
                            else:
                                st.error("公式表未就绪，请先生成公式参数表。")
                    with hf_dl4:
                        m_df_ho = st.session_state.get("n3_ho_formula_measure_df")
                        i_df_ho = st.session_state.get("n3_ho_formula_items_df")
                        if m_df_ho is not None and i_df_ho is not None:
                            if build_formula_params_json:
                                json_bytes_ho = build_formula_params_json(m_df_ho, i_df_ho).encode("utf-8")
                            else:
                                import json
                                def _native(v):
                                    if v is None or isinstance(v, (str, int, float, bool)):
                                        return v
                                    if hasattr(v, "item"):
                                        v = float(v) if hasattr(v, "__float__") else v
                                        return None if (v != v or abs(v) == float("inf")) else v
                                    return str(v) if hasattr(v, "isoformat") else v
                                m_dict_ho = {k: _native(v) for k, v in m_df_ho.iloc[0].to_dict().items()}
                                items_list_ho = [{k: _native(v) for k, v in r.items()} for r in i_df_ho.to_dict(orient="records")]
                                json_bytes_ho = json.dumps({"schema_version": 1, "measure": m_dict_ho, "items": items_list_ho}, ensure_ascii=False, indent=2).encode("utf-8")
                            ho_json_fn = f"{ho_formula_base}.json"
                            st.download_button(
                                "⬇️ 下载公式参数表（JSON）",
                                data=json_bytes_ho,
                                file_name=ho_json_fn,
                                mime="application/json",
                                key="n3_download_ho_formula_json",
                            )
