# EFA & CFA 双数据集模式：合并、四数据集、measure 划分与统计
import re
import io
import zipfile
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import normalize_item_text, get_item_columns, parse_item_col, sort_item_cols_by_number


def assign_efa_item_numbers(df: pd.DataFrame, item_cols: Optional[List[str]] = None, start_col: Optional[str] = None) -> pd.DataFrame:
    """为 EFA 题目列按顺序赋 EFA1_, EFA2_, ... 若提供 start_col，则仅从该列起（含）编号，该列之前的列保持原名。"""
    df = df.copy()
    all_cols = df.columns.tolist()
    if start_col is not None and start_col in all_cols:
        start_idx = all_cols.index(start_col)
        # 仅对 start_col 及之后的列中、尚未带 EFA/CFA 前缀的列编号
        item_cols = [c for c in all_cols[start_idx:] if isinstance(c, str) and not re.match(r"^(EFA|CFA)\d+_", c)]
        prefix_cols = all_cols[:start_idx]  # 保持原名
    else:
        prefix_cols = []
        if item_cols is None:
            item_cols = [c for c in df.columns if isinstance(c, str) and not re.match(r"^(EFA|CFA)\d+_", c)]
    rename = {}
    for i, col in enumerate(item_cols, 1):
        if col in df.columns:
            rename[col] = f"EFA{i}_{col}"
    df = df.rename(columns=rename)
    return df


def align_cfa_to_efa(
    df_cfa: pd.DataFrame,
    efa_item_cols: List[str],
    efa_text_to_col: Dict[str, str],
    start_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    CFA 题目列按题目文本与 EFA 对齐：能匹配到的用 EFA 列名，匹配不到的用 CFA1_, CFA2_, ...
    start_col: 若指定，仅从该列起（含）对齐；该列之前的列保持原名。
    """
    df = df_cfa.copy()
    all_cols = [c for c in df.columns if isinstance(c, str)]
    cfa_item_cols = [c for c in all_cols if not re.match(r"^(EFA|CFA)\d+_", c)]
    if start_col is not None and start_col in all_cols:
        start_idx = all_cols.index(start_col)
        cfa_item_cols = [c for c in cfa_item_cols if all_cols.index(c) >= start_idx]
    rename = {}
    matched_efa = []
    cfa_only_cols = []
    cfa_only_idx = 1
    for col in cfa_item_cols:
        norm = normalize_item_text(col)
        if norm in efa_text_to_col:
            efa_col = efa_text_to_col[norm]
            rename[col] = efa_col
            matched_efa.append(efa_col)
        else:
            new_name = f"CFA{cfa_only_idx}_{col}"
            rename[col] = new_name
            cfa_only_cols.append(new_name)
            cfa_only_idx += 1
    df = df.rename(columns=rename)
    return df, matched_efa, cfa_only_cols


def build_efa_text_to_col(efa_item_cols: List[str]) -> Dict[str, str]:
    """从 EFA 题目列名建立 规范化题目文本 -> 列名 的映射。"""
    out = {}
    for col in efa_item_cols:
        pre, num, text = parse_item_col(col)
        if pre == "EFA" and text:
            out[normalize_item_text(text)] = col
        else:
            # 无前缀时整列名当作文本
            out[normalize_item_text(col)] = col
    return out


def build_four_datasets(
    df_efa: pd.DataFrame,
    df_cfa: pd.DataFrame,
    efa_item_cols: List[str],
    cfa_item_cols: List[str],
    cfa_only_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    构建四个数据集（均保留所有列）。
    - Dataset1: 清洗后 EFA，全部列
    - Dataset2: 清洗后 CFA，全部列
    - Dataset3: 合并 EFA+CFA 行，全部题目列 + 非题目列 + source
    - Dataset4: 同上，但只保留 CFA 题目列 + 非题目列 + source
    """
    all_item_cols = list(dict.fromkeys(efa_item_cols + cfa_item_cols))
    non_item_efa = [c for c in df_efa.columns if c not in efa_item_cols]
    non_item_cfa = [c for c in df_cfa.columns if c not in cfa_item_cols]
    common_non_item = list(dict.fromkeys(non_item_efa + non_item_cfa))

    d1 = df_efa.copy()
    d2 = df_cfa.copy()

    df_efa_rows = df_efa.copy()
    df_efa_rows["source"] = "EFA"
    df_cfa_rows = df_cfa.copy()
    df_cfa_rows["source"] = "CFA"

    cols_d3 = [c for c in common_non_item if c in df_efa.columns or c in df_cfa.columns]
    cols_d3 = list(dict.fromkeys(cols_d3 + ["source"] + all_item_cols))
    for c in all_item_cols:
        if c not in df_efa_rows.columns:
            df_efa_rows[c] = np.nan
        if c not in df_cfa_rows.columns:
            df_cfa_rows[c] = np.nan
    for c in cols_d3:
        if c not in df_efa_rows.columns:
            df_efa_rows[c] = np.nan
        if c not in df_cfa_rows.columns:
            df_cfa_rows[c] = np.nan
    d3 = pd.concat([df_efa_rows[cols_d3], df_cfa_rows[cols_d3]], axis=0, ignore_index=True)

    cfa_item_set = set(cfa_item_cols)
    cols_d4 = [c for c in cols_d3 if c == "source" or c in cfa_item_set or (c in common_non_item and c in d3.columns)]
    d4 = d3[cols_d4].copy()

    return d1, d2, d3, d4


def build_four_datasets_from_merged(
    merged_df: pd.DataFrame,
    efa_item_cols: List[str],
    cfa_item_cols: List[str],
    source_col: str = "source",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从已合并且带「数据来源」列的 DataFrame 生成四份数据集。
    - Dataset1: 数据来源=EFA 的行，全部列（cleaned EFA with all items）
    - Dataset2: 数据来源=CFA 的行，仅 CFA 题目列（cleaned CFA with CFA items only）
    - Dataset3: 全部行，全部题目列（EFA 行在 CFA 独有题上为 NaN）（cleaned merged, all items）
    - Dataset4: 全部行，仅 CFA 题目列（cleaned merged, CFA items only）
    """
    if source_col not in merged_df.columns:
        raise ValueError(f"合并表需包含列「{source_col}」")
    all_item_cols = list(dict.fromkeys(efa_item_cols + cfa_item_cols))
    all_item_cols = [c for c in all_item_cols if c in merged_df.columns]
    cfa_item_set = set(c for c in cfa_item_cols if c in merged_df.columns)
    non_item = [c for c in merged_df.columns if c not in all_item_cols and c != source_col]

    d1 = merged_df[merged_df[source_col] == "EFA"].copy().reset_index(drop=True)
    d2 = merged_df[merged_df[source_col] == "CFA"].copy()
    d2_cols = [c for c in merged_df.columns if c in cfa_item_set or c in non_item or c == source_col]
    d2 = d2[d2_cols].reset_index(drop=True)

    cols_d3 = non_item + [source_col] + all_item_cols
    cols_d3 = [c for c in cols_d3 if c in merged_df.columns]
    d3 = merged_df[cols_d3].copy().reset_index(drop=True)
    cols_d4 = [c for c in cols_d3 if c == source_col or c in cfa_item_set or c in non_item]
    d4 = merged_df[cols_d4].copy().reset_index(drop=True)

    return d1, d2, d3, d4


def item_mean_sd_table(df: pd.DataFrame, item_cols: List[str]) -> pd.DataFrame:
    """每个题目一行：item number (CFA/EFA), item text, item mean, item sd；按题号升序。"""
    item_cols = [c for c in item_cols if c in df.columns]
    item_cols = sort_item_cols_by_number(item_cols)
    df_num = df[item_cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in item_cols:
        pre, num, text = parse_item_col(col)
        label = f"{pre}{num}" if pre else str(num)
        rows.append({
            "item_number": label,
            "item_text": text or col,
            "item_mean": float(df_num[col].mean()),
            "item_sd": float(df_num[col].std()),
        })
    return pd.DataFrame(rows)


def item_covariance_matrix(df: pd.DataFrame, item_cols: List[str]) -> pd.DataFrame:
    """题目协方差矩阵：行列按题号升序，对角线为方差，非对角线为协方差。"""
    item_cols = [c for c in item_cols if c in df.columns]
    item_cols = sort_item_cols_by_number(item_cols)
    df_num = df[item_cols].apply(pd.to_numeric, errors="coerce")
    cov = df_num.cov()
    return cov


def get_measure_item_columns_in_dataset(
    dataset_df: pd.DataFrame,
    measure_item_names: List[str],
) -> List[str]:
    """某数据集中属于该 measure 的题目列（取交集）。"""
    have = set(dataset_df.columns)
    return [c for c in measure_item_names if c in have]


def get_dual_mode_analysis_df(
    dataset_name: str,
    measure_names: List[str],
    dc_dataset_full: dict,
    dc_measures: dict,
    item_columns_only: bool = True,
) -> Optional[pd.DataFrame]:
    """
    双数据集模式下，根据选择的 Dataset 和 Measure(s) 返回用于分析的 DataFrame。
    - dataset_name: "Dataset1" ~ "Dataset4"
    - measure_names: 选中的 measure 名称列表
    - item_columns_only: True 时只返回题目列（供 N1/N2/N3 分析用）
    """
    if not dc_dataset_full or dataset_name not in dc_dataset_full or not measure_names or not dc_measures:
        return None
    df = dc_dataset_full[dataset_name].copy()
    cols = []
    for m in measure_names:
        cols.extend(dc_measures.get(m, []))
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    if not cols:
        return None
    df = df[cols]
    if item_columns_only:
        item_cols = get_item_columns(df)
        df = df[[c for c in df.columns if c in item_cols]]
    return df
