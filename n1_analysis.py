import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import shapiro
from utils import smart_multiselect, parse_item_col, sort_item_cols_by_number
import re
import io
from datetime import date
from difflib import SequenceMatcher
from scipy.stats import chi2
from typing import Any, Tuple


try:
    from db_save import save_formula_params, save_score_records, build_formula_params_json
    _DB_SAVE_AVAILABLE = True
except ImportError:
    save_formula_params = save_score_records = None
    build_formula_params_json = None
    _DB_SAVE_AVAILABLE = False




# ==============================================================================
# EFA 核心算法区域
# ==============================================================================

def sort_items_by_number(items):
    """
    按照题目序号从小到大排序题目列表
    支持的格式：1_xxx, 2_xxx, 01_xxx, 001_xxx 等
    """
    def extract_number(item):
        if isinstance(item, str):
            # 匹配开头的数字部分
            match = re.match(r'^(\d+)', item)
            if match:
                return int(match.group(1))
        return float('inf')  # 没有数字的排在最后

    return sorted(items, key=extract_number)

def sort_dataframe_by_item_names(df, item_column=None):
    """
    按照题目名称的序号排序DataFrame
    如果item_column为None，则使用索引
    """
    if item_column and item_column in df.columns:
        # 按指定列排序
        sorted_items = sort_items_by_number(df[item_column].tolist())
        df_sorted = df.set_index(item_column).reindex(sorted_items).reset_index()
        return df_sorted
    elif item_column is None and isinstance(df.index, pd.Index):
        # 按索引排序
        sorted_index = sort_items_by_number(df.index.tolist())
        df_sorted = df.reindex(sorted_index)
        return df_sorted
    else:
        return df

def sort_table(loadings: pd.DataFrame, X=None) -> pd.DataFrame:
    L = loadings.copy()
    scores = (L ** 2).sum(axis=0).sort_values(ascending=False)
    L = L.loc[:, scores.index]
    return L

def pca_algo(df, graph=False):
    """
    修改后：直接在全样本上计算 PCA，剔除 Bootstrap
    """
    items = df.select_dtypes(include=[np.number])
    items = items.replace([np.inf, -np.inf], np.nan).dropna()
    items = items[~items.isin([np.inf, -np.inf]).any(axis=1)]

    if items.empty:
        return pd.DataFrame(), 1

    try:
        # 直接使用全样本标准化
        Z = StandardScaler().fit_transform(items.values)
        if np.isnan(Z).any() or np.isinf(Z).any():
            st.error("标准化后数据仍包含NaN或Inf值，请检查原始数据质量")
            return pd.DataFrame(), 1
    except Exception as e:
        st.error(f"数据标准化失败: {e}")
        return pd.DataFrame(), 1

    try:
        # 使用全样本评估特征值
        fa_tmp = FactorAnalyzer(n_factors=items.shape[1], rotation=None, method='minres')
        fa_tmp.fit(Z)
        eigen_all = fa_tmp.get_eigenvalues()[0]

        n_factors = int(np.sum(eigen_all > 1))
        if n_factors < 1:
            n_factors = 1

        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='minres')
        fa.fit(Z)
    except Exception as e:
        st.error(f"PCA因子分析失败: {e}")
        return pd.DataFrame(), 1
    
    eigen_extract = eigen_all[:n_factors]
    var_ratio = eigen_extract / eigen_all.sum()
    cum_ratio = np.cumsum(var_ratio)
    
    table1 = pd.DataFrame({
        "Component": [f"Factor{i+1}" for i in range(n_factors)],
        "Initial Eigenvalues": eigen_extract,
        "% of Variance": var_ratio * 100,
        "Cumulative %": cum_ratio * 100
    })

    n_elbow = 0
    if len(eigen_all) > 1:
        x = np.arange(1, len(eigen_all)+1)
        line = np.array([[x[0], eigen_all[0]], [x[-1], eigen_all[-1]]])
        dist = np.abs(np.cross(line[1]-line[0], np.column_stack((x, eigen_all))-line[0])) / \
               np.linalg.norm(line[1]-line[0])
        n_elbow = int(np.argmax(dist)) + 1
    
    if graph:
        st.markdown("### Total Variance Explained")
        st.dataframe(table1.style.format("{:.3f}"))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x_axis = np.arange(1, len(eigen_all)+1)
        ax.plot(x_axis, eigen_all, marker='o', label='Eigenvalues')
        ax.axvline(n_elbow, color='r', linestyle='--', label=f'Elbow (factor={n_elbow})')
        ax.set_xlabel('Component / Factor Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.legend()
        st.pyplot(fig)      
    return table1, n_elbow

def dscpt_stats_mode(val_list):
    mode_result = stats.mode(val_list, keepdims=False)
    if hasattr(mode_result, 'mode'):
        # scipy >= 1.9.0 返回 ModeResult 对象
        mode_val = int(mode_result.mode)
    else:
        # 旧版scipy返回数组
        mode_val = int(mode_result[0])
    return mode_val

def bootstrap_pca(df, boot_time=20):
    """
    修改后：完全取消 Bootstrap 抽样，直接返回全样本的碎石图拐点
    """
    _, factor_num_final = pca_algo(df, graph=False)
    return factor_num_final




def align_loadings(raw_loadings, ref):
    L = raw_loadings.reindex(index=ref.index)
    S = np.zeros((L.shape[1], ref.shape[1]))
    for i, c1 in enumerate(L.columns):
        a = L[c1].fillna(0).values
        for j, c2 in enumerate(ref.columns):
            b = ref[c2].fillna(0).values
            if np.all(a == 0) or np.all(b == 0) or np.std(a) == 0 or np.std(b) == 0:
                S[i, j] = 0.0
            else:
                S[i, j] = abs(np.corrcoef(a, b)[0, 1])
    cost = 1 - S
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = L.iloc[:, row_ind].copy()
    aligned.columns = [ref.columns[j] for j in col_ind]
    return aligned

def efa_once(df, k, scaler=None):
    """
    单次 EFA 核心计算（直接在传入的完整数据集上运行）
    """
    X = df.select_dtypes(include=[np.number]).dropna(axis=0, how='any')
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]

    if X.empty:
        raise RuntimeError("没有合法的数值列可用于全样本 EFA 分析")

    if scaler is None:
        scaler = StandardScaler()

    try:
        Z = scaler.fit_transform(X)
    except Exception as e:
        raise RuntimeError(f"全样本数据标准化失败: {e}")

    if np.isnan(Z).any() or np.isinf(Z).any():
        raise RuntimeError("标准化后的全样本数据包含 NaN 或 Inf 值")

    # 保持原代码的 minres 和 varimax 旋转
    fa = FactorAnalyzer(n_factors=k, rotation='varimax', method='minres')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(Z)
    except Exception as e:
        raise RuntimeError(f"全样本 EFA 模型拟合/旋转失败: {e}")

    loadings = pd.DataFrame(fa.loadings_, index=X.columns, columns=[f'F{i+1}' for i in range(k)])
    return sort_table(loadings, X)

def bootstrap_efa(df, factor_num, boot_time=50, max_retry=3, random_state=42):
    loadings_list = []
    ref_loadings = None
    rng = np.random.default_rng(random_state)

    X = df.select_dtypes(include=[np.number]).dropna(axis=0, how='any')
    n_rows = len(X)
    if n_rows < 5:
        raise RuntimeError("Not enough rows.")
    sample_size = max(int(n_rows * 0.8), min(n_rows - 1, n_rows))

    for _ in range(boot_time):
        tries = 0
        while tries <= max_retry:
            tries += 1
            idx = rng.choice(X.index, size=sample_size, replace=False)
            try:
                raw = efa_once(X.loc[idx], factor_num)
                if ref_loadings is None:
                    ref_loadings = raw
                    loadings_list.append(raw)
                else:
                    aligned = align_loadings(raw, ref_loadings)
                    loadings_list.append(aligned)
                break 
            except RuntimeError:
                if tries > max_retry:
                    break
                continue
    return loadings_list

def calculate_loadings_avg(current_df, factor_num_final):
    """
    【彻底修复版】完全移除所有对齐逻辑、循环、重试和标准差计算。
    只在全样本上运行一次 EFA，直接返回单次运行得到的干净的 DataFrame。
    """
    try:
        # 1. 提取纯数字列，防止类型硬伤
        X = current_df.select_dtypes(include=[np.number]).dropna(axis=0, how='any')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]

        if X.empty or X.shape[1] < factor_num_final:
            raise RuntimeError(f"有效数字题目列数 ({X.shape[1]}) 小于指定的因子数 ({factor_num_final})，无法分析。")

        # 2. 直接在全样本上进行标准化
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(X)
        
        if np.isnan(Z).any() or np.isinf(Z).any():
            raise RuntimeError("标准化后的数据包含 NaN 或 Inf 值，请检查数据是否存在某题所有人得分一样（方差为0）。")

        # 3. 运行 FactorAnalyzer (写死稳定的最小残差法 minres 和正交旋转 varimax)
        # 如果你的业务需要斜交旋转，可以把 varimax 改成 promax
        fa = FactorAnalyzer(n_factors=factor_num_final, rotation='varimax', method='minres')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(Z)
        
        # 4. 组装成带有题项名称作为行索引的 DataFrame 载荷矩阵
        loadings_final = pd.DataFrame(
            fa.loadings_, 
            index=X.columns, 
            columns=[f'F{i+1}' for i in range(factor_num_final)]
        )
        
        # 5. 调用原脚本中自带的排序函数对表格进行格式化
        return sort_table(loadings_final, X)

    except Exception as e:
        # 把底层真正的数学报错或者工程报错抛出来，不再隐藏在“全样本失败”的套话里
        raise RuntimeError(f"底层全样本 EFA 执行失败，病灶原因为: {e}")

def _primary_factor_and_cross(loadings_row):
    abs_vals = loadings_row.abs().values
    order = np.argsort(abs_vals)[::-1]
    p_idx = int(order[0])
    p = float(abs_vals[p_idx])
    s = float(abs_vals[order[1]]) if len(order) > 1 else 0.0
    return p_idx, p, s

def delete_items(loadings_avg, current_df, k, min_items_per_factor=3, 
                 min_primary_loading=0.40, min_cross_loading=0.40, cross_ratio=0.70, min_communality=0.30,
                 whitelist=None):
    """
    增强的删题策略，包含三类删除标准：
      1) 主载荷低于 min_primary_loading
      2) 强交叉载荷（次/主比 > cross_ratio 且 次载荷 > min_cross_loading）
      3) 共同度（communality）低于动态设定的阈值（当对应因子题目数<5时，降至0.2）

    逻辑：
      - 首先进行预扫描，统计每个因子当前分到的题目总数
      - 动态调整共同度阈值：如果某因子当前题目数 < 5，则其共同度判汰标准降为 0.20
      - 根据主载荷 / 交叉载荷 / 共同度进行 item 分类（mutually exclusive，按优先级：主载荷 -> 交叉 -> 共同度）
      - 对每类候选按严重程度排序（主载荷和共同度越低越优先；交叉按 次/主 比值从高到低）
      - 删除时保证不会将某个因子的题目数降到 min_items_per_factor 以下
    """
    primary_assign = {}
    candidates_low = []    # (item, p_idx, primary_loading)
    candidates_cross = []  # (item, p_idx, ratio, second_loading)
    candidates_comm = []   # (item, p_idx, communality, dynamic_threshold)

    if whitelist is None:
        whitelist = []
        
    # 共同度由平均载荷矩阵计算：每题的 communality = sum(loadings^2)
    communalities = (loadings_avg ** 2).sum(axis=1)

    # ---------------------------------------------------------
    # 第一步：【新增】预扫描，统计每个因子当前的初始题目数
    # ---------------------------------------------------------
    current_counts = np.zeros(k, dtype=int)
    for item, row in loadings_avg.iterrows():
        p_idx, p, s = _primary_factor_and_cross(row)
        primary_assign[item] = p_idx
        current_counts[p_idx] += 1

    # ---------------------------------------------------------
    # 第二步：循环检查每道题目，应用动态共同度标准
    # ---------------------------------------------------------
    for item, row in loadings_avg.iterrows():
        # 如果题目在白名单里，直接跳过检查，绝不删除
        if item in whitelist:
            continue 
        
        p_idx = primary_assign[item]
        # 获取预扫描出来的指标
        _, p, s = _primary_factor_and_cross(row)

        # 【核心改动】：动态调整共同度门槛
        # 如果该题所属的因子目前题目总数小于 5 题，标准降到 0.20；否则维持传入的默认值（如 0.30）
        if current_counts[p_idx] < 5:
            current_min_comm = 0.20
        else:
            current_min_comm = min_communality

        # 优先判定：主载荷过低
        if p < min_primary_loading:
            candidates_low.append((item, p_idx, p))
        # 再判定：强交叉载荷
        elif s > min_cross_loading and (p > 0) and (s / p) > cross_ratio:
            candidates_cross.append((item, p_idx, (s / p), s))
        # 再判定：共同度过低（此处使用动态计算出的 current_min_comm）
        elif communalities.loc[item] < current_min_comm:
            candidates_comm.append((item, p_idx, float(communalities.loc[item]), current_min_comm))

    # 如果没有任何候选，直接返回
    if not candidates_low and not candidates_cross and not candidates_comm:
        return None

    # 排序：主载荷越低越先删除；共同度越低越先删除；交叉按 ratio 从高到低
    candidates_low.sort(key=lambda t: t[2])            # ascending primary loading
    candidates_comm.sort(key=lambda t: t[2])           # ascending communality (lower worse)
    candidates_cross.sort(key=lambda t: t[2], reverse=True)  # descending ratio

    # 合并优先级：低主载荷 -> 低共同度 -> 交叉载荷
    # 注意：为了统一解包格式，我们在 comm 的元组里把动态门槛传递过去
    merged = [('low', item, p_idx, p, None) for item, p_idx, p in candidates_low] + \
             [('comm', item, p_idx, comm, thres) for item, p_idx, comm, thres in candidates_comm] + \
             [('cross', item, p_idx, ratio, s) for item, p_idx, ratio, s in candidates_cross]

    # 按合并后的优先级依次尝试删除第一个合适的题目
    for tag, item, p_idx, metric, extra in merged:
        # 不允许把某个因子题目数降到 <= min_items_per_factor (默认 3)
        if current_counts[p_idx] <= min_items_per_factor:
            continue
        
        msg = ""
        if tag == 'low':
            msg = f"删除题目 **{item}**：因子载荷过低 (主载荷={metric:.3f} < {min_primary_loading}) [所属因子当前含 {current_counts[p_idx]} 题]"
        elif tag == 'cross':
            msg = f"删除题目 **{item}**：强交叉载荷 (次/主比={metric:.3f} > {cross_ratio}，次载荷≈{extra:.3f}) [所属因子当前含 {current_counts[p_idx]} 题]"
        elif tag == 'comm':
            # extra 存放的是这一轮该题触发的动态共同度门槛值
            msg = f"删除题目 **{item}**：共同度过低 (Communality={metric:.3f} < 动态阈值 {extra:.2f}) [因该因子题目数 {current_counts[p_idx]} < 5，阈值已自动降为 0.20]"
        
        st.write(f"🛑 {msg}") 
        return item

    # 若未在上面找到合适且满足因子保护（min_items_per_factor）的题目，兜底删除最严重的一个
    if merged:
        _tag, item, p_idx, _metric, _extra = merged[0]
        msg = ""
        if _tag == 'low':
            msg = f"删除题目 **{item}**：因子载荷过低 (主载荷={_metric:.3f}) [兜底删除]"
        elif _tag == 'cross':
            msg = f"删除题目 **{item}**：强交叉载荷 (次/主比={_metric:.3f}) [兜底删除]"
        elif _tag == 'comm':
            msg = f"删除题目 **{item}**：共同度过低 (Communality={_metric:.3f} < 动态阈值 {_extra:.2f}) [兜底删除]"
        
        st.write(f"🛑 {msg}") 
        return item
        
    return None

def run_pipeline_streamlit(df, fixed_factors=None, max_iterations=100, whitelist=None):
    """
    整个迭代删题的主管道，内部逻辑保持完全一致，但调用的底层函数已全面改为全样本计算
    """
    current_df = df.select_dtypes(include=[np.number]).copy()
    
    # 1. 确定因子数量
    factor_num_final = 1
    if fixed_factors is not None:
        factor_num_final = int(fixed_factors)
        st.info(f"ℹ️ 使用用户手动指定的因子数量: **{factor_num_final}**")
    else:
        with st.spinner("正在通过全样本评估最佳因子数量..."):
            factor_num_final = bootstrap_pca(current_df)
        st.success(f"✅ 全样本建议的因子数: **{factor_num_final}**")

    # 2. 首次计算全样本因子载荷
    with st.spinner(f"正在基于 {factor_num_final} 个因子进行全样本 EFA 计算..."):
        loadings_table_avg = calculate_loadings_avg(current_df, factor_num_final)

    # 3. 循环迭代删题流程
    seen = set()
    iteration = 0
    deleted_items = []

    st.markdown("### 🔄 开始迭代删题")
    status_container = st.empty() 
    
    item_to_delete = delete_items(loadings_table_avg, current_df, factor_num_final, whitelist=whitelist)

    while item_to_delete is not None and iteration < max_iterations:
        if item_to_delete in seen:
            st.warning(f"⚠️ 检测到重复建议删除 {item_to_delete}")
            break

        if current_df.shape[1] <= 3:
            st.warning("⚠️ 剩余题目数量已降至 3 题，为了保证模型可识别性，停止继续删题。")
            break
        
        seen.add(item_to_delete)
        current_df = current_df.drop(columns=[item_to_delete])
        deleted_items.append(item_to_delete)

        status_container.info(f"正在进行第 {iteration + 1} 轮迭代计算 (已删除 {len(deleted_items)} 题)...")
        
        # 迭代时也是直接用全样本更新载荷
        loadings_table_avg = calculate_loadings_avg(current_df, factor_num_final)
        item_to_delete = delete_items(loadings_table_avg, current_df, factor_num_final, whitelist=whitelist)

        iteration += 1

    status_container.success(f"迭代完成！共进行了 {iteration} 轮。")
    
    if iteration >= max_iterations:
        st.warning("达到最大迭代次数。")

    kept_items = list(current_df.columns)
    
    return current_df, loadings_table_avg, kept_items, deleted_items, factor_num_final

def cronbach_alpha(df: pd.DataFrame) -> float:
    """计算 Cronbach's Alpha (基于 statsCriteriaCheck.py)"""
    # 确保只处理数值且无缺失
    df = df.select_dtypes(include=[np.number]).dropna()
    k = df.shape[1]
    if k < 2:
        return np.nan
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    # 防止分母为0
    if total_var == 0:
        return 0.0
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

def alpha_after_removal(df: pd.DataFrame) -> pd.DataFrame:
    """计算删除每一项后的 Alpha 变化 (新增功能)"""
    res = []
    df_num = df.select_dtypes(include=[np.number]).dropna()
    for col in df_num.columns:
        tmp = df_num.drop(columns=[col])
        res.append({'删除的题项': col, "Cronbach's α": cronbach_alpha(tmp)})
    return pd.DataFrame(res).sort_values("Cronbach's α", ascending=False).reset_index(drop=True)




def calculate_item_total_correlation(df):
    """
    Calculates Corrected Item-Total Correlation.
    Correlation between an item and the sum of the remaining items.
    """
    results = []
    # Calculate sum of all items
    total_score = df.sum(axis=1)
    
    for col in df.columns:
        # Subtract the item itself to get 'Corrected' total
        corrected_total = total_score - df[col]
        corr = df[col].corr(corrected_total)
        results.append({'Item': col, 'Item-Total Corr': corr})
        
    return pd.DataFrame(results).sort_values('Item-Total Corr', ascending=False).set_index('Item')

def check_residual_normality(df, loadings):
    """
    Calculates residuals (Observed Correlation - Reproduced Correlation)
    and performs Shapiro-Wilk test on off-diagonal elements.
    """
    # 1. Observed Correlation Matrix
    obs_corr = df.corr().values
    
    # 2. Reproduced Correlation Matrix (Loadings * Loadings_Transpose)
    # Ensure loadings are aligned with df columns
    L = loadings.loc[df.columns].values 
    reproduced_corr = np.dot(L, L.T)
    
    # 3. Residual Matrix
    residuals = obs_corr - reproduced_corr
    
    # 4. Extract off-diagonal elements (upper triangle) for normality test
    mask = np.triu(np.ones_like(residuals, dtype=bool), k=1)
    res_values = residuals[mask]
    
    # 5. Shapiro-Wilk Test
    if len(res_values) >= 3:
        stat, p_value = shapiro(res_values)
    else:
        stat, p_value = 0, 0
        
    return residuals, res_values, stat, p_value





# ==============================================================================
# 页面渲染逻辑
# ==============================================================================


def render_stage1_efa_clean():
    st.subheader("删题EFA分析 (批量模式)")
    st.caption("将自动读取您在数据清洗阶段划分的所有 Measure，并依次执行全样本自动化迭代删题。")

    # ==========================================================================
    # 1. 数据来源与 Measure 自动化识别
    # ==========================================================================
    st.sidebar.markdown("### 数据来源设置")
    has_cached_data = 'sub_datasets' in st.session_state and len(st.session_state.sub_datasets) > 0
    has_dual_data = (
        st.session_state.get("dc_merge_done")
        and st.session_state.get("dc_dataset_full")
        and st.session_state.get("dc_measures")
    )

    source_options = ["📤 上传新文件 (Excel/CSV)"]
    if has_cached_data:
        source_options.append("💾 来自 Data Cleaning（子数据集）")
    if has_dual_data:
        source_options.append("💾 来自 Data Cleaning（四数据集）")

    data_source = st.radio("请选择数据来源:", source_options, horizontal=True)

    # 用字典统一存储待分析的 { "Measure名称": pd.DataFrame(包含其所属题项) }
    measures_to_process = {}
    user_selected_dataset = "Dataset1" # 默认

    if data_source == "💾 来自 Data Cleaning（四数据集）":
        from data_cleaning_dual import get_dual_mode_analysis_df
        dataset_names = ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]
        selected_dataset = st.selectbox("1. 选择数据集", dataset_names, key="n1_dual_dataset")
        user_selected_dataset = selected_dataset
        
        measure_names = list(st.session_state.dc_measures.keys())
        if not measure_names:
            st.warning("请在数据清洗模块的「Measure 划分」中至少定义一个 Measure。")
        else:
            selected_measures = st.multiselect(
                "2. 选择要批量运行的 Measure（可多选）",
                measure_names,
                default=measure_names,  # 默认全选，体现批量效率
                key="n1_dual_measures",
            )
            
            if selected_measures:
                for m in selected_measures:
                    df_m = get_dual_mode_analysis_df(
                        selected_dataset,
                        [m],
                        st.session_state.dc_dataset_full,
                        st.session_state.dc_measures,
                        item_columns_only=True,
                    )
                    if df_m is not None and df_m.shape[1] >= 3:
                        measures_to_process[m] = df_m

    elif data_source == "💾 来自 Data Cleaning（子数据集）":
        dataset_names = list(st.session_state.sub_datasets.keys())
        if not dataset_names:
            st.warning("暂无已保存的子数据集，请先前往数据清洗页面保存。")
        else:
            # ==============================================================
            # 🚨 【核心改造】将 st.selectbox 升级为 st.multiselect，支持多选子数据集！
            # ==============================================================
            selected_names = st.multiselect(
                "1. 请选择要批量运行的已保存子数据集（可多选）:", 
                options=dataset_names,
                default=dataset_names, # 默认全选，极大地提高效率
                key="n1_batch_sub_datasets"
            )
            
            if selected_names:
                # 建立一个临时存储，用来汇总所有选中的数据集里捞出来的 Measure
                saved_measures_found = {}
                
                # 第一层循环：遍历用户选中的每一个子数据集
                for current_dataset_name in selected_names:
                    df_sub = st.session_state.sub_datasets[current_dataset_name]
                    
                    # 策略 1：全量扫描内存中所有匹配的 Measure 题目映射关系
                    for key in list(st.session_state.keys()):
                        if key.startswith("dc_measure_cols_"):
                            parts = key.split("_")
                            if len(parts) >= 4:
                                m_name = "_".join(parts[3:-1]) 
                                items_in_measure = st.session_state[key]
                                
                                if items_in_measure and isinstance(items_in_measure, list):
                                    # 验证这些题在当前这个子数据集中是否存在
                                    valid_cols = [c for c in items_in_measure if c in df_sub.columns]
                                    if len(valid_cols) >= 3:
                                        # 为了防止多个数据集里有同名 Measure 导致覆盖，
                                        # 组合成一个新的 Key 名字，例如: "数据集A - 心理资本"
                                        unique_task_name = f"{current_dataset_name} - {m_name}"
                                        saved_measures_found[unique_task_name] = df_sub[valid_cols].copy()

                # ==============================================================
                # 🚀 渲染第二步的多选确认框：展示所有被完美切片出来的任务队列
                # ==============================================================
                if saved_measures_found:
                    st.success(f"🎯 成功从选中的 **{len(selected_names)}** 个数据集中，精准识别到 **{len(saved_measures_found)}** 个分析维度！")
                    
                    selected_sub_measures = st.multiselect(
                        "2. 确认要批量运行的【数据集-Measure】组合（默认已全选）：",
                        options=list(saved_measures_found.keys()),
                        default=list(saved_measures_found.keys()), # 默认全选，一键多跑！
                        key="sub_batch_final_task_select"
                    )
                    
                    # 将最终确认的组合塞入底部的核心计算管道
                    for task_key in selected_sub_measures:
                        measures_to_process[task_key] = saved_measures_found[task_key]
                        
                else:
                    # 兜底：如果没有划分 Measure，直接把每个选中的数据集整张表作为一个大任务
                    st.info("💡 将每个子数据集作为独立问卷加入")
                    for current_dataset_name in selected_names:
                        df_sub = st.session_state.sub_datasets[current_dataset_name]
                        measures_to_process[f"{current_dataset_name} "] = df_sub.copy()

    else:
        uploaded_file = st.file_uploader("请上传用于分析的数据文件", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df_upload = pd.read_excel(uploaded_file)
                else:
                    df_upload = pd.read_csv(uploaded_csv)
                st.write("文件预览 (前5行):")
                st.dataframe(df_upload.head())
                
                st.info("👇 请勾选需要进行分析的【量表题目】")
                all_cols = df_upload.columns.tolist()
                numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
                default_cols = numeric_cols if numeric_cols else all_cols
                
                selected_cols = st.multiselect(
                    "请选择要分析的题目 (至少 3 个):",
                    options=all_cols,
                    default=default_cols
                )
                if len(selected_cols) >= 3:
                    measures_to_process["上传量表问卷"] = df_upload[selected_cols].copy()
            except Exception as e:
                st.error(f"读取文件失败: {e}")

    # ==========================================================================
    # 2. 数据清洗与预处理沙盒 (对每个独立的 Measure 容器做数值强转)
    # ==========================================================================
    cleaned_measures_dict = {}
    for m_name, df_raw in measures_to_process.items():
        # Altair/画图兼容性清洗
        df_raw.columns = [str(col).replace(":", "_") for col in df_raw.columns]
        
        numeric_cols = []
        for col in df_raw.columns:
            converted = pd.to_numeric(df_raw[col], errors='coerce')
            non_null_original = df_raw[col].notna().sum()
            if non_null_original > 0 and converted.notna().sum() / non_null_original >= 0.5:
                numeric_cols.append(col)
        
        df_num = df_raw[numeric_cols].apply(pd.to_numeric, errors='coerce').copy()
        df_num = df_num.dropna()
        df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna()
        df_num = df_num[~df_num.isin([np.inf, -np.inf]).any(axis=1)]
        
        if df_num.shape[0] >= 10 and df_num.shape[1] >= 3:
            cleaned_measures_dict[m_name] = df_num

    if not cleaned_measures_dict:
        st.warning("⏳ 队列中无有效数据集。请通过上方选项完成数据来源配置及题目导入。")
        return

    # ==========================================================================
    # 3. 全局统一参数配置面板
    # ==========================================================================
    st.markdown("---")
    st.subheader("⚙️ 批量分析因子数设置")
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        factor_method = st.radio(
            "因子数量确定方式 (通用):",
            ["👆 强制指定统一因子数","🤖 系统自动评估 (碎石图拐点)"],
            horizontal=True
        )
    with c_p2:
        manual_k_val = None
        if factor_method == "👆 强制指定统一因子数":
            manual_k_val = st.number_input("请输入期望提取的因子数量:", min_value=1, max_value=20, value=2, step=1)

    # 在 Session 缓存中建立批量分析沙盒
    if "batch_n1_results" not in st.session_state:
        st.session_state.batch_n1_results = {}

    st.markdown(f"**📋 待分析问卷任务队列 (共 {len(cleaned_measures_dict)} 个):**")
    for m_name, df_ready in cleaned_measures_dict.items():
        st.caption(f" └─ `维度名: {m_name}` ── 样本行数: `{df_ready.shape[0]}` | 初始题目数: `{df_ready.shape[1]}`")

    # ==========================================================================
    # 4. 执行批量全自动管道循环
    # ==========================================================================
    if st.button("🚀 开始运行所有选定 Measure 的自动化批量 EFA", type="primary", use_container_width=True):
        batch_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (m_name, df_ready) in enumerate(cleaned_measures_dict.items()):
            status_text.markdown(f"⏳ **正在计算 ({idx+1}/{len(cleaned_measures_dict)}):** 正在执行问卷维度 `[{m_name}]` 的迭代删题...")
            
            try:
                # 借助原有的主管道函数进行静默删题计算
                # 利用 st.container 捕获删题历史并在后面进行独立沙盒隔离展示
                with st.expander(f"⚙️ 查看 [{m_name}] 迭代实时删题细节 ", expanded=False):
                    final_df, final_loadings, kept, deleted, n_factors = run_pipeline_streamlit(
                        df_ready,
                        fixed_factors=manual_k_val,
                        whitelist=None
                    )
                
                batch_results[m_name] = {
                    "success": True,
                    "final_df": final_df,
                    "final_loadings": final_loadings,
                    "kept": kept,
                    "deleted": deleted,
                    "n_factors": n_factors,
                    "df_ready_columns": list(df_ready.columns)
                }
            except Exception as e:
                batch_results[m_name] = {
                    "success": False,
                    "error_msg": str(e)
                }
            
            progress_bar.progress((idx + 1) / len(cleaned_measures_dict))
            
        status_text.success("🎉 所有问卷的自动化批量分析与迭代删题全部完成！请在下方审阅结果。")
        st.session_state.batch_n1_results = batch_results

    # ==========================================================================
    # ==========================================================================
    # ==========================================================================
    # 5. 集中化多 Tab 面板呈现与用户单项微调、确认（每个维度独立表单导出 + 状态存储）
    # ==========================================================================
    if st.session_state.batch_n1_results:
        st.markdown("---")
        st.subheader("📥 批量结果确认")
        st.info("💡 切换下方的问卷标签页（Tabs），可以独立审查并单独下载每个维度对应的独立 Excel 报告。")

        active_tab_names = list(st.session_state.batch_n1_results.keys())
        tabs = st.tabs(active_tab_names)

        # 在外部初始化 N1_preEFA 核心全局缓存字典
        if "N1_preEFA" not in st.session_state:
            st.session_state.N1_preEFA = {}

        for i, m_name in enumerate(active_tab_names):
            res = st.session_state.batch_n1_results[m_name]
            
            with tabs[i]:
                st.markdown(f"### 📋 任务维度: {m_name}")
                if not res["success"]:
                    st.error(f"❌ 问卷分析由于数学边界或奇异矩阵崩溃，核心错误原因: {res['error_msg']}")
                    continue

                # 局部沙盒数据解包
                df_final = res["final_df"]
                loadings = res["final_loadings"]
                kept = res["kept"]
                deleted = res["deleted"]
                n_factors = res["n_factors"]

                # 5.1 题目变动报告单
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"✅ **最终保留题目数 ({len(kept)} 题):**")
                    st.caption(", ".join(kept))
                with c2:
                    if deleted:
                        st.warning(f"🛑 **迭代已删除题目 ({len(deleted)} 题):**")
                        st.caption(", ".join(deleted))
                    else:
                        st.info("💯 结构稳定：当前问卷无题目被剔除。")

                # 5.2 KMO 与 结构效度
                st.markdown("#### 1️⃣ KMO & Bartlett 球形检验")
                try:
                    kmo_all, kmo_model = calculate_kmo(df_final)
                    chi_square_value, p_value = calculate_bartlett_sphericity(df_final)
                    
                    summary_df = pd.DataFrame({
                        "统计指标检验名称": ["Kaiser-Meyer-Olkin (KMO)", "Bartlett 球形度卡方值", "Bartlett 显著性 (P-value)"],
                        "输出值": [f"{kmo_model:.4f}", f"{chi_square_value:.4f}", f"{p_value:.4e}"]
                    })
                    st.table(summary_df)
                except Exception as e:
                    st.error(f"效度指标计算受限: {e}")

                # 5.3 内部一致性信度
                st.markdown("#### 2️⃣ 信度检验 (Reliability Analysis)")
                try:
                    current_alpha = cronbach_alpha(df_final)
                    st.markdown(f"**👉 量表整体内部一致性 Cronbach's α = `{current_alpha:.4f}`**")
                    
                    with st.expander("查看删题后的信度变化 (Alpha if item deleted)"):
                        removal_df = alpha_after_removal(df_final)
                        removal_df_sorted = sort_dataframe_by_item_names(removal_df, item_column='删除的题项')
                        st.dataframe(removal_df_sorted.style.format({"Cronbach's α": "{:.4f}"}).background_gradient(cmap='RdYlGn', subset=["Cronbach's α"]))
                except Exception as e:
                    st.error(f"信度指标计算受限: {e}")

                # 5.4 最终对齐与着色的载荷矩阵
                st.markdown("#### 3️⃣ 最终因子载荷矩阵 (Factor Loadings)")
                loadings_sorted = sort_dataframe_by_item_names(loadings)

                def color_loadings(val):
                    try:
                        is_strong = abs(float(val)) > 0.40
                    except (TypeError, ValueError):
                        return ''
                    color = 'blue' if is_strong else 'black'
                    weight = 'bold' if is_strong else 'normal'
                    return f'color: {color}; font-weight: {weight}'

                st.dataframe(loadings_sorted.style.format("{:.3f}").map(color_loadings))

                # 5.5 单一量表独立恢复与微调
                if len(deleted) > 0:
                    st.markdown("#### 🛠️ 独立微调")
                    items_to_restore = st.multiselect(
                        f"从 [{m_name}] 的已删列表中选择【强制恢复】的题目:",
                        options=deleted,
                        key=f"restore_select_{m_name}"
                    )
                    if st.button(f"🔄 仅为 [{m_name}] 恢复题项并重算 EFA", key=f"btn_restore_{m_name}"):
                        if items_to_restore:
                            df_input_raw = cleaned_measures_dict[m_name]
                            with st.spinner("正在单独微调该维度..."):
                                f_df, f_load, k_list, d_list, f_k = run_pipeline_streamlit(
                                    df_input_raw,
                                    fixed_factors=n_factors,
                                    whitelist=items_to_restore
                                )
                                st.session_state.batch_n1_results[m_name] = {
                                    "success": True,
                                    "final_df": f_df,
                                    "final_loadings": f_load,
                                    "kept": k_list,
                                    "deleted": d_list,
                                    "n_factors": f_k,
                                    "df_ready_columns": list(df_ready.columns)
                                }
                                st.rerun()

                # ==============================================================
                # 🚨 【核心升级点 1】：解耦并拆分 复合 Key，将状态稳稳存入 N1_preEFA 缓存
                # ==============================================================
                st.markdown("#### 4️⃣ 独立导出与确认")
                
                # 智能识别当前复合键中包含的「数据集名称」和「Measure名称」
                # 格式预期: "子数据集A - 心理资本"，如无分隔符则兜底归类
                if " - " in m_name:
                    ds_name_extracted, real_measure_id = m_name.split(" - ", 1)
                else:
                    ds_name_extracted = "preEFA_SubDataset"
                    real_measure_id = m_name


                # 🌟【步骤 1】：先询问用户最终结果的 measure_id 是什么（允许用户交互修改）
                # 注意：此处的 final_measure_id 仅用于渲染表格第一列与 Excel 文件重命名，不污染底层 Key
                final_measure_id = st.text_input(
                    f"✍️ 请确认或修改量表【{real_measure_id}】最终用于【报告导出】的展示名称:",
                    value=real_measure_id,
                    key=f"input_measure_id_{m_name}"
                )

                # ==============================================================
                # 🚨 【核心修改】：根据展示名称和核心逻辑生成独立 Excel 数据并预览
                # ==============================================================
                try:
                    # 实时计算当前维度的多特征全量数据指标表
                    k_all, k_mod = calculate_kmo(df_final)
                    chi_v, p_v = calculate_bartlett_sphericity(df_final)
                    alpha_rem_df = alpha_after_removal(df_final)
                    alpha_lookup = alpha_rem_df.set_index("删除的题项")["Cronbach's α"]
                    itc_df = calculate_item_total_correlation(df_final)
                    communalities = (loadings ** 2).sum(axis=1)
                    _, _, s_stat, s_p = check_residual_normality(df_final, loadings)
                    
                    sorted_items = sort_item_cols_by_number(kept)
                    rows = []
                    
                    for item in sorted_items:
                        pre, num, text = parse_item_col(item)
                        item_txt = text or item
                        rev = 1 if (isinstance(item, str) and item.rstrip().endswith("r")) else 0
                        load_row = loadings.loc[item] if item in loadings.index else pd.Series(dtype=float)
                        
                        prefix = str(item_txt).strip().split("_", 1)[0]
                        m_match = re.search(r"(\d+)", prefix)
                        item_num = int(m_match.group(1)) if m_match else ""
                        
                        row = {
                            "measure_id": final_measure_id,   # 🌟【纯展示层】：仅在表格内呈现用户自定义的名称
                            "item_number": item_num,
                            "item_text": item_txt,
                            "reverse": rev,
                        }
                        for c in loadings.columns:
                            row[c] = load_row.get(c, np.nan)
                        row["KMO"] = k_mod
                        row["Bartlett_chi2"] = chi_v
                        row["Bartlett_p"] = p_v
                        row["alpha_after_removal"] = alpha_lookup.get(item, np.nan)
                        row["item_total_correlation"] = itc_df.loc[item, "Item-Total Corr"] if item in itc_df.index else np.nan
                        row["communality"] = communalities.get(item, np.nan)
                        row["residual_Shapiro_W_stat"] = s_stat
                        row["residual_Shapiro_W_p"] = s_p
                        rows.append(row)
                        
                    # 构造仅包含本维度数据的独立 DataFrame
                    single_measure_df = pd.DataFrame(rows)
                    
                    # 🌟【步骤 2】：在界面上呈现完整的表格供用户审查
                    st.caption("📋 当前待导出的全量数据指标预览（第一列展示改名后效果，不影响系统运行）：")
                    st.dataframe(single_measure_df, use_container_width=True)

                    # 🌟【步骤 3】：双重校验判断“对内原始 ID”此前是否已被保存，用于初始化复选框状态
                    is_previously_saved = (
                        ds_name_extracted in st.session_state.N1_preEFA and 
                        real_measure_id in st.session_state.N1_preEFA[ds_name_extracted]  # 🔒 对齐系统内部真实短 ID
                    )

                    # 🌟【步骤 4】：让用户点击复选框确认数据无误
                    is_confirmed = st.checkbox(
                        f"✅ 我已确认上述数据，将该量表缓存同步至后续的 CFA 模块", 
                        value=is_previously_saved,
                        key=f"confirm_check_{m_name}"
                    )

                    # ==============================================================
                    # 🚀 【后置联动】：用户勾选确认后，使用【原始 ID】进行对内闭环存储
                    # ==============================================================
                    if is_confirmed:
                        if ds_name_extracted not in st.session_state.N1_preEFA:
                            st.session_state.N1_preEFA[ds_name_extracted] = {}
                        
                        # 🔒【超级核心修改】：对内永远且必须使用 real_measure_id（最开始创建的ID）作为 Key
                        st.session_state.N1_preEFA[ds_name_extracted][real_measure_id] = {
                            "kept_items": list(kept),      # 过滤后确认保留的题目
                            "n_factors": int(n_factors),   # 模型推荐提取的因子数
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "clean_df": df_final          # 清洗删题后的真实 DataFrame
                        }
                        st.toast(f"🟢 【{real_measure_id}】")

                        # 2. 编译并输出 Excel 下载组件
                        single_buf = io.BytesIO()
                        with pd.ExcelWriter(single_buf, engine="xlsxwriter") as single_writer:
                            single_measure_df.to_excel(single_writer, sheet_name="EFA_Report", index=False)
                        single_buf.seek(0)
                        
                        today_str = date.today().strftime("%Y-%m-%d")
                        safe_measure_id = "".join(c for c in final_measure_id if c not in '[]:*?/\\ ')
                        file_filename = f"EFA_Report_{safe_measure_id}_{today_str}.xlsx"  # 文件名享受个性化长改名
                        
                        st.download_button(
                            label=f"⬇️ 立即下载 【{final_measure_id}】 维度的独立 Excel 报告",
                            data=single_buf.getvalue(),
                            file_name=file_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_btn_single_{m_name}"
                        )
                    else:
                        # 用户未勾选或取消勾选时，动态移除该【内部真实 ID】的缓存
                        if ds_name_extracted in st.session_state.N1_preEFA:
                            if real_measure_id in st.session_state.N1_preEFA[ds_name_extracted]:
                                del st.session_state.N1_preEFA[ds_name_extracted][real_measure_id]
                        st.info("💡 请检查上方预览表，确认无误后勾选上方“✅ 我已确认...”复选框以激活下载。")

                except Exception as ex_build:
                    st.caption(f"⚠️ 该维度的 Excel 独立导出表编译受限: {ex_build}")

        # ==========================================================================
        # 6. 页面最底部全局大看板：实时监测并预览 N1_preEFA 配置资产状态（展示原生干净结构）
        # ==========================================================================
        if st.session_state.N1_preEFA:
            st.markdown("---")
            with st.expander("🚀 查看当前准备对接 Batch CFA 的 `N1_preEFA` 清单", expanded=True):
                
                for d_key, m_dict in st.session_state.N1_preEFA.items():
                    st.markdown(f"#### 📦 数据集容器: `{d_key}`")
                    for sub_m, config in m_dict.items():
                        st.markdown(
                            f" * 🟢:**`{sub_m}`** ── 保留题目: `{len(config['kept_items'])}` 题 "
                        )

# CFA 核心算法区域
# ==============================================================================

def _is_reverse_coded(item_name):
    """判断题目是否为反向题：仅当题目文本去尾部空白/标点后以 r 结尾。"""
    if not isinstance(item_name, str):
        return False
    _, _, text = parse_item_col(item_name)
    s = (text or item_name).strip()
    s = s.replace("ｒ", "r").replace("Ｒ", "R")
    s = re.sub(r"""[\s\u3000\)\]）】》〉'"“”’`~!@#$%^&*+=|\\/:;,.?，。！？、；：-]+$""", "", s)
    return s.lower().endswith("r")


def _extract_item_num_and_text(col_name: str) -> Tuple[Any, str]:
    pre, num, text = parse_item_col(str(col_name))
    item_num = num
    if item_num is None:
        m = re.search(r"(\d+)", str(col_name).split("_", 1)[0])
        item_num = int(m.group(1)) if m else np.nan
    return item_num, (text or str(col_name))


def _reset_smart_multiselect_cache(key_suffix: str) -> None:
    """清理 smart_multiselect 的 checkbox/选择缓存，强制按最新默认值重建。"""
    cb_prefix = f"cb_{key_suffix}_"
    for k in list(st.session_state.keys()):
        if k.startswith(cb_prefix):
            del st.session_state[k]
    for k in (f"{key_suffix}_last_selected", f"{key_suffix}_control_action"):
        st.session_state.pop(k, None)

def calculate_advanced_stats(model, fit_stats, n_samples, df):
    """
    补充计算高级指标 (SABIC) 和 semopy 缺失的指标 (SRMR)
    """
    stats_dict = fit_stats.iloc[0].to_dict()
    
    # 1. SABIC (Sample-size Adjusted BIC)
    try:
        aic = stats_dict.get('AIC', 0)
        logl = stats_dict.get('LogL', stats_dict.get('logl', 0))
        k = (aic + 2 * logl) / 2
        sabic = -2 * logl + k * np.log((n_samples + 2) / 24)
        stats_dict['SABIC'] = sabic
    except:
        stats_dict['SABIC'] = np.nan

    # 2. SRMR (Standardized Root Mean Square Residual) - 手动计算
    # semopy 无 predict_cov，用 calc_sigma() 得到模型隐含协方差；若已有则直接使用
    if 'SRMR' not in stats_dict or (isinstance(stats_dict.get('SRMR'), float) and np.isnan(stats_dict.get('SRMR'))):
        if 'srmr' in stats_dict and isinstance(stats_dict['srmr'], (int, float)):
            stats_dict['SRMR'] = stats_dict['srmr']
        else:
            try:
                obs_order = model.vars['observed']
                df_obs = df[obs_order]
                obs_corr = df_obs.corr().values
                sigma_tuple = model.calc_sigma()
                implied_cov = np.asarray(sigma_tuple[0], dtype=float)
                if implied_cov.shape != obs_corr.shape:
                    raise ValueError("calc_sigma 与观测变量维度不一致")
                d = np.diag(np.sqrt(np.maximum(np.diag(implied_cov), 1e-12)))
                d_inv = np.linalg.inv(d)
                implied_corr = d_inv @ implied_cov @ d_inv
                residuals = obs_corr - implied_corr
                idx = np.tril_indices_from(residuals)
                res_values = residuals[idx]
                stats_dict['SRMR'] = float(np.sqrt(np.mean(res_values**2)))
            except Exception:
                stats_dict['SRMR'] = np.nan

    # 2. RMSEA Confidence Interval & P-value
    # semopy 默认不输出 RMSEA 的 CI 和 P-value，这里尝试手动近似计算
    # 或者直接使用 semopy 的默认输出（如果版本支持）
    # 为了简化，我们暂时只依赖 semopy 提供的基础指标，如果需要精确的 P-close，需要手动实现非中心卡方分布计算
    # 这里我们先保留 semopy 原生值，后续版本可扩展 scipy.stats.ncx2 计算逻辑
    
    return stats_dict

def run_cfa_gui(df, factor_name, factor_items, method_name, method_items):
    """
    根据用户GUI选择运行 CFA
    """
    # 数据质量检查
    df_clean = df.copy()

    # 移除包含NaN或无穷大值的行
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]

    if df_clean.empty:
        return None, "数据清理后没有有效样本，无法进行CFA分析", None

    if len(df_clean) < len(df):
        st.warning(f"⚠️ 已自动移除 {len(df) - len(df_clean)} 行含有无效值的样本")

    # 1. 自动构建模型语法
    # 严格采用 marker-variable 标定：
    # - 主因子第一题载荷固定为 1（1*item1）
    # - 主因子方差自由估计（不对 factor_name ~~ factor_name 施加固定值）
    ordered_factor_items = sort_item_cols_by_number(list(factor_items))
    if not ordered_factor_items:
        return None, "主因子题目为空，无法构建 CFA 模型。", None
    marker_item = ordered_factor_items[0]
    other_items = ordered_factor_items[1:]
    if other_items:
        model_desc = f"{factor_name} =~ 1*{marker_item} + {' + '.join(other_items)}\n"
    else:
        model_desc = f"{factor_name} =~ 1*{marker_item}\n"
    
    if method_items:
        ordered_method_items = sort_item_cols_by_number(list(method_items))
        model_desc += f"{method_name} =~ {' + '.join(ordered_method_items)}\n"
        # 方法因子方差固定为 1
        model_desc += f"{method_name} ~~ 1*{method_name}\n"
    
    # 2. 初始化与拟合
    import semopy
    from semopy import Model
    try:
        model = Model(model_desc)
        model.fit(df_clean)
    except Exception as e:
        return None, f"模型拟合失败: {str(e)}", None

    # 3. 获取统计结果
    try:
        # 获取参数估计 (包含非标准化和标准化)
        estimates = model.inspect(std_est=True)
        
        # semopy 的 inspect 默认没有 Std.lv 列，我们需要手动调整
        # 在 R lavaan 中:
        # Estimate = 非标准化
        # Std.lv = 潜变量标准化 (Latent variable variance = 1)
        # Std.all = 完全标准化 (Latent + Observed variance = 1)
        
        # semopy 不同版本中标准化载荷列名可能不同：'Est. Std'、'est.std'、'Std.Est'、'std.all' 等
        # 自动检测并重命名标准化载荷列
        possible_std_cols = ['Est. Std', 'est.std', 'Std.Est', 'std.all', 'Std.Estimate', 'est_std', 'standardized']
        std_col_found = None
        for col in possible_std_cols:
            if col in estimates.columns:
                std_col_found = col
                break
        
        rename_map = {
            'lval': 'LHS', 'op': 'op', 'rval': 'RHS',
            'Estimate': 'Estimate', 'Std. Err': 'Std.Err', 
            'z-value': 'z-value', 'p-value': 'P(>|z|)'
        }
        if std_col_found:
            rename_map[std_col_found] = 'Std.all'
        
        estimates = estimates.rename(columns=rename_map)
        
        # 获取拟合指数
        fit_stats = semopy.calc_stats(model)
        
        # 补充计算高级指标
        n = len(df_clean)
        if not fit_stats.empty:
            advanced_stats = calculate_advanced_stats(model, fit_stats, n, df_clean)
        else:
            advanced_stats = {}
            
    except Exception as e:
        return None, f"计算统计量失败: {str(e)}", None
        
    return (model, estimates, advanced_stats), None, model_desc






# ==============================================================================
# 🧪 2. 自动删题 CFA 版



def render_stage2_cfa_clean():
    # ==========================================================================
    # 1. 读取上游 EFA 资产（N1_preEFA）
    # ==========================================================================
    n1_asset = st.session_state.get("N1_preEFA")
    if not n1_asset:
        st.info("💡 暂未检测到上游 EFA（N1_preEFA）留存的题目结果。请确保在前置 EFA 模块中完成了分析、勾选了确认并保存。")
        return

    all_upstream_measures = {}
    if isinstance(n1_asset, dict):
        for ds_key, measure_dict in n1_asset.items():
            if isinstance(measure_dict, dict):
                for m_id, m_config in measure_dict.items():
                    if isinstance(m_config, dict) and "kept_items" in m_config:
                        raw_df_entity = m_config.get("clean_df")
                        if raw_df_entity is None:
                            if "sub_datasets" in st.session_state and ds_key in st.session_state.sub_datasets:
                                raw_df_entity = st.session_state.sub_datasets[ds_key]
                            elif "dc_dataset_full" in st.session_state:
                                raw_df_entity = st.session_state.dc_dataset_full
                            else:
                                raw_df_entity = st.session_state.get("df_source")
                        all_upstream_measures[str(m_id)] = {
                            "items": m_config["kept_items"],
                            "clean_df": raw_df_entity,
                            "measure_id_raw": m_id,
                            "ds_key": ds_key
                        }

    if not all_upstream_measures:
        st.error("❌ 无法从上游 N1_preEFA 资产中提取到任何有效的量表数据。")
        return

    # ==========================================================================
    # 2. 选择要处理的量表 & 全局参数
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🔍 第一步：勾选您本次需要分析的量表")
    selected_measure_ids = st.multiselect(
        "📂 请选择要拉入 CFA 自动优化删题流的量表（默认全选）：",
        options=list(all_upstream_measures.keys()),
        default=list(all_upstream_measures.keys())
    )
    if not selected_measure_ids:
        st.warning("⚠️ 请至少勾选一个量表以继续分析。")
        return

    min_items_limit = st.number_input(
        "🛑 最小保留题目底线",
        min_value=3, max_value=30, value=8, step=1,
        help="当维度内题目数减少到该值时，算法强制停止删题。"
    )

    # ==========================================================================
    # 3. 配置因子结构（每个量表一个标签页）
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🛠️ 第二步：CFA 测量模型结构核对与锁定")
    st.caption("请依次进入每个量表的标签页，核对或调整其因子结构配置。")

    cfa_ready_queue = {}
    tabs = st.tabs([f" {m_id}" for m_id in selected_measure_ids])

    for idx, sub_name in enumerate(selected_measure_ids):
        with tabs[idx]:
            asset = all_upstream_measures[sub_name]
            raw_items = asset["items"]
            raw_df = asset["clean_df"]

            # ---- 清洗列名 ----
            orig_to_clean = {}
            clean_cols = []
            for col in raw_df.columns:
                clean_col = re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(col))
                if not clean_col:
                    clean_col = str(col)
                orig_to_clean[col] = clean_col
                clean_cols.append(clean_col)
            df_clean = raw_df.copy()
            df_clean.columns = clean_cols
            clean_to_orig = {v: k for k, v in orig_to_clean.items()}

            clean_items = [orig_to_clean.get(item, item) for item in raw_items if item in orig_to_clean]
            clean_items = [c for c in clean_items if c in df_clean.columns]

            st.markdown(f"####  【{sub_name}】")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🅰️ 主因子 (Trait Factor)")
                factor_name = st.text_input(
                    "主因子名称 (英文/标识符):",
                    value=f"{sub_name}",
                    key=f"cfa_fname_inp_{sub_name}"
                )
                factor_items_raw = smart_multiselect(
                    options=raw_items,
                    label=f"选择属于 {factor_name} 的题目",
                    key_suffix=f"cfa_factor_{sub_name}",
                    default_selected=raw_items,
                    show_selection_controls=True,
                )
                factor_items_clean = [orig_to_clean.get(item, item) for item in factor_items_raw if item in orig_to_clean]
                factor_items_clean = [c for c in factor_items_clean if c in df_clean.columns]

            with col2:
                st.markdown("##### 🅱️ 方法因子 (Method Factor)")
                method_name = st.text_input(
                    "方法因子名称 (英文/标识符):",
                    value="Method",
                    key=f"cfa_mname_inp_{sub_name}"
                )
                method_options_raw = factor_items_raw
                default_method_raw = [item for item in method_options_raw if _is_reverse_coded(item)]
                method_key_suffix = f"cfa_method_{sub_name}"
                method_sig_key = f"{method_key_suffix}_options_sig"
                if st.session_state.get(method_sig_key) != tuple(method_options_raw):
                    _reset_smart_multiselect_cache(method_key_suffix)
                    st.session_state[method_sig_key] = tuple(method_options_raw)

                st.caption("💡 提示：默认已自动预选末尾为 'r' 的反向题至方法因子。")
                def _on_reset_method_tab(k_suffix=method_key_suffix, sig_val=tuple(method_options_raw), def_items=default_method_raw):
                    _reset_smart_multiselect_cache(k_suffix)
                    st.session_state[f"{k_suffix}_options_sig"] = sig_val
                    st.session_state[f"{k_suffix}_last_selected"] = def_items
                st.button('🔄 重新预选方法因子题目', key=f"tab_btn_reset_method_{sub_name}", on_click=_on_reset_method_tab)

                method_items_raw = smart_multiselect(
                    options=method_options_raw,
                    label=f"选择受 {method_name} 影响的题目(选空则不启用)",
                    key_suffix=method_key_suffix,
                    default_selected=default_method_raw,
                    show_selection_controls=True,
                )
                method_items_clean = [orig_to_clean.get(item, item) for item in method_items_raw if item in orig_to_clean]
                method_items_clean = [c for c in method_items_clean if c in df_clean.columns]

            cfa_ready_queue[sub_name] = {
                "df_clean": df_clean,
                "clean_to_orig": clean_to_orig,
                "orig_to_clean": orig_to_clean,
                "factor_name": factor_name,
                "method_name": method_name if method_items_clean else None,
                "factor_items": factor_items_clean,
                "method_items": method_items_clean,
                "raw_items": raw_items,
                "measure_id_raw": sub_name,
                "raw_df": raw_df,  # 保存原始未清洗的数据框，用于最终锁定
            }

    # ==========================================================================
    # 4. 锁定配置并批量运行
    # ==========================================================================
    st.markdown("---")
    st.info("📌 **核对完毕后，点击下方「锁定配置并批量运行」按钮，将对所有已配置的量表执行自动删题 CFA。**")
    if st.button("🔒 锁定配置并批量运行所有量表", type="primary", use_container_width=True):
        st.session_state["cfa_locked_config"] = cfa_ready_queue
        st.toast("✅ 配置已锁定，开始逐个运行量表...", icon="🚀")

        for sub_name, cfg in cfa_ready_queue.items():
            # 清除旧结果
            for key in list(st.session_state.keys()):
                if key.startswith(f"n2_{sub_name}_"):
                    del st.session_state[key]
            _run_auto_cfa_for_measure(
                sub_name=sub_name,
                df_clean=cfg["df_clean"],
                clean_to_orig=cfg["clean_to_orig"],
                factor_name=cfg["factor_name"],
                method_name=cfg["method_name"],
                factor_items=cfg["factor_items"],
                method_items=cfg["method_items"],
                min_items_limit=min_items_limit,
            )
        st.success("🎉 所有量表自动删题分析完成！请在下方各个量表标签页中查看结果并确认锁定。")

    # ==========================================================================
    # 5. 展示结果 & 确认锁定（每个量表独立）
    # ==========================================================================
    st.markdown("---")
    st.subheader("📊 各量表分析报告与确认")

    for sub_name, cfg in cfa_ready_queue.items():
        success_key = f"n2_{sub_name}_success"
        if not st.session_state.get(success_key, False):
            with st.expander(f"⏳ {sub_name} - 等待运行", expanded=False):
                st.info("该量表尚未运行，请点击上方「锁定配置并批量运行」按钮。")
            continue

        with st.expander(f"📈 {sub_name} - 分析结果", expanded=True):
            trace_logs = st.session_state.get(f"n2_{sub_name}_trace_logs", [])
            final_fit = st.session_state.get(f"n2_{sub_name}_fit_stats", {})
            final_estimates = st.session_state.get(f"n2_{sub_name}_estimates", pd.DataFrame())
            final_syntax = st.session_state.get(f"n2_{sub_name}_syntax", "")
            final_factor_items = st.session_state.get(f"n2_{sub_name}_factor_items", [])
            final_method_items = st.session_state.get(f"n2_{sub_name}_method_items", [])
            final_df_cfa = st.session_state.get(f"n2_{sub_name}_df_cfa", pd.DataFrame())
            fname = cfg["factor_name"]
            mname = cfg["method_name"]

            # ---- 删题记录 ----
            st.markdown("##### 📝 自动删题记录")
            if trace_logs:
                log_df = pd.DataFrame(trace_logs)
                st.table(log_df)
            else:
                st.write("无删题记录（模型首次即达标）。")

            # ---- 拟合指标 ----
            st.markdown("##### 🏆 关键拟合指标")
            def _get_val(key):
                if isinstance(final_fit, dict):
                    return final_fit.get(key, np.nan)
                elif isinstance(final_fit, pd.DataFrame):
                    for col in final_fit.columns:
                        if final_fit[col].dtype == object:
                            rows = final_fit[final_fit[col].astype(str).str.upper() == key.upper()]
                            if not rows.empty:
                                val_cols = [c for c in final_fit.columns if c != col]
                                try:
                                    return float(rows[val_cols[0]].values[0])
                                except:
                                    pass
                return np.nan

            metrics = {
                "CFI": _get_val("CFI"),
                "TLI": _get_val("TLI"),
                "RMSEA": _get_val("RMSEA"),
                "SRMR": _get_val("SRMR"),
                "Chi-Square": _get_val("chi2"),
                "AIC": _get_val("AIC"),
                "BIC": _get_val("BIC"),
                "SABIC": _get_val("SABIC"),
            }
            cols1 = st.columns(4)
            for i, k in enumerate(["CFI", "TLI", "RMSEA", "SRMR"]):
                val = metrics[k]
                cols1[i].metric(label=k, value=f"{val:.3f}" if not np.isnan(val) else "N/A")
            cols2 = st.columns(4)
            for i, k in enumerate(["Chi-Square", "AIC", "BIC", "SABIC"]):
                val = metrics[k]
                cols2[i].metric(label=k, value=f"{val:.3f}" if not np.isnan(val) else "N/A")

            # ---- 载荷表 ----
            st.markdown("##### 📋 最终因子载荷")
            if not final_estimates.empty:
                est_display = final_estimates.copy()
                clean_to_orig = cfg["clean_to_orig"]
                for col in ['LHS', 'RHS']:
                    if col in est_display.columns:
                        est_display[col] = est_display[col].apply(lambda x: clean_to_orig.get(x, x))
             

                # 排序
                def _sort_rank(row):
                    lhs, op, rhs = row['LHS'], row['op'], row['RHS']
                    if op == '=~' and lhs == fname: return 1
                    if op == '=~' and lhs == mname: return 2
                    if op == '~~' and lhs == rhs and lhs == fname: return 3
                    if op == '~~' and lhs == rhs and lhs == mname: return 4
                    if op == '~~' and lhs == rhs and lhs not in [fname, mname]: return 5
                    return 6
                est_display['rank'] = est_display.apply(_sort_rank, axis=1)
                est_display = est_display.sort_values('rank').drop(columns=['rank'])
                display_cols = ['LHS', 'op', 'RHS', 'Estimate', 'Std.Err', 'z-value', 'P(>|z|)', 'Std.all']
                display_cols = [c for c in display_cols if c in est_display.columns]
                # ✅ 修复：只对数值列应用格式化
                numeric_cols = est_display[display_cols].select_dtypes(include=[np.number]).columns
                st.dataframe(est_display[display_cols].style.format("{:.3f}", subset=numeric_cols))
            else:
                st.warning("无载荷表输出。")

            # ---- 模型语法 ----
            with st.expander("查看模型语法"):
                st.code(final_syntax, language="text")

            # ---- 确认锁定 ----
            st.markdown("---")
            st.markdown("##### 🔒 确认锁定结果至 N2_preCFA（供后续 Final EFA 使用）")
            measure_id_input = st.text_input(
                "量表 measure_id（唯一编码）",
                value=(st.session_state.get(f"n2_{sub_name}_measure_id") or sub_name),
                key=f"n2_{sub_name}_measure_id_input",
                placeholder="如 LQ、EQ",
                help="此 ID 将作为 N2_preCFA 中的键，并用于报告文件名。"
            )
            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("✅ 确认并锁定此量表", key=f"n2_{sub_name}_lock_btn", use_container_width=True):
                    mid = measure_id_input.strip()
                    if not mid:
                        st.warning("请填写 measure_id。")
                    else:
                        if final_df_cfa.empty or not final_factor_items:
                            st.error("❌ 该量表尚未成功运行，请先运行分析。")
                        else:
                            if "N2_preCFA" not in st.session_state:
                                st.session_state["N2_preCFA"] = {}
                            clean_to_orig = cfg["clean_to_orig"]
                            orig_factor_items = [clean_to_orig.get(c, c) for c in final_factor_items]
                            # 使用原始数据框（未清洗）提取最终保留的列
                            raw_df_orig = cfg.get("raw_df", all_upstream_measures[sub_name]["clean_df"])
                            orig_cols_present = [c for c in orig_factor_items if c in raw_df_orig.columns]
                            final_raw_df = raw_df_orig[orig_cols_present].copy()
                            st.session_state["N2_preCFA"][mid] = {
                                "measure_id": mid,
                                "origin_sub_name": sub_name,
                                "clean_df": final_raw_df,
                                "kept_items": orig_factor_items,
                                "factor_name": fname,
                                "method_name": mname,
                                "fit_stats": final_fit,
                                "estimates": final_estimates,
                            }
                            st.success(f"✅ 量表 {mid} 已锁定至 N2_preCFA！")
                            st.session_state[f"n2_{sub_name}_measure_id"] = mid

            # ---- 下载报告 ----
            if st.session_state.get(f"n2_{sub_name}_measure_id"):
                if st.button("📥 下载此量表 Excel 报告", key=f"n2_{sub_name}_dl_report"):
                    _generate_and_download_report(
                        sub_name=sub_name,
                        cfg=cfg,
                        final_df_cfa=final_df_cfa,
                        final_factor_items=final_factor_items,
                        final_estimates=final_estimates,
                        final_fit=final_fit,
                        fname=fname,
                        measure_id=st.session_state[f"n2_{sub_name}_measure_id"],
                    )


# =============================================================================
# 辅助函数（内部使用）
# =============================================================================

def _run_auto_cfa_for_measure(sub_name, df_clean, clean_to_orig, factor_name, method_name,
                              factor_items, method_items, min_items_limit):
    """单个量表的自动删题流程"""
    active_factor = list(factor_items)
    active_method = list(method_items) if method_items else []
    all_active = list(dict.fromkeys(active_factor + active_method))
    df_sub = df_clean[all_active].dropna(axis=0, how='any').copy()
    if len(df_sub) < 10:
        st.session_state[f"n2_{sub_name}_success"] = False
        st.session_state[f"n2_{sub_name}_error"] = "样本量不足（<10）"
        return

    trace_logs = []
    current_step = 0
    max_steps = 20
    best_score = -1.0
    best_payload = None

    while current_step < max_steps:
        current_count = len(active_factor)
        if current_count < min_items_limit:
            st.session_state[f"n2_{sub_name}_warning"] = f"达到最小题目限制 ({min_items_limit})，停止删题。"
            break

        result, err, syntax = run_cfa_gui(
            df_sub, factor_name, active_factor, method_name, active_method
        )
        if err:
            trace_logs.append({"round": current_step+1, "items": current_count,
                               "cfi": np.nan, "tli": np.nan, "action": f"拟合出错: {err}", "deleted": "无"})
            break

        model_obj, estimates, fit_stats = result
        cfi = _extract_fit(fit_stats, "CFI")
        tli = _extract_fit(fit_stats, "TLI")
        score = cfi + tli
        if score > best_score:
            best_score = score
            best_payload = {
                "result": result,
                "syntax": syntax,
                "cfi": cfi,
                "tli": tli,
                "factor_items": list(active_factor),
                "method_items": list(active_method),
                "estimates": estimates,
                "fit_stats": fit_stats,
            }

        if cfi >= 0.90 and tli >= 0.90:
            trace_logs.append({"round": current_step+1, "items": current_count,
                               "cfi": cfi, "tli": tli, "action": "✨ 达标！", "deleted": "无"})
            _save_cfa_result(sub_name, best_payload, trace_logs, df_sub, active_factor, active_method)
            return

        # 穷举试删
        best_del_score = -1.0
        worst_item = None
        for test_item in active_factor:
            trial_factor = [i for i in active_factor if i != test_item]
            trial_method = [i for i in active_method if i != test_item]
            trial_all = list(dict.fromkeys(trial_factor + trial_method))
            trial_df = df_clean[trial_all].dropna(axis=0, how='any')
            if len(trial_df) < 10:
                continue
            trial_res, trial_err, _ = run_cfa_gui(
                trial_df, factor_name, trial_factor, method_name, trial_method
            )
            if not trial_err:
                _, _, trial_fit = trial_res
                t_cfi = _extract_fit(trial_fit, "CFI")
                t_tli = _extract_fit(trial_fit, "TLI")
                trial_score = t_cfi + t_tli
                if trial_score > best_del_score:
                    best_del_score = trial_score
                    worst_item = test_item

        if worst_item is None:
            trace_logs.append({"round": current_step+1, "items": current_count,
                               "cfi": cfi, "tli": tli, "action": "⚠️ 无法找到可删除的题目", "deleted": "无"})
            break

        active_factor.remove(worst_item)
        if worst_item in active_method:
            active_method.remove(worst_item)
        trace_logs.append({"round": current_step+1, "items": current_count,
                           "cfi": cfi, "tli": tli, "action": f"❌ 删除题目", "deleted": worst_item})
        current_step += 1

    # 循环结束，使用最佳历史
    if best_payload:
        _save_cfa_result(sub_name, best_payload, trace_logs, df_sub,
                         best_payload["factor_items"], best_payload["method_items"])
    else:
        st.session_state[f"n2_{sub_name}_success"] = False
        st.session_state[f"n2_{sub_name}_error"] = "未获得有效模型。"


def _save_cfa_result(sub_name, payload, trace_logs, df_sub, factor_items, method_items):
    """保存单个量表的CFA结果到session_state"""
    st.session_state[f"n2_{sub_name}_success"] = True
    st.session_state[f"n2_{sub_name}_estimates"] = payload["estimates"]
    st.session_state[f"n2_{sub_name}_fit_stats"] = payload["fit_stats"]
    st.session_state[f"n2_{sub_name}_syntax"] = payload["syntax"]
    st.session_state[f"n2_{sub_name}_trace_logs"] = trace_logs
    st.session_state[f"n2_{sub_name}_factor_items"] = factor_items
    st.session_state[f"n2_{sub_name}_method_items"] = method_items
    st.session_state[f"n2_{sub_name}_df_cfa"] = df_sub[list(dict.fromkeys(factor_items + method_items))].dropna(axis=0, how='any')


def _extract_fit(fit_stats, key):
    """从fit_stats中提取指标值（兼容dict和DataFrame）"""
    if isinstance(fit_stats, dict):
        return fit_stats.get(key, np.nan)
    elif isinstance(fit_stats, pd.DataFrame):
        for col in fit_stats.columns:
            if fit_stats[col].dtype == object:
                rows = fit_stats[fit_stats[col].astype(str).str.upper() == key.upper()]
                if not rows.empty:
                    val_cols = [c for c in fit_stats.columns if c != col]
                    try:
                        return float(rows[val_cols[0]].values[0])
                    except:
                        pass
    return np.nan


def _generate_and_download_report(sub_name, cfg, final_df_cfa, final_factor_items, final_estimates, final_fit, fname, measure_id):
    """生成Excel报告并触发下载"""
    try:
        df_cfa = final_df_cfa.copy()
        factor_items = final_factor_items
        estimates = final_estimates
        stats_dict = final_fit
        clean_to_orig = cfg["clean_to_orig"]

        def _to_num(x):
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

        # 提取潜变量方差
        trait_var = np.nan
        for _, row in estimates.iterrows():
            if row.get("op") == "~~" and row.get("LHS") == fname and row.get("RHS") == fname:
                trait_var = row.get("Estimate", np.nan)
                break

        # 提取载荷（清洗名作为键）
        loadings_unstd = {}
        loadings_std = {}
        if "LHS" in estimates.columns and "op" in estimates.columns and "RHS" in estimates.columns:
            # 去除空格
            estimates['op_strip'] = estimates['op'].astype(str).str.strip()
            estimates['LHS_strip'] = estimates['LHS'].astype(str).str.strip()
            estimates['RHS_strip'] = estimates['RHS'].astype(str).str.strip()
            fname_clean = str(fname).strip()
            # 查找主因子载荷
            trait_loadings = estimates[(estimates['op_strip'] == "=~") & (estimates['LHS_strip'] == fname_clean)]
            if trait_loadings.empty:
                # 尝试忽略大小写
                trait_loadings = estimates[
                    (estimates['op_strip'] == "=~") & 
                    (estimates['LHS_strip'].str.lower() == fname_clean.lower())
                ]
            if trait_loadings.empty:
                # 可能 op 是 '~'（回归形式）
                trait_loadings = estimates[(estimates['op_strip'] == "~") & (estimates['RHS_strip'] == fname_clean)]
            if trait_loadings.empty:
                trait_loadings = estimates[
                    (estimates['op_strip'] == "~") & 
                    (estimates['RHS_strip'].str.lower() == fname_clean.lower())
                ]
            # 提取载荷
            for _, row in trait_loadings.iterrows():
                item_key = row['RHS_strip']  # 清洗名
                loadings_unstd[item_key] = _to_num(row.get('Estimate', np.nan))
                loadings_std[item_key] = _to_num(row.get('Std.all', np.nan))
        
        
        


        def _get_any(d, keys, default=np.nan):
            for k in keys:
                if k in d:
                    v = _to_num(d.get(k))
                    if not np.isnan(v):
                        return v
            return default

        chi2_val = _get_any(stats_dict, ["chi2", "Chi2"])
        dof_val = _get_any(stats_dict, ["DoF", "dof", "df"])
        p_val = _get_any(stats_dict, ["chi2 p-value", "p-value", "pvalue", "p_value"])
        alpha_val = cronbach_alpha(df_cfa) if not df_cfa.empty else np.nan

        # 构建题目明细表（显示原始名）
        sorted_items = sort_item_cols_by_number(factor_items)
        rows = []
        for idx, item_clean in enumerate(sorted_items, start=1):
            
            item_key = item_clean.strip()   # 关键：去除空格
            # 后续使用 item_key 去 loadings_unstd 和 loadings_std 中取值
            unstd = loadings_unstd.get(item_key, np.nan)
            std = loadings_std.get(item_key, np.nan)
            
            item_raw = clean_to_orig.get(item_clean, item_clean)
            _, num, text = parse_item_col(item_raw)
            rev = 1 if _is_reverse_coded(item_raw) else 0
            item_number = num if num is not None else idx


            # =============================================================================
            # 🚀 优化后的载荷与方差动态提取逻辑
            # =============================================================================
            
            # 1. 提取潜在变量方差 (variance_latent)
            # 条件：LHS == sub_name 且 RHS == sub_name
            # 注：有时候 semopy 内部名字也是清洗过的，我们同时用原始名和清洗名做兼容匹配
            sub_name_clean = cfg.get("sub_name_clean", sub_name) 
            
            latent_var_row = final_estimates[
                ((final_estimates['LHS'] == sub_name) | (final_estimates['LHS'] == sub_name_clean)) & 
                ((final_estimates['RHS'] == sub_name) | (final_estimates['RHS'] == sub_name_clean))
            ]
            
            if not latent_var_row.empty:
                # 优先取 'Estimate' 列作为方差
                trait_var = latent_var_row.iloc[0].get('Estimate', np.nan)
            else:
                trait_var = np.nan
            
            # 2. 准备针对题目的前缀数字提取函数（处理类似 1_xxx, 01_xxx 的情况）
            # =============================================================================


            
            # 1. 统一前缀数字提取函数（例如 1_xxx 或 01_xxx 都能提取出 1）
            def get_prefix_num(item_str):
                if not isinstance(item_str, str):
                    return None
                match = re.match(r'^.*?(\d+)', item_str)
                return int(match.group(1)) if match else None
            
            # 获取当前循环中这道题目的前缀数字
            current_item_num = get_prefix_num(item_raw)
            
            # 2. 清洗 estimates 数据，去掉任何可能干扰的隐藏空格
            estimates_clean = final_estimates.copy()
            for col in ['LHS', 'op', 'RHS']:
                if col in estimates_clean.columns:
                    estimates_clean[col] = estimates_clean[col].astype(str).str.strip()
            
            # 3. 🎯 严格提取潜变量方差 (variance_latent)
            # 条件：op 是 '~~'，LHS == RHS，且排除题目（有数字前缀）和 Method 因子
            trait_var = np.nan
            mname_clean = str(mname).strip().lower() if mname else ""
            
            for _, row in estimates_clean[estimates_clean['op'] == "~~"].iterrows():
                l_val = row['LHS']
                r_val = row['RHS']
                
                # 条件甲：LHS 和 RHS 必须完全相同
                if l_val == r_val:
                    # 条件乙：不能是题目（即不能带有数字前缀）
                    if get_prefix_num(l_val) is None:
                        # 条件丙：不能是 Method 效应因子
                        if mname_clean and (mname_clean in l_val.lower()):
                            continue # 跳过 Method 因子行
                            
                        # 此时剩下的就是真正的主成分潜变量方差了！
                        trait_var = _to_num(row.get('Estimate', np.nan))
                        break
            
            # 4. 提取当前题目的非标准化与标准化载荷
            unstd_load = np.nan
            std_load = np.nan
            
            if current_item_num is not None:
                # 筛选出属于载荷的操作符行（=~ 或 ~）
                loading_rows = estimates_clean[estimates_clean['op'].isin(["=~", "~"])]
                
                for _, row in loading_rows.iterrows():
                    lhs_num = get_prefix_num(row['LHS'])
                    rhs_num = get_prefix_num(row['RHS'])
                    
                    # 核心逻辑：只要这一行的 LHS 或者 RHS 的前缀数字等同于当前题目的数字，即命中！
                    if lhs_num == current_item_num or rhs_num == current_item_num:
                        unstd_load = _to_num(row.get('Estimate', np.nan))
                        std_load = _to_num(row.get('Std.all', row.get('Std. All', np.nan)))
                        break
 
            rows.append({
                "measure_id": measure_id,
                "item_number": item_number,
                "item_text": text or item_raw,
                "reverse": rev,
                "variance_latent": trait_var, # ✨ 已成功获取
                
                "unstandardised_loading": unstd_load, # ✨ 精准匹配获取
                "standardised_loading": std_load,     # ✨ 精准匹配获取
                
                "chi2_user_model": chi2_val,
                "df_user_model": dof_val,
                "p_value_user_model": p_val,
                "CFI": _get_any(stats_dict, ["CFI"]),
                "TLI": _get_any(stats_dict, ["TLI"]),
                "RMSEA": _get_any(stats_dict, ["RMSEA"]),
                "SRMR": _get_any(stats_dict, ["SRMR", "srmr"]),
                "GFI": _get_any(stats_dict, ["GFI"]),
                "AGFI": _get_any(stats_dict, ["AGFI"]),
                "NFI": _get_any(stats_dict, ["NFI"]),
                "LogL": _get_any(stats_dict, ["LogL", "logl", "LogLik", "loglik", "log_likelihood", "log-likelihood"]),
                "AIC": _get_any(stats_dict, ["AIC"]),
                "BIC": _get_any(stats_dict, ["BIC"]),
                "SABIC": _get_any(stats_dict, ["SABIC"]),
                "item_mean": df_cfa[item_clean].mean() if item_clean in df_cfa.columns else np.nan,
                "item_sd": df_cfa[item_clean].std() if item_clean in df_cfa.columns else np.nan,
                "cronbach_alpha": alpha_val,
            })
            
            
        sheet_items = pd.DataFrame(rows)
       

        #if sheet_items["unstandardised_loading"].isna().all() and sheet_items["standardised_loading"].isna().all():
        #    st.error("❌ 报告生成失败：未提取到任何载荷，请检查模型是否包含载荷行。")
        #    return

        cov_matrix = df_cfa[factor_items].cov()

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            sheet_items.to_excel(w, sheet_name="Items", index=False)
            cov_matrix.to_excel(w, sheet_name="Covariance", index=True)
        buf.seek(0)

        today = date.today().strftime("%Y-%m-%d")
        safe_mid = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(measure_id)).strip(" .") or "measure"
        user_name = st.session_state.get("user_name", "unknown_user")
        safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name)).strip() or "unknown_user"
        filename = f"{safe_mid}_cfa_report_{today}.xlsx"
        st.download_button(
            label="⬇️ 点击下载 Excel 报告",
            data=buf.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"n2_{sub_name}_dl_excel",
        )
        st.success("报告已生成！")
    except Exception as e:
        st.error(f"生成报告时出错: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    

        




# ==============================================================================
# 🌟 顶层三大板块隔离调度中心
# ==============================================================================
def render_n1_analysis():
    st.title("模块 2: N1数据分析")

    # 使用 st.tabs 将三大核心分析板块在水平方向彻底隔离
    tab_efa_clean, tab_cfa_clean, tab_efa_final = st.tabs([
        "1. 自动删题 EFA 板块", 
        "2. 自动删题 CFA 板块", 
        "3. 最终不删题 EFA 板块"
    ])

    # 板块一：直接渲染原逻辑改名后的核心 EFA
    with tab_efa_clean:
        render_stage1_efa_clean()

    # 板块二：自动删题 CFA (读取 Stage 1 的 N1_preEFA 资产进行分析)
    with tab_cfa_clean:
        render_stage2_cfa_clean()

    # 板块三：不删题 EFA (用于最终论文汇报或验证最终锁定的题目)
    # with tab_efa_final:
        # render_stage3_efa_no_deletion()
