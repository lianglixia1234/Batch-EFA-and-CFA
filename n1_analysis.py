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
                        f"✅ 我已确认上述表格数据，并同意将该量表缓存同步至后续的 CFA 模块", 
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
                            "kept_items": list(kept),      # N1 过滤后确认保留的题目
                            "n_factors": int(n_factors),   # N1 模型推荐提取的因子数
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
                            f" * 🟢 内部原生标志: **`{sub_m}`** ── 保留题目: `{len(config['kept_items'])}` 题 "
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
# ==============================================================================
def render_stage2_cfa_clean():
    # 🧬 1. 精准读取上游 EFA 资产与底层实体数据
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
                    # 🌟 核心联动：精准提取 EFA 留下来的结果
                    if isinstance(m_config, dict) and "kept_items" in m_config:
                        raw_df_entity = m_config.get("clean_df")
                        
                        if raw_df_entity is None:
                            if "sub_datasets" in st.session_state and ds_key in st.session_state.sub_datasets:
                                raw_df_entity = st.session_state.sub_datasets[ds_key]
                            elif "dc_dataset_full" in st.session_state:
                                raw_df_entity = st.session_state.dc_dataset_full
                            else:
                                raw_df_entity = st.session_state.get("df_source")
                        
                        # 将 EFA 保留的题目作为 CFA 的初始分析目标
                        all_upstream_measures[str(m_id)] = {
                            "items": m_config["kept_items"],  # 👈 EFA 过滤后的纯净题目
                            "clean_df": raw_df_entity,        
                            "measure_id_raw": m_id,
                            "ds_key": ds_key
                        }

    if not all_upstream_measures:
        st.error("❌ 无法从上游 N1_preEFA 资产中提取到任何有效的量表数据。")
        return

    # ==========================================================================
    # 📂 2. 用户自主选择与精简保护参数配置
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🔍 第一步：勾选您本次需要分析的量表")
    
    selected_measure_ids = st.multiselect(
        "📂 请在下方选择要拉入 CFA 自动优化删题流的量表（默认已自动加载 EFA 保留项）：",
        options=list(all_upstream_measures.keys()),
        default=list(all_upstream_measures.keys())
    )
    
    if not selected_measure_ids:
        st.warning("⚠️ 请至少勾选一个量表以继续分析。")
        return

    sub_datasets = {k: all_upstream_measures[k] for k in selected_measure_ids}

    # 阈值与兜底限制设置
    col_param1 = st.columns(1)[0]
    with col_param1:
        min_items_limit = st.number_input(
            "🛑 最小保留题目底线",
            min_value=3, max_value=30, value=8, step=1,
            help="当维度内题目数减少到该值时，算法必须触发强制安全保护停止删题，防止被删空。"
        )

    
    



    # 🛠️ 第二步：CFA 测量模型结构核对看板 (Tabs 标签切换版)
    # ==========================================================================
    st.markdown("### 🛠️ 第二步：CFA 测量模型结构核对")
    
    # 🧬 1. 读取并解析批量量表底层资产 (联动上游结构)
    all_upstream_measures = {}
    n1_asset = st.session_state.get("N1_preEFA")
    
    if isinstance(n1_asset, dict) and n1_asset:
        for ds_key, measure_dict in n1_asset.items():
            if isinstance(measure_dict, dict):
                for m_id, m_config in measure_dict.items():
                    if isinstance(m_config, dict) and "kept_items" in m_config:
                        raw_df_entity = m_config.get("clean_df")
                        if raw_df_entity is None:
                            if "sub_datasets" in st.session_state and ds_key in st.session_state.sub_datasets:
                                raw_df_entity = st.session_state.sub_datasets[ds_key]
                            else:
                                raw_df_entity = st.session_state.get("df_source")
                        
                        all_upstream_measures[str(m_id)] = {
                            "items": m_config["kept_items"],
                            "clean_df": raw_df_entity,
                            "ds_key": ds_key
                        }
    elif 'efa_suggested_structure' in st.session_state:
        structure = st.session_state.efa_suggested_structure
        if isinstance(structure, dict):
            for factor_key, items_list in structure.items():
                all_upstream_measures[str(factor_key)] = {
                    "items": items_list,
                    "clean_df": st.session_state.get("df_source"),
                    "ds_key": "default"
                }

    if not all_upstream_measures:
        st.info("💡 暂未检测到上游 EFA 留存的题目资产结构。已自动使用当前数据集的全部列作为默认量表池。")
        current_df = st.session_state.get("df_source")
        all_items_pool = list(current_df.columns) if current_df is not None else []
        all_upstream_measures["Default_Measure"] = {"items": all_items_pool, "clean_df": current_df, "ds_key": "default"}
    
    # 📂 允许用户筛选参与核对的量表
    active_measure_ids = list(all_upstream_measures.keys())
    
    # 🌟 核心改动：使用 st.tabs 动态创建量表切换标签页
    if active_measure_ids:
        tabs = st.tabs([f"{m_id}" for m_id in active_measure_ids])
        
        # 建立全局就绪队列缓存，供后续批量运行按钮直接提取
        cfa_ready_queue = {}
        
        # 🔄 2. 遍历标签页，每个 Tab 内独立配置一个量表的结构
        for index, sub_name in enumerate(active_measure_ids):
            with tabs[index]:
                asset_body = all_upstream_measures[sub_name]
                all_items = sort_item_cols_by_number(list(asset_body.get("items", [])))
                
                st.markdown(f"#### ⚙️ 配置 【{sub_name}】 的因子结构")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🅰️ 主因子 (Trait Factor)")
                    factor_name = st.text_input(
                        "给主因子起个名 (英文):", 
                        value=f"{sub_name}", 
                        key=f"cfa_fname_inp_{sub_name}"
                    )
                    
                    factor_items = smart_multiselect(
                        options=all_items,
                        label=f"选择属于 {factor_name} 的题目",
                        key_suffix=f"cfa_factor_{sub_name}",
                        default_selected=all_items, # 默认预选 EFA 留下来的全部题目
                        show_selection_controls=True,
                    )

                with col2:
                    st.markdown("##### 🅱️ 方法因子 (Method Factor)")
                    method_name = st.text_input(
                        "给方法因子起个名 (英文):", 
                        value=f"Method", 
                        key=f"cfa_mname_inp_{sub_name}"
                    )
                    
                    method_options = factor_items if factor_items else []
                    method_key_suffix = f"cfa_method_{sub_name}"
                    method_sig_key = f"{method_key_suffix}_options_sig"
                    method_options_sig = tuple(method_options)
                    
                    # 智能缓存判定与清理
                    if st.session_state.get(method_sig_key) != method_options_sig:
                        if '_reset_smart_multiselect_cache' in globals() or '_reset_smart_multiselect_cache' in locals():
                            _reset_smart_multiselect_cache(method_key_suffix)
                        st.session_state[method_sig_key] = method_options_sig
                    
                    # 规则自动预选
                    default_method_items = [x for x in method_options if _is_reverse_coded(x)] if method_options else []
                    st.caption("已根据统一规则（题目末尾为 r）自动预选方法因子题目。")
                    
                    # 闭包重置回调函数
                    def _on_reset_method_tab(k_suffix=method_key_suffix, sig_val=method_options_sig, def_items=default_method_items):
                        if '_reset_smart_multiselect_cache' in globals() or '_reset_smart_multiselect_cache' in locals():
                            _reset_smart_multiselect_cache(k_suffix)
                        st.session_state[f"{k_suffix}_options_sig"] = sig_val
                        st.session_state[f"{k_suffix}_last_selected"] = def_items

                    st.button(
                        '🔄 重新预选方法因子题目', 
                        key=f"tab_btn_reset_method_{sub_name}",
                        on_click=_on_reset_method_tab
                    )
                    
                    method_items = smart_multiselect(
                        options=method_options,
                        label=f"选择受到 {method_name} 影响的题目",
                        key_suffix=method_key_suffix,
                        default_selected=default_method_items,
                        show_selection_controls=True,
                    )
                
                
                # 安全压入全局准备队列缓存
                if factor_items:
                    cfa_ready_queue[sub_name] = {
                        "asset_body": asset_body,
                        "df_numeric": asset_body.get("clean_df"),
                        "factor_name": factor_name,
                        "method_name": method_name if method_items else None,
                        "factor_items": list(factor_items),
                        "method_items": list(method_items) if method_items else []
                    }
        
        # 🚀 3. 统一运行与批量调整控制台 (呈现在 Tabs 下方)
        st.markdown("---")
        st.info(f"📋 当前标签页中已就绪的量表队列: `{', '.join(list(cfa_ready_queue.keys()))}`")
        
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            if st.button("🚀 开始批量多量表 CFA 分析 ", type="primary", key="batch_run_cfa_all_tabs"):
                if not cfa_ready_queue:
                    st.error("❌ 队列中没有配置有效的量表题目，请检查各 Tab 标签页内的选择。")
                else:
                    st.session_state["cfa_batch_queue_payload"] = cfa_ready_queue

                    # 下方即可直接触发批量纯化引擎的遍历循环
        
    # --- 3. 模型拟合 ---
    st.markdown("---")
    cfa_btn_col, cfa_prelim_col = st.columns([1, 1])
    with cfa_btn_col:
        run_clicked = st.button("🚀 开始运行 CFA 分析", type="primary")
    with cfa_prelim_col:
        if "n2_prelim_single_cfa" not in st.session_state:
            st.session_state.n2_prelim_single_cfa = False
        prelim_checked = st.checkbox("当前为 preliminary CFA", value=st.session_state.n2_prelim_single_cfa, key="n2_prelim_checkbox")
        st.session_state.n2_prelim_single_cfa = prelim_checked
    if run_clicked:
        if not factor_items:
            st.error("❌ 错误：请至少为主因子选择 1 个题目。")
        else:
            with st.spinner("正在拟合模型，请稍候..."):
                factor_items_for_model = sort_item_cols_by_number(list(factor_items))
                method_items_for_model = sort_item_cols_by_number(list(method_items)) if method_items else []
                result, err_msg, syntax_used = run_cfa_gui(
                    df_numeric, factor_name, factor_items_for_model, method_name, method_items_for_model
                )
                
                if err_msg:
                    st.error(err_msg)
                    st.code(syntax_used, language="text")
                else:
                    model_obj, estimates, fit_stats = result
                    st.success("✅ 模型拟合成功！")
                    
                    st.session_state.n2_estimates = estimates
                    st.session_state.n2_fit_stats = fit_stats
                    st.session_state.n2_syntax = syntax_used
                    st.session_state.n2_factor_name = factor_name
                    st.session_state.n2_method_name = method_name
                    # 保存用于可下载报告：CFA 使用的数据与题目列表
                    df_cfa_used = df_numeric[factor_items_for_model].dropna(axis=0)
                    st.session_state.n2_df_cfa = df_cfa_used
                    st.session_state.n2_factor_items = list(factor_items_for_model)
                    st.session_state.n2_method_items = list(method_items_for_model)

    # --- 4. 结果展示 ---
    if 'n2_fit_stats' in st.session_state:
        st.markdown("---")
        st.subheader("2. 分析结果")
        
        stats_dict = st.session_state.n2_fit_stats
        
        # 1. 关键指标高亮 (Top 8 Highlight)
        st.markdown("###### 🏆 关键模型拟合指标 (Key Fit Indices)")
        
        def get_val(key):
            val = stats_dict.get(key, np.nan)
            return val if isinstance(val, (int, float)) else np.nan

        metrics = {
            "CFI": get_val("CFI"),
            "TLI": get_val("TLI"),
            "RMSEA": get_val("RMSEA"),
            "SRMSR": get_val("SRMR"),
            # 🆕 修改标签: Explicitly User Model
            "Chi-Square (User Model)": get_val("chi2"),
            "AIC": get_val("AIC"),
            "BIC": get_val("BIC"),
            "SABIC": get_val("SABIC")
        }

        m_cols1 = st.columns(4)
        keys1 = ["CFI", "TLI", "RMSEA", "SRMSR"]
        for i, k in enumerate(keys1):
            val = metrics[k]
            display_val = f"{val:.3f}" if not np.isnan(val) else "N/A"
            m_cols1[i].metric(label=k, value=display_val)

        st.markdown("") 
        m_cols2 = st.columns(4)
        # 🆕 修改标签列表
        keys2 = ["Chi-Square (User Model)", "AIC", "BIC", "SABIC"] 
        for i, k in enumerate(keys2):
            val = metrics[k]
            display_val = f"{val:.3f}" if not np.isnan(val) else "N/A"
            m_cols2[i].metric(label=k, value=display_val)

        # 2. 详细表格 (Estimates)
        st.markdown("---")
        t1, t2 = st.tabs(["📄 详细参数估计 (Estimates)", "🔍 完整拟合报告"])
        
        with t1:
            #st.caption("Standardized Estimates (Std. Est) 为标准化载荷。")
            st.caption("Latent Variables (Factor Loadings) & Covariances")
            est_df = st.session_state.n2_estimates.copy() # 复制一份，避免修改原数据
            # --- [新增] 排序逻辑 (Part 1-5) ---
            fname = st.session_state.n2_factor_name
            mname = st.session_state.n2_method_name
            
            def get_sort_rank(row):
                lhs, op, rhs = row['LHS'], row['op'], row['RHS']
                # Part 1: 每个题目 ~ 主因子 (op='=~', LHS=主因子)
                if op == '=~' and lhs == fname: return 1
                # Part 2: 每个题目 ~ 方法因子 (op='=~', LHS=方法因子)
                if op == '=~' and lhs == mname: return 2
                # Part 3: 主因子 ~ 主因子 (Variance: op='~~', LHS=RHS=主因子)
                if op == '~~' and lhs == rhs and lhs == fname: return 3
                # Part 4: 方法因子 ~ 方法因子 (Variance: op='~~', LHS=RHS=方法因子)
                if op == '~~' and lhs == rhs and lhs == mname: return 4
                # Part 5: 每个题目 ~ 每个题目 (Residuals: op='~~', LHS=RHS, LHS!=因子)
                if op == '~~' and lhs == rhs and lhs not in [fname, mname]: return 5
                
                # 其他 (如 Covariance: Factor ~~ Method，放在最后)
                return 6

            # 应用排序
            est_df['rank'] = est_df.apply(get_sort_rank, axis=1)
            est_df = est_df.sort_values('rank').drop(columns=['rank'])
            
            # --- 格式化并显示 ---
            numeric_cols = est_df.select_dtypes(include=[np.number]).columns
            format_dict = {col: "{:.3f}" for col in numeric_cols}
            
            # 筛选展示列 (模仿 lavaan)
            display_cols = ['LHS', 'op', 'RHS', 'Estimate', 'Std.Err', 'z-value', 'P(>|z|)', 'Std.all']
            # 防止某些列不存在
            final_cols = [c for c in display_cols if c in est_df.columns]
            
            st.dataframe(est_df[final_cols].style.format(format_dict))
            
            csv = est_df.to_csv().encode('utf-8-sig')
            st.download_button("📥 下载参数估计表", csv, "cfa_estimates.csv", "text/csv")
            '''
            # === 🆕 新增：自定义排序逻辑 ===
            # 获取运行分析时使用的因子名称
            if 'n2_factor_names' in st.session_state:
                trait_name, method_name = st.session_state.n2_factor_names
            else:
                # 兼容旧状态（以防万一）
                trait_name, method_name = factor_name, method_name

            def get_sort_rank(row):
                lhs, op, rhs = row['LHS'], row['op'], row['RHS']
                
                # Rank 1: 每个题目 ~ 主因子 (Factor Loadings - Trait)
                # op 是 =~, 左边是主因子名
                if op == '=~' and lhs == trait_name:
                    return 1
                
                # Rank 2: 每个题目 ~ 方法因子 (Factor Loadings - Method)
                # op 是 =~, 左边是方法因子名
                if op == '=~' and lhs == method_name:
                    return 2
                
                # Rank 3: 主因子 ~ 主因子 (Latent Variance)
                # op 是 ~~, 左右相等, 且是主因子
                if op == '~~' and lhs == rhs and lhs == trait_name:
                    return 3
                
                # Rank 4: 方法因子 ~ 方法因子 (Method Variance)
                # op 是 ~~, 左右相等, 且是方法因子
                if op == '~~' and lhs == rhs and lhs == method_name:
                    return 4
                
                # Rank 5: 每个题目 ~ 每个题目 (Residual Variances)
                # op 是 ~~, 左右相等, 且不是因子名
                if op == '~~' and lhs == rhs and lhs not in [trait_name, method_name]:
                    return 5
                
                # 其他 (例如因子间的协方差，虽然这里设为0但也存在)
                return 6

            # 应用排序: 先按 Rank 排，Rank 相同的按 RHS (题目名) 排
            est_df['Sort_Rank'] = est_df.apply(get_sort_rank, axis=1)
            est_df = est_df.sort_values(by=['Sort_Rank', 'RHS'])
            
            # 移除辅助列，准备展示
            # 筛选我们关心的列 (模仿 lavaan 输出)
            display_cols = ['LHS', 'op', 'RHS', 'Estimate', 'Std.Err', 'z-value', 'P(>|z|)', 'Std.all']
            final_cols = [c for c in display_cols if c in est_df.columns]
            est_display = est_df[final_cols].copy()
            # =================================
            
            # 只对数值列应用格式化
            numeric_cols = est_display.select_dtypes(include=[np.number]).columns
            format_dict = {col: "{:.3f}" for col in numeric_cols}
            
            st.dataframe(est_display.style.format(format_dict))
            
            csv = est_display.to_csv().encode('utf-8-sig')
            st.download_button("📥 下载参数估计表", csv, "cfa_estimates.csv", "text/csv")
            '''
        '''    
        with t2:
            st.write("所有计算出的拟合指数：")
            # 将字典转为 DataFrame 展示
            fit_df_full = pd.DataFrame([stats_dict]).T
            fit_df_full.columns = ["Value"]
            
            st.dataframe(fit_df_full.style.format("{:.3f}"))
            
            st.markdown("**生成的模型语法 (Syntax Used):**")
            st.code(st.session_state.n2_syntax, language="text")
        '''
        with t2:
            st.write("### Model Test User Model:")
            
            # --- [新增] 单独展示 User Model Chi-Square ---
            # semopy calc_stats 使用 DoF、chi2 p-value 等键名，兼容多种写法
            def _get_any(d, keys, default=np.nan):
                for k in keys:
                    v = d.get(k, default)
                    if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        return v
                return default
            chi2_val = _get_any(stats_dict, ['chi2', 'Chi2'])
            dof_val = _get_any(stats_dict, ['DoF', 'dof', 'df'])
            p_val = _get_any(stats_dict, ['chi2 p-value', 'p-value', 'pvalue', 'p_value'])
            
            # 构建特定格式的表格
            model_test_df = pd.DataFrame({
                "Statistic": ["Test statistic", "Degrees of freedom", "P-value (Chi-square)"],
                "Value": [
                    f"{chi2_val:.3f}" if not np.isnan(chi2_val) else "N/A",
                    f"{int(dof_val)}" if not np.isnan(dof_val) else "N/A",
                    f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
                ]
            })
            
            st.table(model_test_df)
            
            st.write("### User Model versus Baseline Model:")
            # 展示其他所有指标 (作为 Baseline 对比参考)
            fit_df_full = pd.DataFrame([stats_dict]).T
            fit_df_full.columns = ["Value"]
            
            # 安全格式化
            num_cols_fit = fit_df_full.select_dtypes(include=[np.number]).columns
            format_dict_fit = {col: "{:.3f}" for col in num_cols_fit}
            
            st.dataframe(fit_df_full.style.format(format_dict_fit))
            
            st.markdown("**生成的模型语法 (Syntax Used):**")
            st.code(st.session_state.n2_syntax, language="text")

        # --- 5. 生成可下载报告 (measure_id + 题目表 + 协方差矩阵) ---
        if all(k in st.session_state for k in ("n2_df_cfa", "n2_factor_items", "n2_estimates", "n2_fit_stats")):
            st.markdown("---")
            st.text_input(
                "量表 measure_id（唯一编码，用于所有可下载文件命名）",
                value=(st.session_state.get("n2_measure_id") or ""),
                key="n2_measure_id",
                placeholder="如 LQ、EQ 等问卷缩写",
                help="下方「生成可下载报告表」「最终得分计算」「导出公式参数表」等功能的下载文件均沿用此 measure_id。",
            )
            st.markdown("#### 📥 生成可下载报告表")
            st.caption("一表为每题一行的报告，一表为题目协方差矩阵。")
            if st.button("生成并下载 Excel 报告", key="n2_btn_gen_report"):
                mid = (st.session_state.get("n2_measure_id") or "").strip() or "measure"
                if not mid:
                    st.warning("请填写 measure_id。")
                else:
                    try:
                        df_cfa = st.session_state.n2_df_cfa
                        factor_items = st.session_state.n2_factor_items
                        estimates = st.session_state.n2_estimates
                        stats_dict = st.session_state.n2_fit_stats
                        fname = st.session_state.n2_factor_name

                        # 列名清洗函数（与 run_cfa_gui 中一致）
                        def _clean_col(name):
                            return re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(name))

                        # 建立原始列名到清洗后列名的映射（因为 estimates 中的 RHS 是清洗后的列名）
                        item_clean_map = {item: _clean_col(item) for item in factor_items}

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

                        # 兼容不同来源/版本的拟合指标键名（大小写/符号不敏感）
                        def _norm_key(k):
                            return re.sub(r"[^a-z0-9]+", "", str(k).lower())

                        _stats_norm = {_norm_key(k): v for k, v in stats_dict.items()}

                        def _get_any(d, keys, default=np.nan):
                            # 1) 先按原键精确匹配
                            for k in keys:
                                if k in d:
                                    v = _to_num(d.get(k))
                                    if not np.isnan(v):
                                        return v
                            # 2) 再按归一化键名匹配
                            for k in keys:
                                nk = _norm_key(k)
                                if nk in _stats_norm:
                                    v = _to_num(_stats_norm.get(nk))
                                    if not np.isnan(v):
                                        return v
                            return default

                        # 潜变量方差 (主因子)
                        trait_var = np.nan
                        est = estimates
                        for _, row in est.iterrows():
                            if row.get("op") == "~~" and row.get("LHS") == fname and row.get("RHS") == fname:
                                trait_var = row.get("Estimate", np.nan)
                                break

                        # 载荷：与「Latent Variables (Factor Loadings) & Covariances」表同一逻辑取数
                        # 兼容 semopy 两种常见表示：
                        # A) op='=~' 且 LHS=潜变量, RHS=题目
                        # B) op='~'  且 RHS=潜变量, LHS=题目
                        loadings_unstd = {}
                        loadings_std = {}
                        if "LHS" in est.columns and "op" in est.columns and "RHS" in est.columns:
                            trait_loadings = est[(est["op"] == "=~") & (est["LHS"] == fname)]
                            if not trait_loadings.empty:
                                for _, row in trait_loadings.iterrows():
                                    item_key = row["RHS"]
                                    loadings_unstd[item_key] = _to_num(row["Estimate"]) if "Estimate" in est.columns else np.nan
                                    loadings_std[item_key] = _to_num(row["Std.all"]) if "Std.all" in est.columns else np.nan
                            else:
                                trait_loadings = est[(est["op"] == "~") & (est["RHS"] == fname)]
                                for _, row in trait_loadings.iterrows():
                                    item_key = row["LHS"]
                                    loadings_unstd[item_key] = _to_num(row["Estimate"]) if "Estimate" in est.columns else np.nan
                                    loadings_std[item_key] = _to_num(row["Std.all"]) if "Std.all" in est.columns else np.nan

                        chi2_val = _get_any(stats_dict, ["chi2", "Chi2"])
                        dof_val = _get_any(stats_dict, ["DoF", "dof", "df"])
                        p_val = _get_any(stats_dict, ["chi2 p-value", "p-value", "pvalue", "p_value"])
                        alpha_val = cronbach_alpha(df_cfa) if not df_cfa.empty else np.nan

                        # Composite Reliability (CR_F): 使用完全标准化载荷
                        # lambda_std_i = lambda_i * sqrt(phi) / s_i, 其中 s_i 来自题目协方差矩阵 Sigma 的对角线
                        cr_val = np.nan
                        cr_reason = ""
                        try:
                            sorted_items_for_cr = sort_item_cols_by_number(factor_items)
                            sorted_items_clean_for_cr = [item_clean_map.get(c, c) for c in sorted_items_for_cr]
                            used_cols_for_cr = [c for c in sorted_items_clean_for_cr if c in df_cfa.columns]
                            if not used_cols_for_cr:
                                cr_reason = "CR 未计算：未找到用于 CR 的题目列。"
                            else:
                                x_cr = df_cfa[used_cols_for_cr].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
                                if x_cr.empty:
                                    cr_reason = "CR 未计算：用于 CR 的有效样本为空（题目存在缺失/非数值）。"
                                else:
                                    sigma_cr = x_cr.cov().values
                                    s_vec = np.sqrt(np.diag(sigma_cr))
                                    lambda_unstd_vec = np.array(
                                        [_to_num(loadings_unstd.get(c, np.nan)) for c in used_cols_for_cr],
                                        dtype=float,
                                    )
                                    phi_num = _to_num(trait_var)
                                    if np.isnan(phi_num) or phi_num <= 0:
                                        cr_reason = "CR 未计算：主因子方差 φ 缺失或非正数。"
                                    elif np.isnan(lambda_unstd_vec).any():
                                        miss_cols = [used_cols_for_cr[i] for i, v in enumerate(lambda_unstd_vec) if np.isnan(v)]
                                        cr_reason = f"CR 未计算：以下题目缺少非标准化载荷：{', '.join(miss_cols[:6])}"
                                    elif (not np.all(np.isfinite(s_vec))) or np.any(s_vec <= 0):
                                        cr_reason = "CR 未计算：题目标准差（来自协方差矩阵对角线）存在无效值。"
                                    else:
                                        lambda_std = (lambda_unstd_vec * np.sqrt(phi_num)) / s_vec
                                        S = float(np.sum(lambda_std))
                                        E = float(np.sum(1.0 - lambda_std ** 2))
                                        den = (S ** 2) + E
                                        if np.isfinite(den) and den > 0:
                                            cr_val = float((S ** 2) / den)
                                        else:
                                            cr_reason = "CR 未计算：分母无效（可能由异常载荷导致）。"
                        except Exception as cr_e:
                            cr_val = np.nan
                            cr_reason = f"CR 未计算：计算过程异常（{cr_e}）。"

                        def _extract_item_number(item_name, item_clean_name, fallback_idx):
                            # 1) 优先使用现有解析逻辑（EFA/CFA 前缀）
                            _, num_parsed, _ = parse_item_col(item_name)
                            if num_parsed is not None:
                                return num_parsed
                            _, num_clean_parsed, _ = parse_item_col(item_clean_name)
                            if num_clean_parsed is not None:
                                return num_clean_parsed

                            # 2) 次选：从“前缀段”里提取数字（如 Q12_xxx / item_3_xxx）
                            prefix_orig = str(item_name).split("_", 1)[0]
                            prefix_clean = str(item_clean_name).split("_", 1)[0]
                            m = re.search(r"(\d+)", prefix_orig)
                            if m:
                                return int(m.group(1))
                            m2 = re.search(r"(\d+)", prefix_clean)
                            if m2:
                                return int(m2.group(1))

                            # 3) 兜底：用当前顺序序号，避免导出空值
                            return fallback_idx

                        sorted_items = sort_item_cols_by_number(factor_items)
                        rows = []
                        for idx, item in enumerate(sorted_items, start=1):
                            _, num, text = parse_item_col(item)
                            rev = 1 if _is_reverse_coded(item) else 0
                            # 使用清洗后的列名去 estimates 中查找载荷值
                            item_clean = item_clean_map.get(item, item)
                            item_number = num if num is not None else _extract_item_number(item, item_clean, idx)
                            rows.append({
                                "measure_id": mid,
                                "item_number": item_number,
                                "item_text": text or item,
                                "reverse": rev,
                                "variance_latent": trait_var,
                                "unstandardised_loading": loadings_unstd.get(item_clean, np.nan),
                                "standardised_loading": loadings_std.get(item_clean, np.nan),
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
                                "Composite Reliability (CR)": cr_val,
                            })
                        sheet_items = pd.DataFrame(rows)

                        # 生成前校验提示：避免导出“看似成功但关键列为空”的报告
                        unstd_empty = ("unstandardised_loading" not in sheet_items.columns) or sheet_items["unstandardised_loading"].isna().all()
                        std_empty = ("standardised_loading" not in sheet_items.columns) or sheet_items["standardised_loading"].isna().all()
                        logl_empty = ("LogL" not in sheet_items.columns) or sheet_items["LogL"].isna().all()

                        # 载荷列是核心字段：若全空则阻止生成并给出定位提示
                        if unstd_empty and std_empty:
                            st.error(
                                "生成前校验未通过：`unstandardised_loading` 与 `standardised_loading` 全为空。"
                                "请先确认参数表中存在因子载荷行，并且题目列名与模型估计表可匹配后再生成。"
                            )
                            st.info(
                                "排查建议：\n"
                                "1) 查看「Latent Variables (Factor Loadings) & Covariances」是否有载荷行；\n"
                                "2) 确认选中的题目确实进入模型；\n"
                                "3) 重新运行一次 CFA 后再生成报告。"
                            )
                            st.stop()

                        # LogL 在部分版本/估计器下可能缺失：提示但允许生成
                        if logl_empty:
                            st.warning(
                                "提示：LogL 当前为空（可能由 semopy 版本或拟合输出键名差异导致），"
                                "其余字段将正常导出。"
                            )

                        # 协方差矩阵：题目按序号排序，对角线为方差（使用清洗后列名）
                        sorted_items_clean = [item_clean_map.get(c, c) for c in sorted_items]
                        df_cfa_ordered = df_cfa[[c for c in sorted_items_clean if c in df_cfa.columns]]
                        cov_matrix = df_cfa_ordered.cov()

                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                            # 固定 sheet 名，避免 measure_id 过长导致截断后重名覆盖
                            sheet_items.to_excel(w, sheet_name="Items", index=False)
                            cov_matrix.to_excel(w, sheet_name="Covariance", index=True)
                        buf.seek(0)
                        st.session_state.n2_excel_report_bytes = buf.getvalue()
                        cfa_type = "prelim_single_cfa" if st.session_state.get("n2_prelim_single_cfa") else "single_cfa"
                        safe_mid_for_file = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(mid)).strip(" .") or "measure"
                        user_name = st.session_state.get("user_name", "unknown_user")
                        safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name)).strip() or "unknown_user"
                        today = date.today().strftime("%Y-%m-%d")
                        st.session_state.n2_excel_report_filename = f"{safe_mid_for_file}_{cfa_type}_report_{today}_{safe_user}.xlsx"
                        st.session_state.n2_report_sheet_items_preview = sheet_items.copy()
                        st.session_state.n2_report_cov_preview = cov_matrix.copy()
                        st.session_state.n2_cr_warning = cr_reason if (np.isnan(_to_num(cr_val)) and cr_reason) else ""
                        st.success("已生成报告，请点击下方按钮下载。")
                    except Exception as e:
                        st.error(f"生成报告时出错: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            if st.session_state.get("n2_report_sheet_items_preview") is not None:
                st.markdown("##### 预览：题目明细表（前20行）")
                st.dataframe(
                    st.session_state.n2_report_sheet_items_preview.head(20),
                    use_container_width=True,
                )
            if st.session_state.get("n2_report_cov_preview") is not None:
                st.markdown("##### 预览：题目协方差矩阵（前20行）")
                st.dataframe(
                    st.session_state.n2_report_cov_preview.head(20),
                    use_container_width=True,
                )
            if st.session_state.get("n2_cr_warning"):
                st.warning(st.session_state.n2_cr_warning)

            if st.session_state.get("n2_excel_report_bytes"):
                st.download_button(
                    "⬇️ 下载 Excel 报告",
                    data=st.session_state.n2_excel_report_bytes,
                    file_name=st.session_state.get("n2_excel_report_filename", "measure_single_cfa_report.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="n2_download_excel_report",
                )




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
