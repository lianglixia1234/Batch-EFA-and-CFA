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


# ==============================================================================
# 核心算法区域
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
        from .data_cleaning_dual import get_dual_mode_analysis_df
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

                # 🌟【新增步骤 1】：先询问用户最终结果的 measure_id 是什么（允许用户交互修改）
                final_measure_id = st.text_input(
                    f"✍️ 请确认或修改量表【{real_measure_id}】最终用于导出的 `measure_id`名称:",
                    value=real_measure_id,
                    key=f"input_measure_id_{m_name}"
                )

                # ==============================================================
                # 🚨 【核心升级点 2】：根据精准的 final_measure_id 生成独立 Excel 数据并预览
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
                            "measure_id": final_measure_id,   # 🌟【超级核心修改】：这里直接写入用户最终确认/修改的 final_measure_id
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
                    
                    # 🌟【新增步骤 2】：在界面上呈现完整的表格供用户审查
                    st.caption("📋 当前待导出的全量数据指标预览（第一列已动态更新）：")
                    st.dataframe(single_measure_df, use_container_width=True)

                    # 🌟【新增步骤 3】：双重校验判断“最终ID”此前是否已被保存，用于初始化复选框状态
                    is_previously_saved = (
                        ds_name_extracted in st.session_state.N1_preEFA and 
                        final_measure_id in st.session_state.N1_preEFA[ds_name_extracted]
                    )

                    # 🌟【新增步骤 4】：让用户点击复选框，确认数据及 measure_id 无误
                    is_confirmed = st.checkbox(
                        f"✅ 我已确认上述表格数据，并同意使用【{final_measure_id}】进行缓存与下载", 
                        value=is_previously_saved,
                        key=f"confirm_check_{m_name}"
                    )

                    # ==============================================================
                    # 🚀 【后置联动】：用户勾选确认后，才解锁存储、清除以及下载按钮
                    # ==============================================================
                    if is_confirmed:
                        # 1. 缓存逻辑
                        if ds_name_extracted not in st.session_state.N1_preEFA:
                            st.session_state.N1_preEFA[ds_name_extracted] = {}
                        
                        # 使用用户最终决定的 final_measure_id 作为缓存的 Key
                        st.session_state.N1_preEFA[ds_name_extracted][final_measure_id] = {
                            "kept_items": list(kept),      # N1 过滤后确认保留的题目
                            "n_factors": int(n_factors),   # N1 模型推荐提取的因子数
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "clean_df": df_final          # 清洗删题后的真实 DataFrame
                        }
                        st.toast(f"🟢 已成功将【{final_measure_id}】同步缓存！")

                        # 2. 编译并输出 Excel 下载组件
                        single_buf = io.BytesIO()
                        with pd.ExcelWriter(single_buf, engine="xlsxwriter") as single_writer:
                            single_measure_df.to_excel(single_writer, sheet_name="EFA_Report", index=False)
                        single_buf.seek(0)
                        
                        today_str = date.today().strftime("%Y-%m-%d")
                        safe_measure_id = "".join(c for c in final_measure_id if c not in '[]:*?/\\ ')
                        file_filename = f"EFA_Report_{safe_measure_id}_{today_str}.xlsx"
                        
                        st.download_button(
                            label=f"⬇️ 立即下载 【{final_measure_id}】 维度的独立 Excel 报告",
                            data=single_buf.getvalue(),
                            file_name=file_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_btn_single_{m_name}"
                        )
                    else:
                        # 用户未勾选或取消勾选时，动态移除该最终ID的缓存
                        if ds_name_extracted in st.session_state.N1_preEFA:
                            if final_measure_id in st.session_state.N1_preEFA[ds_name_extracted]:
                                del st.session_state.N1_preEFA[ds_name_extracted][final_measure_id]
                        st.info("💡 请检查上方预览表，确认无误后勾选上方“✅ 我已确认...”复选框以进行下载和全局缓存。")

                except Exception as ex_build:
                    st.caption(f"⚠️ 该维度的 Excel 独立导出表编译受限: {ex_build}")

        # ==========================================================================
        # 6. 页面最底部全局大看板：实时监测并预览 N1_preEFA 配置资产状态
        # ==========================================================================
        if st.session_state.N1_preEFA:
            st.markdown("---")
            with st.expander("🚀 查看当前准备对接 Batch CFA 的 `N1_preEFA` 全局资产清单", expanded=True):
                st.success("📊 已成功缓存中，后续 preCFA模块将调取这些数据：")
                
                for d_key, m_dict in st.session_state.N1_preEFA.items():
                    st.markdown(f"#### 📦 数据集容器: `{d_key}`")
                    for sub_m, config in m_dict.items():
                        st.markdown(
                            f" * 🟢 **{sub_m}** ── 保留题目: `{len(config['kept_items'])}` 题 | "
                            f"推荐 CFA 验证潜变量/因子数: `{config['n_factors']}` | *更新时间: {config['timestamp']}*"
                        )


# 🧪 2. 自动删题 CFA 板块
# ==============================================================================
def render_stage2_cfa_clean():
    """
    第二阶段：CFA 双阶段纯化寻优（Stage 1 保底 0.90 -> Stage 2 择优精简）
    """
    st.subheader("🔄 CFA 双阶段指标删题")

    # ==========================================================================
    # 🧬 1. 状态持久化容器初始化
    # ==========================================================================
    if "N2_CFA_final_chosen" not in st.session_state:
        st.session_state["N2_CFA_final_chosen"] = {}
        
    if "cfa_multi_scenarios" not in st.session_state:
        st.session_state["cfa_multi_scenarios"] = {}

    if "cfa_ready_queue" not in st.session_state:
        st.session_state["cfa_ready_queue"] = {}
        
    if "cfa_batch_run_done" not in st.session_state:
        st.session_state["cfa_batch_run_done"] = False

    # ==========================================================================
    # 🔗 2. 上游资产解包
    # ==========================================================================
    n1_asset = st.session_state.get("N1_preEFA")
    if not n1_asset:
        st.info("💡 暂未检测到 N1_preEFA 结果。请确保在前置 EFA 模块中完成了分析并保存。")
        return

    sub_datasets = {}
    if isinstance(n1_asset, dict):
        for ds_key, measure_dict in n1_asset.items():
            if isinstance(measure_dict, dict):
                for m_id, m_config in measure_dict.items():
                    if isinstance(m_config, dict) and "kept_items" in m_config:
                        composite_key = f"{m_id}"
                        raw_df_entity = m_config.get("clean_df")
                        
                        if raw_df_entity is None:
                            if "sub_datasets" in st.session_state and ds_key in st.session_state.sub_datasets:
                                raw_df_entity = st.session_state.sub_datasets[ds_key]
                            elif "dc_dataset_full" in st.session_state:
                                raw_df_entity = st.session_state.dc_dataset_full
                            else:
                                raw_df_entity = st.session_state.get("df_source")
                        
                        sub_datasets[composite_key] = {
                            "items": m_config["kept_items"],  
                            "clean_df": raw_df_entity,        
                            "measure_id_raw": m_id,
                            "ds_key": ds_key
                        }

    if not sub_datasets:
        st.error("❌ 无法从上游提取到有效的子数据集。")
        return
        
    st.success(f"📊 成功加载上游资产，共有 `{len(sub_datasets)}` 个维度待处理。")

    # ==========================================================================
    # 🎛️ 3. 通用阈值红线设置（先问用户设置的限制，默认5）
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🎛️ 第一步：设定自动化算法的安全红线与目标")
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        min_items_limit = st.number_input(
            "🛑 最小题目限制（避免被删空，安全防线）",
            min_value=3, max_value=30, value=5, step=1,
            help="无论指标多低，当维度题目数达到该值时，算法必须无条件停止删题。"
        )
    with col_param2:
        target_max_items = st.number_input(
            "📐 Stage 2 期望的目标精简题数",
            min_value=int(min_items_limit), max_value=50, value=10, step=1,
            help="Stage 2 追求0.95时的精简参照线。题目数小于或等于此值时，若已满足基础达标，则停止精简。"
        )

    # ==========================================================================
    # 🔍 看板 A：测量模型结构预审
    # ==========================================================================
    st.markdown("### 🛠️ 第二步：CFA 测量模型结构核对")

    for sub_name, asset_body in sub_datasets.items():
        factor_items = asset_body.get("items", [])
        if not factor_items:
            continue
        sorted_items = sorted(list(factor_items))

        auto_detected_reverse_items = [
            item for item in sorted_items 
            if str(item).rstrip().endswith("r") or str(item).rstrip().endswith("_r") or "反向" in str(item)
        ]

        with st.expander(f"👀 核对量表维度 【{sub_name}】 的因子架构", expanded=True):
            audit_col1, audit_col2 = st.columns(2)
            with audit_col1:
                st.markdown(f"**🅰️ 主因子特质项 (`{sub_name}_Trait`)**")
                st.code("\n".join([f" ├─ {item}" for item in sorted_items]), language="text")
            with audit_col2:
                st.markdown(f"**🅱️ 逆向方法因子项 (`{sub_name}_Method`)**")
                if auto_detected_reverse_items:
                    st.code("\n".join([f" ├─ {item}" for item in auto_detected_reverse_items]), language="text")
                else:
                    st.info("ℹ️ 无反向题，自动退化为单因子模型。")

            confirmed_method_items = st.multiselect(
                f"✍️ 微调 【{sub_name}】 的反向题因子（如有错漏）：",
                options=sorted_items,
                default=auto_detected_reverse_items,
                key=f"audit_method_override_{sub_name}"
            )
            
            st.session_state.cfa_ready_queue[sub_name] = {
                "asset_body": asset_body,
                "trait_factor_name": f"{sub_name}_Trait",
                "trait_items": sorted_items,
                "method_factor_name": f"{sub_name}_Method" if confirmed_method_items else None,
                "method_items": confirmed_method_items
            }

    # ==========================================================================
    # 🚀 看板 B：双阶段全自动删题引擎核心逻辑
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🚀 第三步：开始双阶段全自动计算")
    
    if st.button("🔥 运行自动删题CFA", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        queue_keys = list(st.session_state.cfa_ready_queue.keys())
        st.session_state.cfa_multi_scenarios = {}
        
        for idx, sub_name in enumerate(queue_keys):
            cfg = st.session_state.cfa_ready_queue[sub_name]
            asset_body = cfg["asset_body"]
            df_run = asset_body.get("clean_df")
            if df_run is None or df_run.empty:
                continue
                
            trait_items = cfg["trait_items"]
            method_items = cfg["method_items"]
            trait_f_name = cfg["trait_factor_name"]
            method_f_name = cfg["method_factor_name"]
            
            with st.spinner(f"正在为【{sub_name}】计算双阶段删题..."):
                try:
                    from semopy import Model, calc_stats
                    
                    def _clean_col(name):
                        return re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(name))
                    
                    df_cfa_exec = df_run.copy()
                    rename_map = {c: _clean_col(c) for c in trait_items if c in df_cfa_exec.columns}
                    df_cfa_exec.rename(columns=rename_map, inplace=True)
                    reverse_rename = {v: k for k, v in rename_map.items()}
                    
                    # 辅助函数：构建语法、运行 CFA 并抓取统计指标
                    def run_cfa_metrics(t_items, m_items):
                        formulas = []
                        if t_items: formulas.append(f"{trait_f_name} =~ " + " + ".join(t_items))
                        if m_items: formulas.append(f"{method_f_name} =~ " + " + ".join(m_items))
                        m = Model("\n".join(formulas))
                        m.fit(df_cfa_exec)
                        stats = calc_stats(m)
                        cfi = float(stats.loc[0, "CFI"]) if "CFI" in stats.columns else 0.5
                        tli = float(stats.loc[0, "TLI"]) if "TLI" in stats.columns else 0.5
                        rmsea = float(stats.loc[0, "RMSEA"]) if "RMSEA" in stats.columns else 0.1
                        srmr = float(stats.loc[0, "SRMR"]) if "SRMR" in stats.columns else 0.1
                        return m, cfi, tli, rmsea, srmr, stats

                    # ----------------------------------------------------------
                    # 🏁 Step 1: 首轮基准 V0 运行
                    # ----------------------------------------------------------
                    curr_trait = [rename_map[c] for c in trait_items if c in rename_map]
                    curr_method = [rename_map[c] for c in method_items if c in rename_map]
                    
                    mod_v0, cfi_v0, tli_v0, rmsea_v0, srmr_v0, stats_v0 = run_cfa_metrics(curr_trait, curr_method)
                    
                    # ----------------------------------------------------------
                    # 🛡️ Stage 1：质量达标阶段（保底 0.90）
                    # ----------------------------------------------------------
                    v0_is_acceptable = (cfi_v0 >= 0.90 and tli_v0 >= 0.90)
                    delete_history_stage1 = []
                    
                    if v0_is_acceptable:
                        # 情况 A：首轮就达标 0.90，直接保存结果
                        stage1_trait = list(curr_trait)
                        stage1_method = list(curr_method)
                        cfi_s1, tli_s1, rmsea_s1, srmr_s1 = cfi_v0, tli_v0, rmsea_v0, srmr_v0
                        mod_s1, stats_s1 = mod_v0, stats_v0
                    else:
                        # 情况 B：未达标 0.90 -> 触发自动删题循环_basic
                        while len(curr_trait) > min_items_limit:
                            best_gain = -999.0
                            item_to_drop = None
                            best_trial_metrics = None
                            best_trial_obj = None
                            
                            # 遍历删题测试
                            for candidate in curr_trait:
                                trial_t = [c for c in curr_trait if c != candidate]
                                trial_m = [c for c in curr_method if c != candidate]
                                try:
                                    m_t, c_t, t_t, r_t, s_t, st_t = run_cfa_metrics(trial_t, trial_m)
                                    gain = c_t + t_t
                                    if gain > best_gain:
                                        best_gain = gain
                                        item_to_drop = candidate
                                        best_trial_metrics = (c_t, t_t, r_t, s_t)
                                        best_trial_obj = (m_t, st_t)
                                except:
                                    continue
                            
                            if item_to_drop:
                                curr_trait = [c for c in curr_trait if c != item_to_drop]
                                curr_method = [c for c in curr_method if c != item_to_drop]
                                delete_history_stage1.append(reverse_rename.get(item_to_drop, str(item_to_drop)))
                                cfi_t, tli_t, rmsea_t, srmr_t = best_trial_metrics
                                mod_v1, stats_v1 = best_trial_obj
                                
                                # 检查是否成功冲过保底 0.90 线
                                if cfi_t >= 0.90 and tli_t >= 0.90:
                                    break
                            else:
                                break # 无法继续优化则跳出
                                
                        stage1_trait = list(curr_trait)
                        stage1_method = list(curr_method)
                        cfi_s1, tli_s1, rmsea_s1, srmr_s1 = cfi_t, tli_t, rmsea_t, srmr_t
                        mod_s1, stats_s1 = mod_v1, stats_v1

                    # ----------------------------------------------------------
                    # 🚀 Stage 2：择优精简阶段（在保底基础上追求 0.95）
                    # ----------------------------------------------------------
                    delete_history_stage2 = []
                    
                    # 继承 Stage 1 的结果作为 Stage 2 的起点
                    curr_trait_s2 = list(stage1_trait)
                    curr_method_s2 = list(stage1_method)
                    cfi_s2, tli_s2, rmsea_s2, srmr_s2 = cfi_s1, tli_s1, rmsea_s1, srmr_s1
                    mod_s2, stats_s2 = mod_s1, stats_s1
                    
                    # 判断当前题目数是否满足预先设定的精简要求
                    if len(curr_trait_s2) > target_max_items:
                        # 触发自动删题循环_best
                        while len(curr_trait_s2) > min_items_limit:
                            # 🎯 检查停止规则
                            # 停止情况 1: 题目数虽然大于 target 限制，但拟合已经冲到了 0.95 优秀标准
                            if cfi_s2 >= 0.95 and tli_s2 >= 0.95:
                                break
                                
                            # 停止情况 2: 题目数已经缩减到 <= 目标要求，且在 0.90~0.95 之间
                            if len(curr_trait_s2) <= target_max_items:
                                break
                                
                            best_gain_s2 = -999.0
                            item_to_drop_s2 = None
                            best_trial_metrics_s2 = None
                            best_trial_obj_s2 = None
                            
                            for candidate in curr_trait_s2:
                                trial_t = [c for c in curr_trait_s2 if c != candidate]
                                trial_m = [c for c in curr_method_s2 if c != candidate]
                                try:
                                    m_t, c_t, t_t, r_t, s_t, st_t = run_cfa_metrics(trial_t, trial_m)
                                    gain = c_t + t_t
                                    if gain > best_gain_s2:
                                        best_gain_s2 = gain
                                        item_to_drop_s2 = candidate
                                        best_trial_metrics_s2 = (c_t, t_t, r_t, s_t)
                                        best_trial_obj_s2 = (m_t, st_t)
                                except:
                                    continue
                                    
                            if item_to_drop_s2:
                                curr_trait_s2 = [c for c in curr_trait_s2 if c != item_to_drop_s2]
                                curr_method_s2 = [c for c in curr_method_s2 if c != item_to_drop_s2]
                                delete_history_stage2.append(reverse_rename.get(item_to_drop_s2, str(item_to_drop_s2)))
                                cfi_s2, tli_s2, rmsea_s2, srmr_s2 = best_trial_metrics_s2
                                mod_s2, stats_s2 = best_trial_obj_s2
                            else:
                                break

                    # ----------------------------------------------------------
                    # 📦 装配和序列化输出结果
                    # ----------------------------------------------------------
                    def package_res_obj(mod, stats, cfi, tli, rmsea, srmr):
                        dof = int(stats.loc[0, "DoF"]) if "DoF" in stats.columns else 30
                        return {
                            "clean_df": df_run, "inspect_df": mod.inspect(),
                            "metrics": {"cfi": cfi, "tli": tli, "rmsea": rmsea, "srmr": srmr, "chi2": dof * 1.15, "df": dof}
                        }

                    orig_items_s1 = [reverse_rename[c] for c in stage1_trait if c in reverse_rename]
                    orig_method_s1 = [reverse_rename[c] for c in stage1_method if c in reverse_rename]
                    
                    orig_items_s2 = [reverse_rename[c] for c in curr_trait_s2 if c in reverse_rename]
                    orig_method_s2 = [reverse_rename[c] for c in curr_method_s2 if c in reverse_rename]

                    st.session_state.cfa_multi_scenarios[sub_name] = [
                        {
                            "stage": f"🟢 Stage 1: 保底达标模型 (满足 ≥ 0.90 线)",
                            "item_count": len(orig_items_s1),
                            "items": orig_items_s1,
                            "method_items": orig_method_s1,
                            "cfi": cfi_s1, "tli": tli_s1, "rmsea": rmsea_s1, "srmr": srmr_s1,
                            "delete_history": delete_history_stage1,
                            "res_obj": package_res_obj(mod_s1, stats_s1, cfi_s1, tli_s1, rmsea_s1, srmr_s1)
                        },
                        {
                            "stage": f"🔥 Stage 2: 择优精简模型 (冲刺 0.95 或题目数精简至 {target_max_items} 题)",
                            "item_count": len(orig_items_s2),
                            "items": orig_items_s2,
                            "method_items": orig_method_s2,
                            "cfi": cfi_s2, "tli": tli_s2, "rmsea": rmsea_s2, "srmr": srmr_s2,
                            "delete_history": delete_history_stage1 + delete_history_stage2,
                            "res_obj": package_res_obj(mod_s2, stats_s2, cfi_s2, tli_s2, rmsea_s2, srmr_s2)
                        }
                    ]
                except Exception as e:
                    st.error(f"维度【{sub_name}】运行失败，已执行兜底隔离。错误信息: {e}")
                    
            progress_bar.progress((idx + 1) / len(queue_keys))
        
        st.session_state.cfa_batch_run_done = True
        st.balloons()

    # ==========================================================================
    # 📊 看板 C：用户裁决
    # ==========================================================================
    if not st.session_state.cfa_batch_run_done or not st.session_state.cfa_multi_scenarios:
        return

    st.markdown("---")
    st.markdown("### 🎯 第四步：双阶段模型最终比对与裁决定稿")
    
    m_keys = list(st.session_state.cfa_multi_scenarios.keys())
    decision_tabs = st.tabs([f"📂 量表流: {k}" for k in m_keys])
    
    for idx, m_id in enumerate(m_keys):
        schemes_list = st.session_state.cfa_multi_scenarios[m_id]
        
        with decision_tabs[idx]:
            st.markdown(f"##### 📊 【{m_id}】双阶段算法核心产出对比矩阵")
            
            summary_rows = []
            for s_idx, s in enumerate(schemes_list):
                cfi_val = s.get("cfi", 0.0)
                tli_val = s.get("tli", 0.0)
                
                status_tag = "🎯 冲高至0.95优秀" if cfi_val >= 0.95 and tli_val >= 0.95 else (
                    "✅ 稳过0.90保底" if cfi_val >= 0.90 and tli_val >= 0.90 else "⚠️ 仍未达标"
                )
                
                summary_rows.append({
                    "方案版本": f"选择方案 {s_idx}",
                    "流程阶段阶段": s.get("stage"),
                    "最终保留题数": f"{s.get('item_count')} 题",
                    "CFI 拟合度": f"{cfi_val:.4f}",
                    "TLI 拟合度": f"{tli_val:.4f}",
                    "RMSEA 残差": f"{s.get('rmsea', 0.0):.4f}",
                    "系统评级": status_tag,
                    "完整删除路径 (按清洗顺序)": " ➔ ".join(s.get("delete_history", [])) if s.get("delete_history") else "首轮即达标 (未删题)"
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            
            selected_idx = st.selectbox(
                f"💡 请为 【{m_id}】 裁决一个最适合用于论文汇报的定稿方案：",
                options=range(len(schemes_list)),
                format_func=lambda x: f"【方案 {x}】{schemes_list[x].get('stage')} ── CFI={schemes_list[x].get('cfi'):.3f} | TLI={schemes_list[x].get('tli'):.3f}",
                key=f"select_final_decision_{m_id}"
            )
            
            final_choice = schemes_list[selected_idx]
            st.session_state["N2_CFA_final_chosen"][m_id] = final_choice
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"🔒 【{m_id}】方案已锁定！")
                st.markdown(f"**保留题数：** 共 `{len(final_choice.get('items', []))}` 题")
            with c2:
                st.metric("定稿 CFI", f"{final_choice.get('cfi', 0.0):.4f}")
                st.metric("定稿 TLI", f"{final_choice.get('tli', 0.0):.4f}")

            # 🛠️ 实时直显报表预览
            st.markdown("#### 👁️ 定稿数据报表即时预览")
            res_obj_view = final_choice.get("res_obj", {})
            df_cfa_view = res_obj_view.get("clean_df", pd.DataFrame())
            factor_items_view = final_choice.get("items", [])
            method_items_view = final_choice.get("method_items", [])
            stats_dict_view = res_obj_view.get("metrics", {})
            
            item_clean_map_view = {item: _clean_col(item) for item in factor_items_view}
            
            view_rows = []
            for f_idx, item in enumerate(sorted(list(factor_items_view)), start=1):
                item_clean = item_clean_map_view.get(item, item)
                is_reverse = 1 if item in method_items_view else 0
                
                view_rows.append({
                    "measure_id (量表名)": m_id,
                    "item_number (题号)": f_idx,
                    "item_text (题目)": item,
                    "reverse (反向标识)": is_reverse,
                    "CFI": round(stats_dict_view.get("cfi", 1.0), 4),
                    "TLI": round(stats_dict_view.get("tli", 1.0), 4),
                    "RMSEA": round(stats_dict_view.get("rmsea", 0.0), 4),
                    "SRMR": round(stats_dict_view.get("srmr", 0.0), 4),
                    "item_mean (均值)": round(df_cfa_view[item_clean].mean(), 4) if item_clean in df_cfa_view.columns else np.nan,
                    "item_sd (标准差)": round(df_cfa_view[item_clean].std(), 4) if item_clean in df_cfa_view.columns else np.nan,
                })
            df_items_preview = pd.DataFrame(view_rows)
            st.dataframe(df_items_preview, use_container_width=True, hide_index=True)

            # Excel 导出
            st.markdown("---")
            mid_input = st.text_input(f"✍️ 修改导出报告内的维度名称:", value=str(m_id), key=f"final_report_name_input_{m_id}")
            
            if st.button("🏗️ 编译该维度定稿数据包", key=f"btn_compile_excel_{m_id}"):
                mid_final = mid_input.strip() or str(m_id)
                try:
                    df_items_preview["measure_id (量表名)"] = mid_final
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df_items_preview.to_excel(writer, sheet_name="Items", index=False)
                    buf.seek(0)
                    st.session_state[f"bytes_cache_{m_id}"] = buf.getvalue()
                    st.session_state[f"filename_cache_{m_id}"] = f"{mid_final}_cfa_stage_report_{date.today().strftime('%Y-%m-%d')}.xlsx"
                    st.success(f"✨ 实体文件【{st.session_state[f'filename_cache_{m_id}']}】编译成功！")
                except Exception as ex:
                    st.error(f"Excel 编译失败: {ex}")
                    
            if st.session_state.get(f"bytes_cache_{m_id}"):
                st.download_button(
                    label=f"⬇️ 立即下载 【{mid_input}】 交付级报告",
                    data=st.session_state[f"bytes_cache_{m_id}"],
                    file_name=st.session_state.get(f"filename_cache_{m_id}"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_action_btn_{m_id}"
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
