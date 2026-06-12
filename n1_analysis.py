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
    # 获取数值列并进行全面清理
    items = df.select_dtypes(include=[np.number])

    # 移除NaN和Inf值
    items = items.replace([np.inf, -np.inf], np.nan).dropna()
    items = items[~items.isin([np.inf, -np.inf]).any(axis=1)]

    if items.empty:
        return pd.DataFrame(), 1

    # 再次确认没有NaN或Inf值
    if items.isnull().any().any() or np.isinf(items.values).any():
        # 如果仍有问题，进行最后的清理
        items = items.dropna()
        items = items.replace([np.inf, -np.inf], np.nan).dropna()
        items = items[~items.isin([np.inf, -np.inf]).any(axis=1)]

    if items.empty:
        return pd.DataFrame(), 1

    try:
        Z = StandardScaler().fit_transform(items.values)  # 使用.values确保是numpy数组

        # 最后检查标准化数据
        if np.isnan(Z).any() or np.isinf(Z).any():
            st.error("标准化后数据仍包含NaN或Inf值，请检查原始数据质量")
            return pd.DataFrame(), 1

    except Exception as e:
        st.error(f"数据标准化失败: {e}")
        return pd.DataFrame(), 1

    try:
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
    factor_num_list = []
    progress_text = "正在进行 PCA Bootstrapping..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(boot_time):
        idx = np.random.choice(df.index, size=int(len(df)*0.8), replace=False)
        temp = df.loc[idx].copy()
        
        if temp.std().sum() == 0:
            continue
            
        _, factor_num = pca_algo(temp, graph=False)
        factor_num_list.append(factor_num)
        my_bar.progress((i + 1) / boot_time, text=f"{progress_text} ({i+1}/{boot_time})")
    
    my_bar.empty()
    if not factor_num_list:
        return 1
    factor_num_final = dscpt_stats_mode(factor_num_list)
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
    # 获取数值列并移除缺失值
    X = df.select_dtypes(include=[np.number]).dropna(axis=0, how='any')

    # 额外检查：移除包含无穷大值的行
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]

    if X.empty:
        raise RuntimeError("No valid numeric data available for EFA analysis")

    if scaler is None:
        scaler = StandardScaler()

    try:
        Z = scaler.fit_transform(X)
    except Exception as e:
        raise RuntimeError(f"Data standardization failed: {e}")

    # 检查标准化后的数据是否包含NaN或Inf
    if np.isnan(Z).any() or np.isinf(Z).any():
        raise RuntimeError("Standardized data contains NaN or Inf values")

    fa = FactorAnalyzer(n_factors=k, rotation='varimax', method='minres')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(Z)
    except Exception as e:
        raise RuntimeError(f"EFA failed: {e}")

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
    loadings_list = bootstrap_efa(current_df, factor_num_final, boot_time=50)
    if len(loadings_list) == 0:
        raise RuntimeError("Bootstrap EFA failed for all samples.")
    loadings_cat = pd.concat(loadings_list)
    loadings_avg = loadings_cat.groupby(level=0).mean()
    loadings_avg = loadings_avg.loc[loadings_list[0].index, loadings_list[0].columns]
    return loadings_avg

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
      3) 共同度（communality）低于 min_communality

    逻辑：
      - 首先根据主载荷 / 交叉载荷 / 共同度进行 item 分类（mutually exclusive，按优先级：主载荷 -> 交叉 -> 共同度）
      - 对每类候选按严重程度排序（主载荷和共同度越低越优先；交叉按 次/主 比值从高到低）
      - 删除时保证不会将某个因子的题目数降到 min_items_per_factor 以下
    """
    primary_assign = {}
    candidates_low = []    # (item, p_idx, primary_loading)
    candidates_cross = []  # (item, p_idx, ratio, second_loading)
    candidates_comm = []   # (item, p_idx, communality)

    if whitelist is None:
        whitelist = []
        
    # 共同度由平均载荷矩阵计算：每题的 communality = sum(loadings^2)
    communalities = (loadings_avg ** 2).sum(axis=1)

    for item, row in loadings_avg.iterrows():
        # 【关键逻辑】如果题目在白名单里，直接跳过检查，绝不删除
        if item in whitelist:
            continue 
        
        p_idx, p, s = _primary_factor_and_cross(row)
        primary_assign[item] = p_idx

        # 优先判定：主载荷过低
        if p < min_primary_loading:
            candidates_low.append((item, p_idx, p))
        # 再判定：强交叉载荷
        elif s > min_cross_loading and (p > 0) and (s / p) > cross_ratio:
            candidates_cross.append((item, p_idx, (s / p), s))
        # 再判定：共同度过低
        elif communalities.loc[item] < min_communality:
            candidates_comm.append((item, p_idx, float(communalities.loc[item])))

    # 如果没有任何候选，直接返回
    if not candidates_low and not candidates_cross and not candidates_comm:
        return None

    # 统计每因子当前题目数
    counts = np.zeros(k, dtype=int)
    for item, p_idx in primary_assign.items():
        counts[p_idx] += 1

    # 排序：主载荷越低越先删除；共同度越低越先删除；交叉按 ratio 从高到低
    candidates_low.sort(key=lambda t: t[2])            # ascending primary loading
    candidates_comm.sort(key=lambda t: t[2])           # ascending communality (lower worse)
    candidates_cross.sort(key=lambda t: t[2], reverse=True)  # descending ratio

    # 合并优先级：低主载荷 -> 低共同度 -> 交叉载荷
    merged = [('low',) + c for c in candidates_low] + \
             [('comm',) + c for c in candidates_comm] + \
             [('cross',) + c for c in candidates_cross]

    # 按合并后的优先级依次尝试删除第一个合适的题目
    for tag, item, p_idx, metric, *rest in merged:
        # 不允许把某个因子题目数降到 <= min_items_per_factor
        if counts[p_idx] <= min_items_per_factor:
            continue
        
        msg = ""
        if tag == 'low':
            msg = f"删除题目 **{item}**：因子载荷过低 (主载荷={metric:.3f} < {min_primary_loading})"
        elif tag == 'cross':
            second_loading = rest[0] if rest else float('nan')
            msg = f"删除题目 **{item}**：强交叉载荷 (次/主比={metric:.3f} > {cross_ratio}，次载荷≈{second_loading:.3f})"
        elif tag == 'comm':
            msg = f"删除题目 **{item}**：共同度过低 (Communality={metric:.3f} < {min_communality})"
        
        st.write(f"🛑 {msg}") 
        return item

    # 若未在上面找到合适且满足因子保护（min_items_per_factor）的题目，兜底删除最严重的一个
    if merged:
        _tag, item, p_idx, _metric, *_rest = merged[0]
        msg = ""
        if _tag == 'low':
            msg = f"删除题目 **{item}**：因子载荷过低 (主载荷={_metric:.3f}) [兜底删除]"
        elif _tag == 'cross':
            msg = f"删除题目 **{item}**：强交叉载荷 (次/主比={_metric:.3f}) [兜底删除]"
        elif _tag == 'comm':
            msg = f"删除题目 **{item}**：共同度过低 (Communality={_metric:.3f}) [兜底删除]"
        
        st.write(f"🛑 {msg}") 
        return item
    return None

def run_pipeline_streamlit(df, fixed_factors=None, max_iterations=100, whitelist=None):
    current_df = df.select_dtypes(include=[np.number]).copy()
    
    # 1. Determine Factor Number
    factor_num_final = 1
    
    if fixed_factors is not None:
        factor_num_final = int(fixed_factors)
        st.info(f"ℹ️ 使用用户手动指定的因子数量: **{factor_num_final}**")
    else:
        with st.spinner("正在通过 Bootstrap PCA 评估最佳因子数量..."):
            factor_num_final = bootstrap_pca(current_df, boot_time=20)
        st.success(f"✅ Bootstrap PCA 建议的因子数: **{factor_num_final}**")

    # 2. First Average Loadings
    with st.spinner(f"正在基于 {factor_num_final} 个因子进行初始 Bootstrap EFA 计算..."):
        loadings_table_avg = calculate_loadings_avg(current_df, factor_num_final)

    # 3. Iterative Deletion
    seen = set()
    iteration = 0
    deleted_items = []

    st.markdown("### 🔄 开始迭代删题流程")
    status_container = st.empty() 
    
    item_to_delete = delete_items(loadings_table_avg, current_df, factor_num_final, whitelist=whitelist)

    while item_to_delete is not None and iteration < max_iterations:
        if item_to_delete in seen:
            st.warning(f"⚠️ 检测到重复建议删除 {item_to_delete}，提前停止以避免震荡。")
            break

        # 剩余题目 <= 3 时停止
        if current_df.shape[1] <= 3:
            st.warning("⚠️ 剩余题目数量已降至 3 题，为了保证模型可识别性，停止继续删题。")
            # 这里的 item_to_delete 虽然被计算出来了，选择不执行删除
            break
        # =========================================================
        
        seen.add(item_to_delete)

        current_df = current_df.drop(columns=[item_to_delete])
        deleted_items.append(item_to_delete)

        status_container.info(f"正在进行第 {iteration + 1} 轮迭代计算 (已删除 {len(deleted_items)} 题)...")
        
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

def render_n1_analysis():
    st.title("模块 2: N1数据分析 ")

    # --- 1. 数据来源 ---
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

    df_analysis = None

    if data_source == "💾 来自 Data Cleaning（四数据集）":
        from .data_cleaning_dual import get_dual_mode_analysis_df
        dataset_names = ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]
        selected_dataset = st.selectbox("1. 选择数据集", dataset_names, key="n1_dual_dataset")
        measure_names = list(st.session_state.dc_measures.keys())
        if not measure_names:
            st.warning("请在数据清洗模块的「Measure 划分」中至少定义一个 Measure。")
        else:
            selected_measures = st.multiselect(
                "2. 选择 Measure（可多选）",
                measure_names,
                default=[measure_names[0]] if measure_names else [],
                key="n1_dual_measures",
            )
            if selected_measures:
                df_analysis = get_dual_mode_analysis_df(
                    selected_dataset,
                    selected_measures,
                    st.session_state.dc_dataset_full,
                    st.session_state.dc_measures,
                    item_columns_only=True,
                )
                st.session_state.n1_selected_measures = list(selected_measures)
            if df_analysis is not None:
                st.info(f"使用 **{selected_dataset}**，Measure: **{', '.join(selected_measures)}**（{df_analysis.shape[0]} 行 × {df_analysis.shape[1]} 列，仅题目列）")

    elif data_source == "💾 来自 Data Cleaning（子数据集）":
        dataset_names = list(st.session_state.sub_datasets.keys())
        selected_name = st.selectbox("请选择已保存的子数据集:", dataset_names)
        if selected_name:
            df_analysis = st.session_state.sub_datasets[selected_name]

            update_info = ""
            if ('sub_datasets_updated' in st.session_state and
                selected_name in st.session_state.sub_datasets_updated):
                update_time = st.session_state.sub_datasets_updated[selected_name]
                update_info = f" (最后更新: {update_time.strftime('%H:%M:%S')})"

            st.info(f"正在使用缓存数据集: **{selected_name}** ({df_analysis.shape[0]} 行, {df_analysis.shape[1]} 列){update_info}")
            st.success("✅ 数据集包含最新的序号重命名结果")

    else:
        uploaded_file = st.file_uploader("请上传用于分析的数据文件", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df_upload = pd.read_excel(uploaded_file)
                else:
                    df_upload = pd.read_csv(uploaded_file)
                st.write("文件预览 (前5行):")
                st.dataframe(df_upload.head())
                # <Function 1>：用户手动选择题目
                # =========================================================
                st.info("👇 请从上传的文件中，勾选需要进行 EFA 分析的【量表题目】（请勿勾选ID、姓名等无关列）")
                
                all_cols = df_upload.columns.tolist()
                # 默认全选太乱，我们尝试简单的智能预选（选出数值类型的列），如果没有数值列则全选
                numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
                default_cols = numeric_cols if numeric_cols else all_cols
                
                selected_cols = st.multiselect(
                    "请选择要分析的题目 (至少 3 个):",
                    options=all_cols,
                    default=default_cols
                )
                
                if len(selected_cols) < 3:
                    st.warning("⚠️ 请至少选择 3 个题目才能进行分析。")
                    df_analysis = None
                else:
                    # 只保留用户选中的列
                    df_analysis = df_upload[selected_cols].copy()
                    st.success(f"已选择 {len(selected_cols)} 个题目，共 {df_analysis.shape[0]} 行。")
                # =========================================================
            
            except Exception as e:
                st.error(f"读取文件失败: {e}")

    st.markdown("---")

    # ⚠️ 关键检查：没有数据就 Return
    if df_analysis is None:
        if data_source == "📤 上传新文件 (Excel/CSV)":
            st.info("👈 请先上传文件或选择数据来源。")
        elif data_source == "💾 来自 Data Cleaning（四数据集）":
            st.warning("请至少选择一个 Measure，或先在数据清洗中定义 Measure。")
        else:
            st.warning("👈 请先在上方选择数据来源。")
        return
    
    # Altair 画图库不支持列名中包含英文冒号 ":"，因此把“：”改成“_”
    df_analysis.columns = [str(col).replace(":", "_") for col in df_analysis.columns]
    
    # 数据预处理（三步法）：
    # 第一步：逐列判断内容能否转为数字，识别出数字列
    # 第二步：将数字列统一转换为数值类型
    # 第三步：只用数字列进行 EFA 分析，文本列自动排除
    numeric_cols = []
    for col in df_analysis.columns:
        converted = pd.to_numeric(df_analysis[col], errors='coerce')
        non_null_original = df_analysis[col].notna().sum()
        if non_null_original > 0 and converted.notna().sum() / non_null_original >= 0.5:
            numeric_cols.append(col)
    df_numeric = df_analysis[numeric_cols].apply(pd.to_numeric, errors='coerce').copy()
    original_len = len(df_numeric)

    # 移除包含NaN的行
    df_numeric = df_numeric.dropna()

    # 移除包含无穷大值(Inf)的行
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()

    # 再次检查是否有任何无穷大值（以防万一）
    df_numeric = df_numeric[~df_numeric.isin([np.inf, -np.inf]).any(axis=1)]

    cleaned_len = len(df_numeric)

    removed_rows = original_len - cleaned_len
    if removed_rows > 0:
        st.caption(f"⚠️ 已自动移除含有非数值、缺失值或无穷大值的行。分析样本量: {original_len} -> {cleaned_len} (移除了 {removed_rows} 行)")

    if cleaned_len < 10: # 再次兜底检查
        st.error(f"❌ 有效样本量不足 ({cleaned_len} 行)，无法分析。请检查是否误选了包含大量文本的列。")
        return
        
    if df_numeric.shape[1] < 2:
        st.error("数据列数太少 (<2)，无法进行因子分析。请检查数据是否正确。")
        return

    
    # --- Feature 1: Bar Plots ---
    st.subheader("1. 题目分布可视化 (Bar Plots)")
    with st.expander("点击展开/折叠题目分布图", expanded=True):
        cols_to_plot = df_numeric.columns.tolist()
        num_cols = 2
        rows = [st.columns(num_cols) for _ in range((len(cols_to_plot) + num_cols - 1) // num_cols)]
        for i, col_name in enumerate(cols_to_plot):
            row_idx = i // num_cols
            col_idx = i % num_cols
            with rows[row_idx][col_idx]:
                st.markdown(f"**{col_name}**")
                counts = df_numeric[col_name].value_counts().sort_index()
                chart_df = pd.DataFrame({"频数": counts.values}, index=counts.index.astype(str))
                chart_df.index.name = None
                st.bar_chart(chart_df)

    st.markdown("---")

    # --- Feature 2: Bootstrap EFA Pipeline ---
    st.subheader("2. Bootstrap EFA 分析")
    st.info("此过程将执行 Bootstrapping EFA，并进行迭代删题。")
    
    st.markdown("##### ⚙️ 参数设置")
    factor_method = st.radio(
        "请选择因子数量 (K) 的确定方式:",
        ["🤖 自动计算 (Bootstrap PCA)", "👆 手动指定"],
        horizontal=True
    )
    
    manual_k_val = None
    if factor_method == "👆 手动指定":
        max_k = max(1, df_numeric.shape[1])
        manual_k_val = st.number_input(
            "请输入您期望提取的因子数量:", 
            min_value=1, 
            max_value=max_k, 
            value=2, 
            step=1
        )

    st.markdown("##### 选择参与本次分析的题目")
    st.caption("勾选要纳入本次 Bootstrap EFA 的题目；未勾选的题目将不参与本次运行。")
    n1_items_to_run = smart_multiselect(
        options=df_numeric.columns.tolist(),
        label="选择题目（至少 3 个）",
        key_suffix="n1_pre_run_items",
        default_selected=df_numeric.columns.tolist(),
        show_selection_controls=True,
    )
    df_for_run = df_numeric[[c for c in n1_items_to_run if c in df_numeric.columns]] if n1_items_to_run else df_numeric
    if len(df_for_run.columns) < 3:
        st.warning("⚠️ 请至少选择 3 个题目再运行。")
    
    run_efa = st.button("开始运行 Bootstrap EFA", type="primary")

    if run_efa:
        if len(df_for_run.columns) < 3:
            st.error("至少需选择 3 个题目才能运行。")
        else:
            if 'n1_result_df' in st.session_state:
                del st.session_state['n1_result_df']
            
            with st.container():
                final_df, final_loadings, kept, deleted, n_factors = run_pipeline_streamlit(
                    df_for_run,
                    fixed_factors=manual_k_val, whitelist=None
                )
                
                st.session_state.n1_result_df = final_df
                st.session_state.n1_loadings = final_loadings
                st.session_state.n1_kept = kept
                st.session_state.n1_deleted = deleted
                st.session_state.n1_factors = n_factors
                st.session_state.n1_df_for_run_columns = list(df_for_run.columns)
                
                st.success("🎉 EFA 分析完成！")

    # --- Feature 3: Post-Analysis Checks ---
    if 'n1_result_df' in st.session_state:
        st.markdown("---")
        st.subheader("3. 最终模型检验与报告")
        
        # 获取最终用于分析的数据
        df_final = st.session_state.n1_result_df
        loadings = st.session_state.n1_loadings
        
        # 3.0 展示保留/删除题目概况
        with st.expander("查看题目保留/删除详情", expanded=False):
            c1, c2 = st.columns(2)
            c1.write(f"**保留的题目 ({len(st.session_state.n1_kept)})**")
            c1.write(st.session_state.n1_kept)
            c2.write(f"**删除的题目 ({len(st.session_state.n1_deleted)})**")
            c2.write(st.session_state.n1_deleted)

        # =========================================================
        # 1) KMO & Bartlett’s test (依据 statsCriteriaCheck.py)
        # =========================================================
        st.markdown("#### 1️⃣ KMO & Bartlett’s Test")
        
        try:
            # 确保无缺失值
            df_clean = df_final.dropna()
            kmo_all, kmo_model = calculate_kmo(df_clean)
            chi_square_value, p_value = calculate_bartlett_sphericity(df_clean)
            
            summary_df = pd.DataFrame({
                "Statistic": ["Kaiser-Meyer-Olkin (KMO > 0.6)", "Bartlett’s chi-square", "Bartlett’s p-value"],
                "Value": [f"{kmo_model:.4f}", f"{chi_square_value:.4f}", f"{p_value:.4f}"]
            })
            
            # 使用 table 展示更整洁
            st.table(summary_df)
            
            if kmo_model < 0.6:
                st.warning("⚠️ KMO 值较低 (< 0.6)，数据可能不太适合进行因子分析。")
            if p_value > 0.05:
                st.warning("⚠️ Bartlett 球形检验未显著 (p > 0.05)，变量间可能缺乏相关性。")

        except Exception as e:
            st.error(f"计算 KMO/Bartlett 出错: {e}")

        # =========================================================
        # 2) Internal Consistency (Cronbach's Alpha)
        # =========================================================
        st.markdown("#### 2️⃣ 信度检验 (Internal Consistency)")
        
        try:
            current_alpha = cronbach_alpha(df_final)
            st.markdown(f"**👉 所有题项的 Cronbach's α = `{current_alpha:.4f}`**")
            
            if current_alpha < 0.6:
                st.error("❌ 信度不可接受 (< 0.6)")
            elif current_alpha < 0.7:
                st.warning("⚠️ 信度一般 (0.6 - 0.7)")
            else:
                st.success("✅ 信度良好 (> 0.7)")

            st.markdown("**📉 删除特定题项后的 α 变化 (Alpha if item deleted):**")
            st.caption("如果删除某题后 α 值显著升高，说明该题可能降低了量表的内部一致性。")
            
            removal_df = alpha_after_removal(df_final)

            # 按照题目序号排序
            removal_df_sorted = sort_dataframe_by_item_names(removal_df, item_column='删除的题项')

            # 使用 Pandas Styler 进行渐变色背景显示
            st.dataframe(
                removal_df_sorted.style
                .format({"Cronbach's α": "{:.4f}"})
                .background_gradient(cmap='RdYlGn', subset=["Cronbach's α"])
            )

        except Exception as e:
            st.error(f"计算信度时出错: {e}")

        # =========================================================
        # 3) Item Analysis & Correlations (New Feature)
        # =========================================================
        st.markdown("#### 3️⃣ 题目关联性分析 (Item Relationships)")
        
        t1, t2 = st.tabs(["Item-Total Correlation", "Pairwise Correlation Matrix"])
        
        with t1:
            st.caption("校正项总相关 (Corrected Item-Total Correlation): 单个题目与剩余题目总分的相关性。通常建议 > 0.3。")
            try:
                itc_df = calculate_item_total_correlation(df_final)
                # 按照题目序号排序
                itc_df_sorted = sort_dataframe_by_item_names(itc_df, item_column='Item')
                st.dataframe(
                    itc_df_sorted.style
                    .format("{:.4f}")
                    .background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                )
            except Exception as e:
                st.error(f"Error calculating Item-Total Corr: {e}")
                
        with t2:
            st.caption("题目间的两两相关系数矩阵 (Pearson Correlation)。")
            with st.expander("点击展开相关系数矩阵", expanded=False):
                corr_matrix = df_final.corr()
                # 按照题目序号排序行和列
                sorted_columns = sort_items_by_number(corr_matrix.columns.tolist())
                corr_matrix_sorted = corr_matrix.loc[sorted_columns, sorted_columns]
                st.dataframe(
                    corr_matrix_sorted.style
                    .format("{:.2f}")
                    .background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
                )



        # =========================================================
        # 4) Model Fit & Residuals (New Feature + Communalities)
        # =========================================================
        st.markdown("#### 4️⃣ 模型拟合与残差 (Model Fit & Residuals)")
        
        # --- Communalities (Moved here) ---
        st.write("**共同度 (Communalities)**")
        try:
            communalities = (loadings ** 2).sum(axis=1)
            comm_df = pd.DataFrame({'Communality': communalities})
            # 按照题目序号排序
            comm_df_sorted = sort_dataframe_by_item_names(comm_df)
            st.dataframe(comm_df_sorted.style.format('{:.4f}').background_gradient(cmap="Blues"))
        except Exception as e:
            st.error(f"Error calculating communalities: {e}")

        # --- Residual Normality (New) ---
        st.write("**残差正态性检验 (Residual Normality)**")
        st.caption("检验观察到的相关矩阵与模型重构的相关矩阵之间的差异（残差）是否符合正态分布。")
        
        try:
            res_matrix, res_values, shapiro_stat, shapiro_p = check_residual_normality(df_final, loadings)
            
            c1, c2 = st.columns(2)
            c1.metric("Shapiro-Wilk Statistic", f"{shapiro_stat:.4f}")
            c2.metric("P-Value", f"{shapiro_p:.4f}")
            
            if shapiro_p > 0.05:
                st.success("✅ 残差服从正态分布 (p > 0.05)，模型拟合良好。")
            else:
                st.warning("⚠️ 残差不服从正态分布 (p < 0.05)，模型拟合可能存在偏差。")
            
            with st.expander("查看残差直方图"):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(res_values, bins=15, edgecolor='black', alpha=0.7)
                ax.set_title("Histogram of Residuals (Off-diagonal)")
                ax.set_xlabel("Residual Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error checking residuals: {e}")

        '''
        
        # =========================================================
        # 3) Communalities (共同度)--原来的code
        st.markdown("#### 3️⃣ 共同度 (Communalities)")
        st.caption("表示每个题目被提取出的因子所解释的方差比例。")
        
        try:
            # 计算 communalities = loadings的平方和
            communalities = (loadings ** 2).sum(axis=1)
            
            comm_df = pd.DataFrame({
                'Item': loadings.index,
                'Communality': communalities
            })
            
            st.dataframe(
                comm_df.style
                .format({'Communality': '{:.4f}'})
                .background_gradient(cmap="Blues")
            )
        except Exception as e:
            st.error(f"计算共同度出错: {e}")
        '''
       

        # 🚀 结果微调：恢复被删题目
        if len(st.session_state.n1_deleted) > 0:
            st.markdown("---")
            st.markdown("#### 🛠️ 结果微调：恢复被删题目")
            st.caption("如果你认为某些被删的题目具有重要的理论意义，可以将它们选中并强制保留。")
            
            items_to_restore = st.multiselect(
                "请选择要【强制保留】的题目:",
                options=st.session_state.n1_deleted
            )
            
            if st.button("🔄 恢复选中题目并重新运行 EFA", type="secondary"):
                if items_to_restore:
                    with st.spinner(f"正在强制保留 {len(items_to_restore)} 个题目并重新运行..."):
                        if 'n1_result_df' in st.session_state:
                            del st.session_state['n1_result_df']
                            
                        current_k = st.session_state.n1_factors
                        run_cols = st.session_state.get("n1_df_for_run_columns", df_numeric.columns.tolist())
                        run_cols = [c for c in run_cols if c in df_numeric.columns]
                        df_input = df_numeric[run_cols] if run_cols else df_numeric

                        final_df, final_loadings, kept, deleted, n_factors = run_pipeline_streamlit(
                            df_input,
                            fixed_factors=current_k,
                            whitelist=items_to_restore
                        )
                        
                        st.session_state.n1_result_df = final_df
                        st.session_state.n1_loadings = final_loadings
                        st.session_state.n1_kept = kept
                        st.session_state.n1_deleted = deleted
                        st.session_state.n1_factors = n_factors
                        
                        st.rerun()
                else:
                    st.warning("请至少选择一个要恢复的题目。")

        # =========================================================
        # 5) 最终载荷矩阵 (含排序、着色、导出)
        # =========================================================
        loadings = st.session_state.n1_loadings
        st.markdown("#### 5️⃣ 最终因子载荷矩阵 (Factor Loadings)")
        
        # 1. 排序: 按照题目序号从小到大排序
        loadings_sorted = sort_dataframe_by_item_names(loadings)

        # 2. 颜色编码函数
        def color_loadings(val):
            """
            大于 0.4 或 小于 -0.4 显示蓝色加粗，否则无色
            """
            try:
                is_strong = abs(float(val)) > 0.40
            except (TypeError, ValueError):
                return ''
            color = 'blue' if is_strong else 'black'
            weight = 'bold' if is_strong else 'normal'
            return f'color: {color}; font-weight: {weight}'

        # 3. 展示 Style 后的表格
        st.dataframe(
            loadings_sorted.style
            .format("{:.3f}")
            .map(color_loadings)
        )

        csv = loadings_sorted.to_csv().encode('utf-8-sig')
        st.download_button(
            label="📥 下载最终载荷矩阵 (CSV)",
            data=csv,
            file_name='final_factor_loadings.csv',
            mime='text/csv',
        )

        # =========================================================
        # 生成可下载 Excel 报告表（每 measure 一 sheet，按题目序号排序）
        # =========================================================
        st.markdown("---")
        st.markdown("#### 📥 生成可下载 Excel 报告表")
        st.caption("根据当前 EFA 结果生成按题目排列的报告表，可填写量表编号(measure_id)后下载。")
        kept = st.session_state.n1_kept
        df_final = st.session_state.n1_result_df
        dc_measures = st.session_state.get("dc_measures") or {}
        selected_measures = st.session_state.get("n1_selected_measures") or []

        # 确定要生成报告的“量表”及其题目：四数据集下按 measure 分，否则整表为一个量表
        measure_item_list = []
        if dc_measures and selected_measures:
            for m in selected_measures:
                items_m = [c for c in (dc_measures.get(m) or []) if c in kept]
                if items_m:
                    measure_item_list.append((m, items_m))
        if not measure_item_list:
            measure_item_list = [("当前量表", list(kept))]

        with st.form("n1_excel_report_form"):
            measure_ids = {}
            for m, items_m in measure_item_list:
                measure_ids[m] = st.text_input(
                    f"量表「{m}」的 measure_id（唯一编码）",
                    value="",
                    key=f"n1_measure_id_{m}",
                    help="用于报告中标识该量表的唯一编号，如问卷缩写。"
                )
            submitted = st.form_submit_button("生成并下载 Excel 报告表")
            if submitted:
                missing = [m for m, _ in measure_item_list if not (measure_ids.get(m) or "").strip()]
                if missing:
                    st.error(f"请为以下量表填写 measure_id：{', '.join(missing)}")
                    st.session_state.pop("n1_excel_report_bytes", None)
                else:
                    try:
                        kmo_all, kmo_model = calculate_kmo(df_final)
                        chi_square_value, p_value = calculate_bartlett_sphericity(df_final)
                        alpha_removal_df = alpha_after_removal(df_final)
                        itc_df = calculate_item_total_correlation(df_final)
                        communalities = (loadings ** 2).sum(axis=1)
                        _, _, shapiro_stat, shapiro_p = check_residual_normality(df_final, loadings)
                        alpha_removal_by_item = alpha_removal_df.set_index("删除的题项")["Cronbach's α"]

                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                            for m, items_m in measure_item_list:
                                mid = (measure_ids.get(m) or m).strip()
                                sorted_items = sort_item_cols_by_number(items_m)
                                rows = []

                                def _item_number_from_item_text(item_text_val):
                                    """
                                    item_number 直接取自 item_text 的前缀序号：
                                    - 先取 item_text 在第一个下划线前的前缀
                                    - 再从该前缀提取数字序号
                                    """
                                    s = str(item_text_val).strip()
                                    prefix = s.split("_", 1)[0]
                                    m = re.search(r"(\d+)", prefix)
                                    return int(m.group(1)) if m else ""

                                for item in sorted_items:
                                    pre, num, text = parse_item_col(item)
                                    item_txt = text or item
                                    item_num = _item_number_from_item_text(item_txt)
                                    rev = 1 if (isinstance(item, str) and item.rstrip().endswith("r")) else 0
                                    load_row = loadings.loc[item] if item in loadings.index else pd.Series(dtype=float)
                                    alpha_rem = alpha_removal_by_item.get(item, np.nan)
                                    itc_val = itc_df.loc[item, "Item-Total Corr"] if item in itc_df.index else np.nan
                                    comm = communalities.get(item, np.nan)
                                    row = {
                                        "measure_id": mid,
                                        "item_number": item_num,
                                        "item_text": item_txt,
                                        "reverse": rev,
                                    }
                                    for c in loadings.columns:
                                        row[c] = load_row.get(c, np.nan)
                                    row["KMO"] = kmo_model
                                    row["Bartlett_chi2"] = chi_square_value
                                    row["Bartlett_p"] = p_value
                                    row["alpha_after_removal"] = alpha_rem
                                    row["item_total_correlation"] = itc_val
                                    row["communality"] = comm
                                    row["residual_Shapiro_W_stat"] = shapiro_stat
                                    row["residual_Shapiro_W_p"] = shapiro_p
                                    rows.append(row)
                                sheet_df = pd.DataFrame(rows)
                                sheet_name = (mid[:31]) if len(mid) > 31 else mid or "Sheet"
                                sheet_name = "".join(c for c in sheet_name if c not in '[]:*?/\\')
                                sheet_df.to_excel(w, sheet_name=sheet_name or "Sheet", index=False)
                        buf.seek(0)
                        st.session_state.n1_excel_report_bytes = buf.getvalue()
                        # 存储 measure_ids 用于下载文件名
                        mids_used = [(measure_ids.get(m) or m).strip() for m, _ in measure_item_list]
                        st.session_state.n1_excel_report_measure_ids = mids_used
                        st.success("已生成报告表，请点击下方按钮下载。")
                    except Exception as e:
                        st.error(f"生成报告表时出错: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        if st.session_state.get("n1_excel_report_bytes"):
            mids = st.session_state.get("n1_excel_report_measure_ids") or ["measure"]
            safe_mid = re.sub(r'[\\/:*?"<>|]+', '_', "-".join(str(m).strip() for m in mids)).strip(" .") or "measure"
            user_name = st.session_state.get("user_name", "unknown_user")
            safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name)).strip() or "unknown_user"
            today = date.today().strftime("%Y-%m-%d")
            file_name = f"{safe_mid}_efa_report_{today}_{safe_user}.xlsx"
            st.download_button(
                "⬇️ 下载 Excel 报告表",
                data=st.session_state.n1_excel_report_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="n1_download_excel_report",
            )

        '''
        # =========================================================
        # 🆕 新增功能: 保存结构并跳转 CFA
        # =========================================================
        st.markdown("---")
        st.markdown("### 🚀 下一步: 验证性因子分析 (CFA)")
        
        col_next_1, col_next_2 = st.columns([2, 1])
        
        with col_next_1:
            st.info("点击下方按钮，将当前的 EFA 因子结构（题目归属）保存，并自动跳转到 N2 模块进行验证。")
            
            # 提取因子结构: 找出每个因子中载荷 > 0.4 的题目
            def extract_efa_structure(load_df, threshold=0.4):
                structure = {}
                for factor_col in load_df.columns:
                    # 筛选该因子下绝对值大于阈值的题目
                    items = load_df.index[load_df[factor_col].abs() > threshold].tolist()
                    structure[factor_col] = items
                return structure

            if st.button("💾 保存结构并跳转至 N2 CFA 模块 ->", type="primary"):
                # 1. 提取结构
                efa_structure = extract_efa_structure(loadings_sorted)
                st.session_state.efa_suggested_structure = efa_structure
                st.session_state.efa_source_data = df_final # 同时把清洗好的数据带过去
                
                # 2. 跳转导航
                # 定义回调函数改状态 (避免直接改报错)
                st.session_state.nav_selection = "3. N2 CFA数据分析"
                st.rerun()
    '''
    # --- Feature 3: Post-Analysis Checks ---
    '''
    
    if 'n1_result_df' in st.session_state:
        st.markdown("---")
        st.subheader("3. 最终模型检验 (KMO, Bartlett & Reliability)")
        
        res_df = st.session_state.n1_result_df
        loadings = st.session_state.n1_loadings
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**保留的题目 ({len(st.session_state.n1_kept)})**")
            st.write(st.session_state.n1_kept)
        with col2:
            st.write(f"**删除的题目 ({len(st.session_state.n1_deleted)})**")
            st.write(st.session_state.n1_deleted)

            # 手动恢复题目 (结果微调)
            # ========================================================
            if len(st.session_state.n1_deleted) > 0:
                st.markdown("---")
                st.markdown("#### 🛠️ 结果手动调整：恢复被删题目")
                st.caption("如果你认为某些被删的题目具有重要的理论意义，可以将它们选中并强制保留。")
                
                # 1. 多选框：从已删除的列表中选择
                items_to_restore = st.multiselect(
                    "请选择要【强制保留】的题目:",
                    options=st.session_state.n1_deleted
                )
                
                # 2. 重新运行按钮
                if st.button("🔄 恢复选中题目并重新运行 EFA", type="secondary"):
                    if items_to_restore:
                        with st.spinner(f"正在强制保留 {len(items_to_restore)} 个题目并重新运行..."):
                            # 清除旧结果
                            if 'n1_result_df' in st.session_state:
                                del st.session_state['n1_result_df']
                                
                            # 重新运行管道，传入 whitelist
                            # 注意：这里需要获取 manual_k_val，虽然它在上面的作用域，但在 streamlit 中通常可以获取到
                            # 为了保险，我们重新判断一下 k 的获取逻辑，或者直接复用 session_state 里的 k
                            # 但最简单的方法是复用上面的 df_numeric 和 manual_k_val
                            
                            k_arg = st.session_state.get('n1_factors', None)
                            run_cols = st.session_state.get("n1_df_for_run_columns", df_numeric.columns.tolist())
                            run_cols = [c for c in run_cols if c in df_numeric.columns]
                            df_input = df_numeric[run_cols] if run_cols else df_numeric

                            # 再次调用核心函数
                            final_df, final_loadings, kept, deleted, n_factors = run_pipeline_streamlit(
                                df_input,
                                fixed_factors=manual_k_val,
                                whitelist=items_to_restore
                            )
                            
                            # 更新 Session State
                            st.session_state.n1_result_df = final_df
                            st.session_state.n1_loadings = final_loadings
                            st.session_state.n1_kept = kept
                            st.session_state.n1_deleted = deleted
                            st.session_state.n1_factors = n_factors
                            
                            st.rerun() # 强制刷新页面显示新结果
                    else:
                        st.warning("请至少选择一个要恢复的题目。")


        
        st.markdown("#### 最终因子载荷矩阵 (Factor Loadings)")
        # 按照题目序号排序
        loadings_sorted_final = sort_dataframe_by_item_names(loadings)
        st.dataframe(loadings_sorted_final.style.background_gradient(cmap="Blues").format("{:.3f}"))

        st.markdown("#### 效度检验")
        try:
            kmo_all, kmo_model = calculate_kmo(res_df)
            chi_square_value, p_value = calculate_bartlett_sphericity(res_df)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("KMO (总体)", f"{kmo_model:.3f}")
            k2.metric("Bartlett 卡方", f"{chi_square_value:.2f}")
            k3.metric("Bartlett P值", f"{p_value:.3e}")
            
            if kmo_model > 0.6 and p_value < 0.05:
                st.success("✅ 数据通过 KMO (>0.6) 和 Bartlett 球形检验 (p<0.05)。")
            else:
                st.error("❌ 数据可能不适合做因子分析。")
        except Exception as e:
            st.error(f"计算 KMO/Bartlett 时出错: {e}")

        st.markdown("#### 信度检验 (Internal Consistency)")
        try:
            alpha = cronbach_alpha(res_df)
            st.metric("Cronbach's Alpha", f"{alpha:.3f}")
            if alpha > 0.7:
                st.success("✅ 信度良好 (>0.7)")
            elif alpha > 0.6:
                st.warning("⚠️ 信度一般 (0.6 - 0.7)")
            else:
                st.error("❌ 信度不可接受 (<0.6)")
        except Exception as e:
            st.error(f"计算 Alpha 时出错: {e}")
    '''
    
