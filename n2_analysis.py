import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
# semopy 延迟导入：在实际使用时才加载，减少启动时间
import io
import re
from difflib import SequenceMatcher
from scipy.stats import chi2
from typing import Any, Tuple

# 导入通用工具函数
from utils import smart_multiselect, parse_item_col, sort_item_cols_by_number
from n1_analysis import cronbach_alpha

try:
    from db_save import save_formula_params, save_score_records, build_formula_params_json
    _DB_SAVE_AVAILABLE = True
except ImportError:
    save_formula_params = save_score_records = None
    build_formula_params_json = None
    _DB_SAVE_AVAILABLE = False
# ==============================================================================
# 核心算法区域 (增强版)
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
# 页面渲染逻辑
# ==============================================================================

def render_single_cfa_clean():
    st.title("模块 3: Single-Factor CFA")

    # --- 1. 数据来源 ---
    st.sidebar.markdown("### 数据来源设置")

    has_cleaning_data = 'sub_datasets' in st.session_state and len(st.session_state.sub_datasets) > 0
    has_dual_data = (
        st.session_state.get("dc_merge_done")
        and st.session_state.get("dc_dataset_full")
        and st.session_state.get("dc_measures")
    )
    has_n1_data = 'n1_result_df' in st.session_state and st.session_state.n1_result_df is not None

    source_options = ["📤 上传新文件 (Excel/CSV)"]
    if has_cleaning_data:
        source_options.append("💾 来自 Data Cleaning（子数据集）")
    if has_dual_data:
        source_options.append("💾 来自 Data Cleaning（四数据集）")
    if has_n1_data:
        source_options.append("🧬 来自 N1 模块生成的数据 (EFA Result)")

    data_source = st.radio("请选择数据来源:", source_options, horizontal=False)

    df_analysis = None

    if data_source == "🧬 来自 N1 模块生成的数据 (EFA Result)":
        df_analysis = st.session_state.n1_result_df
        st.info(f"正在使用 N1 模块 EFA 分析后的最终数据 (包含 {len(st.session_state.n1_kept)} 个题目)。")

    elif data_source == "💾 来自 Data Cleaning（四数据集）":
        from .data_cleaning_dual import get_dual_mode_analysis_df
        dataset_names = ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]
        selected_dataset = st.selectbox("1. 选择数据集", dataset_names, key="n2_dual_dataset")
        measure_names = list(st.session_state.dc_measures.keys())
        if not measure_names:
            st.warning("请在数据清洗模块的「Measure 划分」中至少定义一个 Measure。")
        else:
            selected_measures = st.multiselect(
                "2. 选择 Measure（可多选）",
                measure_names,
                default=[measure_names[0]] if measure_names else [],
                key="n2_dual_measures",
            )
            if selected_measures:
                df_analysis = get_dual_mode_analysis_df(
                    selected_dataset,
                    selected_measures,
                    st.session_state.dc_dataset_full,
                    st.session_state.dc_measures,
                    item_columns_only=True,
                )
            if df_analysis is not None:
                st.info(f"使用 **{selected_dataset}**，Measure: **{', '.join(selected_measures)}**（{df_analysis.shape[0]} 行 × {df_analysis.shape[1]} 列，仅题目列）")

    elif data_source == "💾 来自 Data Cleaning（子数据集）":
        # ... (原来的代码: dataset_names = ... selected_name = ...)
        dataset_names = list(st.session_state.sub_datasets.keys())
        selected_name = st.selectbox("请选择已保存的子数据集:", dataset_names)
        if selected_name:
            df_analysis = st.session_state.sub_datasets[selected_name]

            # 显示数据集信息和更新状态
            update_info = ""
            if ('sub_datasets_updated' in st.session_state and
                selected_name in st.session_state.sub_datasets_updated):
                update_time = st.session_state.sub_datasets_updated[selected_name]
                update_info = f" (最后更新: {update_time.strftime('%H:%M:%S')})"

            st.info(f"正在使用 Data Cleaning 缓存数据集: **{selected_name}**{update_info}")
            st.success("✅ 数据集包含最新的序号重命名结果")

    else:  # 上传文件
        # ... (原来的上传代码保持不变)
        uploaded_file = st.file_uploader("请上传用于分析的数据文件", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
             # ... (读取文件的逻辑保持不变)
             try:
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df_upload = pd.read_excel(uploaded_file)
                else:
                    df_upload = pd.read_csv(uploaded_file)
                st.write("文件预览 (前5行):")
                st.dataframe(df_upload.head())
                df_analysis = df_upload
                st.success(f"成功加载文件! 共 {df_analysis.shape[0]} 行。")
             except Exception as e:
                st.error(f"读取文件失败: {e}")

    st.markdown("---")

    if df_analysis is None:
        st.warning("👈 请先在上方选择数据来源或上传文件。")
        return

    # 清洗列名
    def clean_col_name(name):
        return re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(name))

    df_analysis.columns = [clean_col_name(c) for c in df_analysis.columns]
    
    # ==========================================================
    # 🔴 核心修改：智能保留数值列 (Smart Numeric Filter)
    # ==========================================================
    st.write("📊 **数据预处理报告**")

    # 1. 强制转换为数值 (非数字变为 NaN)
    df_numeric = df_analysis.apply(pd.to_numeric, errors='coerce')
    
    # 2. 剔除“非数值列”
    # 如果某一列转换后全是 NaN (例如"姓名"列)，说明它不是量表题，直接删除该列
    # 这样可以防止后面 dropna 时把整行数据都误删了
    cols_before = df_numeric.columns
    df_numeric = df_numeric.dropna(axis=1, how='all')
    cols_after = df_numeric.columns
    
    # 提示用户删除了哪些列
    dropped_cols = set(cols_before) - set(cols_after)
    if dropped_cols:
        st.info(f"ℹ️ 已自动过滤掉 {len(dropped_cols)} 个非数值列 (仅用于分析的量表题被保留): {', '.join(list(dropped_cols)[:5])}...")

    # 3. 剔除“含有缺失值或无穷大值的行”
    # 现在剩下的列都是纯数字了，可以安全地删除漏填的样本
    original_len = len(df_numeric)

    # 移除包含NaN的行
    df_numeric = df_numeric.dropna(axis=0, how='any')

    # 移除包含无穷大值(Inf)的行
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()
    df_numeric = df_numeric[~df_numeric.isin([np.inf, -np.inf]).any(axis=1)]

    cleaned_len = len(df_numeric)

    removed_rows = original_len - cleaned_len
    if removed_rows > 0:
        st.warning(f"⚠️ 已移除 {removed_rows} 行含有缺失值或无穷大值的样本。CFA 最终分析样本量: {cleaned_len}")
    else:
        st.success(f"✅ 数据完整，有效样本量: {cleaned_len}")
    
    if cleaned_len < 10:
        st.error("❌ 有效样本量太少 (<10)，无法进行 CFA 分析。请检查数据是否包含大量缺失值。")
        return

    # 更新供用户选择的题目列表 (现在只包含数值列了)
    all_items = df_numeric.columns.tolist()
    # ==========================================================
    # --- 2. 模型构建 ---
    st.subheader("1. 构建模型 (Model Configuration)")
    
    # 🆕 导入 EFA 结构按钮
    if 'efa_suggested_structure' in st.session_state:
        st.markdown("##### 🔗 EFA 连接")
        if st.button("📥 导入 N1 模块生成的 EFA 结构"):
            structure = st.session_state.efa_suggested_structure
            # 假设 EFA 只有一个主因子 F1 (或者取第一个因子作为 Trait)
            # 如果有多个因子，这里默认取第一个作为演示，或者让用户选
            # 这里简单化：取所有因子的题目合集作为 Trait Items (通常单因子CFA)
            # 或者取 F1 的题目。
            
            # 策略：默认把 EFA 的第一个因子的题目填入 A
            first_factor = list(structure.keys())[0]
            items_for_f1 = structure[first_factor]
            
            st.session_state.auto_fill_items = items_for_f1
            st.success(f"已导入 EFA {first_factor} 的 {len(items_for_f1)} 个题目到主因子。")

    col1, col2 = st.columns(2)
    
    # 获取默认选项
    default_items = st.session_state.get('auto_fill_items', [])
    # 确保默认选项都在当前 all_items 里 (防止文件名不同导致的 mismatch)
    default_items = [i for i in default_items if i in all_items]

    with col1:
        st.markdown("#### 🅰️ 主因子 (Trait Factor)")
        factor_name = st.text_input("给主因子起个名 (英文):", value="Factor1")
        
        # 主因子选择
        factor_items = smart_multiselect(
            options=all_items,
            label=f"选择属于 {factor_name} 的题目",
            key_suffix="cfa_factor1",
            default_selected=default_items,
            show_selection_controls=True,
        )

    with col2:
        st.markdown("#### 🅱️ 方法因子 (Method Factor)")
        method_name = st.text_input("给方法因子起个名 (英文):", value="Method")
        
        # 限制范围: 只能从 factor_items (主因子已选题目) 中选择
        method_options = factor_items if factor_items else []
        method_key_suffix = "cfa_method"
        method_sig_key = f"{method_key_suffix}_options_sig"
        method_options_sig = tuple(method_options)
        if st.session_state.get(method_sig_key) != method_options_sig:
            _reset_smart_multiselect_cache(method_key_suffix)
            st.session_state[method_sig_key] = method_options_sig
        # 根据统一规则（末尾 r）自动预选方法因子，供用户确认
        default_method_items = [x for x in method_options if _is_reverse_coded(x)] if method_options else []
        st.caption("已根据统一规则（题目文本末尾为 r）预选方法因子题目，请确认或修改。")
        def _on_reset_method_n2():
            _reset_smart_multiselect_cache(method_key_suffix)
            st.session_state[method_sig_key] = method_options_sig
            st.session_state[f"{method_key_suffix}_last_selected"] = default_method_items
        st.button('按”末尾 r”规则重新预选方法因子题目', key="n2_btn_reset_method_defaults",
                  on_click=_on_reset_method_n2)
        method_items = smart_multiselect(
            options=method_options,
            label=f"选择受到 {method_name} 影响的题目",
            key_suffix=method_key_suffix,
            default_selected=default_method_items,
            show_selection_controls=True,
        )
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

        # --- 6. 基于 Single-Factor CFA 结果计算每个样本“最终得分” ---
        if all(k in st.session_state for k in ("n2_df_cfa", "n2_factor_items", "n2_estimates", "n2_factor_name")):
            st.markdown("---")
            st.markdown("#### 🧮 最终得分计算（基于 CFA）")
            st.caption("按公式：w = Σ⁻¹λφ，f_raw = wᵀ(x-μ)，z = f_raw / sqrt(wᵀΣw)，FinalScore = 80 + 10·z。absolute final score 与下方导出公式参数表将复用此处填写的 scale_min / scale_max。")
            n2_col1, n2_col2 = st.columns(2)
            with n2_col1:
                n2_scale_min = st.number_input("scale_min", min_value=0, value=1, step=1, key="n2_scale_min")
            with n2_col2:
                n2_scale_max = st.number_input("scale_max", min_value=1, value=7, step=1, key="n2_scale_max")
            # 非四数据集时，需用户填写「当前是 dataset 几」用于下载文件名
            is_four_dataset = (data_source == "💾 来自 Data Cleaning（四数据集）")
            if not is_four_dataset:
                st.number_input("当前是 dataset 1-4中的哪一个？（用于下载文件命名）", min_value=1, value=1, step=1, key="n2_scored_dataset_n")
            if int(n2_scale_min) >= int(n2_scale_max):
                st.error("scale_min 必须小于 scale_max。")
            elif (int(n2_scale_min), int(n2_scale_max)) != (1, 7):
                st.warning("⚠️ 当前 scale_min / scale_max 非默认 [1, 7]。请确认量表边界与问卷设计一致，否则 absolute 0–100 分数解释可能产生偏差。")
            if st.button("计算最终得分并生成可下载数据集", key="n2_btn_gen_final_score"):
                try:
                    df_cfa = st.session_state.n2_df_cfa.copy()
                    factor_items = st.session_state.n2_factor_items
                    est = st.session_state.n2_estimates.copy()
                    fname = st.session_state.n2_factor_name

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

                    def _clean_col(name):
                        return re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(name))

                    # 按 EFA/CFA 题号升序（顺序必须一致）
                    sorted_items = sort_item_cols_by_number(factor_items)
                    sorted_items_clean = [_clean_col(c) for c in sorted_items]
                    used_cols = [c for c in sorted_items_clean if c in df_cfa.columns]
                    if not used_cols:
                        st.error("未找到可用于计算最终得分的题目列。")
                        st.stop()

                    x_df_all = df_cfa[used_cols].apply(pd.to_numeric, errors="coerce")
                    missing_mask = x_df_all.isna().any(axis=1)
                    n_missing = int(missing_mask.sum())
                    x_df = x_df_all.loc[~missing_mask].copy()

                    scored_df = x_df_all.copy()
                    scored_df["relative_final_score"] = -999.0
                    scored_df["absolute_final_score"] = -999.0

                    if not x_df.empty:
                        # μ 与 Σ（仅基于有效样本）
                        mu_vec = x_df.mean(axis=0).values.reshape(-1, 1)
                        sigma = x_df.cov().values

                        # 提取非标准化载荷 λ（兼容两种 semopy 输出结构）
                        lambda_map = {}
                        if {"LHS", "op", "RHS"}.issubset(est.columns):
                            load_rows = est[(est["op"] == "=~") & (est["LHS"] == fname)]
                            if not load_rows.empty:
                                for _, r in load_rows.iterrows():
                                    lambda_map[str(r["RHS"])] = _to_num(r.get("Estimate", np.nan))
                            else:
                                load_rows = est[(est["op"] == "~") & (est["RHS"] == fname)]
                                for _, r in load_rows.iterrows():
                                    lambda_map[str(r["LHS"])] = _to_num(r.get("Estimate", np.nan))

                        lambda_vec = np.array([_to_num(lambda_map.get(c, np.nan)) for c in used_cols], dtype=float).reshape(-1, 1)
                        if np.isnan(lambda_vec).any():
                            miss = [used_cols[i] for i, v in enumerate(lambda_vec.flatten()) if np.isnan(v)]
                            st.error(f"部分题目无法匹配到 CFA 非标准化载荷（Estimate）：{', '.join(miss[:8])}")
                            st.stop()

                        # 提取主因子方差 φ
                        phi = np.nan
                        if {"LHS", "op", "RHS"}.issubset(est.columns):
                            vv = est[(est["op"] == "~~") & (est["LHS"] == fname) & (est["RHS"] == fname)]
                            if not vv.empty:
                                phi = _to_num(vv.iloc[0].get("Estimate", np.nan))
                        if np.isnan(phi):
                            st.error("未能从 CFA 结果中提取主因子方差 φ（Estimate of F ~~ F）。")
                            st.stop()

                        # w = Σ^{-1} λ φ，优先解线性方程，奇异时做轻微岭稳定
                        rhs = lambda_vec * phi
                        try:
                            w = np.linalg.solve(sigma, rhs)
                        except np.linalg.LinAlgError:
                            ridge = 1e-8 * np.eye(sigma.shape[0])
                            w = np.linalg.solve(sigma + ridge, rhs)
                            st.warning("协方差矩阵接近奇异，已使用微小岭稳定项计算权重。")

                        # f_raw, SD_cal, z, FinalScore（仅填充有效样本行）
                        x_centered = x_df.values - mu_vec.T
                        f_raw = (x_centered @ w).flatten()
                        sd_cal = float(np.sqrt(np.maximum((w.T @ sigma @ w).item(), 1e-12)))
                        z = f_raw / sd_cal
                        final_score = 80 + 10 * z
                        scored_df.loc[x_df.index, "relative_final_score"] = final_score

                        # absolute_final_score（严格按文档）：复用上方 scale_min / scale_max
                        # 1) S = w^T x（不做均值中心化）
                        # 2) 由权重符号与量表上下限推导 S_min / S_max
                        # 3) Score_0_100 = 100 * (S - S_min) / (S_max - S_min)
                        w_vec = w.reshape(-1)
                        L = float(st.session_state.get("n2_scale_min", 1))
                        U = float(st.session_state.get("n2_scale_max", 7))
                        if L >= U:
                            st.error("scale_min 必须小于 scale_max，请修正后再计算。")
                            st.stop()
                        lo = np.where(w_vec >= 0, L, U)
                        hi = np.where(w_vec >= 0, U, L)
                        s_min = float(w_vec @ lo)
                        s_max = float(w_vec @ hi)
                        den = s_max - s_min
                        if den <= 1e-12:
                            st.error("absolute final score 计算失败：S_max 与 S_min 过近，无法线性映射。")
                            st.stop()
                        s_raw = (x_df.values @ w_vec).flatten()
                        absolute_final_score = 100.0 * (s_raw - s_min) / den
                        absolute_final_score = np.clip(absolute_final_score, 0.0, 100.0)
                        scored_df.loc[x_df.index, "absolute_final_score"] = absolute_final_score
                    st.session_state.n2_scored_df = scored_df

                    csv_bytes = scored_df.to_csv(index=False).encode("utf-8-sig")
                    st.session_state.n2_scored_csv_bytes = csv_bytes

                    xlsx_buf = io.BytesIO()
                    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w_x:
                        scored_df.to_excel(w_x, sheet_name="ScoredData", index=False)
                    xlsx_buf.seek(0)
                    st.session_state.n2_scored_xlsx_bytes = xlsx_buf.getvalue()

                    if n_missing > 0:
                        st.warning(f"检测到 {n_missing} 行样本存在缺失值，这些样本的“最终得分”已记为 -999。")
                    else:
                        st.info("检测到 0 行样本存在缺失值。")
                    st.success(f"最终得分计算完成：{len(scored_df)} 个样本（有效计分 {len(x_df)} 行）。")
                except Exception as e:
                    st.error(f"计算最终得分时出错: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # 预览窗口 + 下载按钮
            if st.session_state.get("n2_scored_df") is not None:
                st.markdown("##### 数据预览（含最终得分）")
                preview_df = st.session_state.n2_scored_df.head(20).copy()
                if "relative_final_score" in preview_df.columns:
                    preview_df["relative_final_score"] = pd.to_numeric(preview_df["relative_final_score"], errors="coerce").round(3)
                if "absolute_final_score" in preview_df.columns:
                    preview_df["absolute_final_score"] = pd.to_numeric(preview_df["absolute_final_score"], errors="coerce").round(3)
                st.dataframe(preview_df, use_container_width=True)
                measure_id = st.session_state.get("n2_measure_id") or "measure"
                cfa_type = "prelim_single_cfa" if st.session_state.get("n2_prelim_single_cfa") else "single_cfa"
                safe_mid = re.sub(r'[\\/:*?"<>|]+', '_', str(measure_id)).strip() or "measure"
                user_name = st.session_state.get("user_name", "unknown_user")
                safe_user = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name)).strip() or "unknown_user"
                today = date.today().strftime("%Y-%m-%d")
                if data_source == "💾 来自 Data Cleaning（四数据集）":
                    ds_name = st.session_state.get("n2_dual_dataset", "Dataset4")
                    m = re.search(r"(\d+)", str(ds_name))
                    dataset_n = int(m.group(1)) if m else 4
                else:
                    dataset_n = int(st.session_state.get("n2_scored_dataset_n", 1))
                scored_base = f"{safe_mid}_{cfa_type}_dataset{dataset_n}_scored_{today}_{safe_user}"
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    if st.session_state.get("n2_scored_csv_bytes"):
                        st.download_button(
                            "⬇️ 下载含最终得分数据（CSV）",
                            data=st.session_state.n2_scored_csv_bytes,
                            file_name=f"{scored_base}.csv",
                            mime="text/csv",
                            key="n2_download_scored_csv",
                        )
                with col_dl2:
                    if st.session_state.get("n2_scored_xlsx_bytes"):
                        st.download_button(
                            "⬇️ 下载含最终得分数据（Excel）",
                            data=st.session_state.n2_scored_xlsx_bytes,
                            file_name=f"{scored_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="n2_download_scored_xlsx",
                        )
                with col_dl3:
                    if _DB_SAVE_AVAILABLE and st.button("☁️ 保存最终得分到云端", key="n2_save_scored_df_cloud"):
                        scored_df = st.session_state.n2_scored_df
                        measure_id = st.session_state.get("n2_measure_id") or "measure"
                        ok, msg = save_score_records(scored_df, measure_id=measure_id)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(f"保存失败: {msg}")

            st.markdown("#### 📤 导出最终得分公式参数表")
            st.caption("导出部署时使用的线性公式参数：FinalScore = intercept + Σ(beta * item)。")
            st.caption("注意：此处 reverse 列仅用于记录题目是否已 reverse-coded（末尾 r）；生产侧请直接使用已反向处理后的题目作答值，不再执行二次 reverse coding。")
            _n2_smin = int(st.session_state.get("n2_scale_min", 1))
            _n2_smax = int(st.session_state.get("n2_scale_max", 7))
            st.caption(f"公式参数将使用**上方** scale_min / scale_max（当前为 [{_n2_smin}, {_n2_smax}]）。measure_id 沿用上方统一填写的值。若非 1–7 量表，请先在上方修改后再生成。")
            if st.button("生成公式参数表", key="n2_btn_build_formula_table"):
                try:
                    formula_scale_min = int(st.session_state.get("n2_scale_min", 1))
                    formula_scale_max = int(st.session_state.get("n2_scale_max", 7))
                    if formula_scale_min >= formula_scale_max:
                        st.error("scale_min 必须小于 scale_max，请在上方修正后再生成公式参数表。")
                        st.stop()

                    df_cfa = st.session_state.n2_df_cfa.copy()
                    factor_items = st.session_state.n2_factor_items
                    est = st.session_state.n2_estimates.copy()
                    fname = st.session_state.n2_factor_name

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

                    def _clean_col(name):
                        return re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(name))

                    sorted_items = sort_item_cols_by_number(factor_items)
                    sorted_items_clean = [_clean_col(c) for c in sorted_items]
                    used_cols = [c for c in sorted_items_clean if c in df_cfa.columns]
                    if not used_cols:
                        st.error("未找到可用于导出公式的题目列。")
                        st.stop()

                    x_df_all = df_cfa[used_cols].apply(pd.to_numeric, errors="coerce")
                    x_df = x_df_all.dropna(axis=0, how="any")
                    if x_df.empty:
                        st.error("有效样本为空，无法导出公式参数。")
                        st.stop()

                    mu_vec = x_df.mean(axis=0).values.reshape(-1, 1)
                    sigma = x_df.cov().values

                    lambda_map = {}
                    if {"LHS", "op", "RHS"}.issubset(est.columns):
                        load_rows = est[(est["op"] == "=~") & (est["LHS"] == fname)]
                        if not load_rows.empty:
                            for _, r in load_rows.iterrows():
                                lambda_map[str(r["RHS"])] = _to_num(r.get("Estimate", np.nan))
                        else:
                            load_rows = est[(est["op"] == "~") & (est["RHS"] == fname)]
                            for _, r in load_rows.iterrows():
                                lambda_map[str(r["LHS"])] = _to_num(r.get("Estimate", np.nan))

                    lambda_vec = np.array([_to_num(lambda_map.get(c, np.nan)) for c in used_cols], dtype=float).reshape(-1, 1)
                    if np.isnan(lambda_vec).any():
                        miss = [used_cols[i] for i, v in enumerate(lambda_vec.flatten()) if np.isnan(v)]
                        st.error(f"部分题目无法匹配到 CFA 非标准化载荷（Estimate）：{', '.join(miss[:8])}")
                        st.stop()

                    phi = np.nan
                    if {"LHS", "op", "RHS"}.issubset(est.columns):
                        vv = est[(est["op"] == "~~") & (est["LHS"] == fname) & (est["RHS"] == fname)]
                        if not vv.empty:
                            phi = _to_num(vv.iloc[0].get("Estimate", np.nan))
                    if np.isnan(phi):
                        st.error("未能从 CFA 结果中提取主因子方差 φ。")
                        st.stop()

                    rhs = lambda_vec * phi
                    try:
                        w = np.linalg.solve(sigma, rhs)
                    except np.linalg.LinAlgError:
                        ridge = 1e-8 * np.eye(sigma.shape[0])
                        w = np.linalg.solve(sigma + ridge, rhs)

                    reporting_mean = 80.0
                    reporting_sd = 10.0
                    sd_cal = float(np.sqrt(np.maximum((w.T @ sigma @ w).item(), 1e-12)))
                    beta_star = (reporting_sd / sd_cal) * w.flatten()
                    intercept_star = float(reporting_mean - float(np.sum(beta_star * mu_vec.flatten())))

                    # absolute 参数（0-100）：复用上方 scale_min/scale_max
                    w_flat = w.flatten()
                    l_abs = float(formula_scale_min)
                    u_abs = float(formula_scale_max)
                    lo = np.where(w_flat >= 0, l_abs, u_abs)
                    hi = np.where(w_flat >= 0, u_abs, l_abs)
                    s_min = float(w_flat @ lo)
                    s_max = float(w_flat @ hi)
                    den_abs = s_max - s_min
                    if den_abs <= 1e-12:
                        st.error("absolute 参数计算失败：S_max 与 S_min 过近，无法线性映射。")
                        st.stop()
                    beta_abs = (100.0 / den_abs) * w_flat
                    intercept_abs = float(-(100.0 / den_abs) * s_min)

                    dataset_name = (
                        st.session_state.get("n2_dual_dataset")
                        or st.session_state.get("n2_selected_dataset")
                        or "n2_dataset"
                    )
                    measuregroup_title = str(st.session_state.get("n3_measuregroup_title", "") or "").strip()
                    created_date = date.today().strftime("%Y-%m-%d")
                    formula_measure_id = (st.session_state.get("n2_measure_id") or "").strip() or "measure"
                    # 方案 A：formula_measure（1 行）+ formula_items（N 行）
                    formula_measure_df = pd.DataFrame([{
                        "measure_id": formula_measure_id,
                        "measuregroup_title": measuregroup_title,
                        "dataset": str(dataset_name),
                        "CFA_model": "single-factor",
                        "intercept_rel": intercept_star,
                        "intercept_abs": intercept_abs,
                        "reporting_mean": reporting_mean,
                        "reporting_sd": reporting_sd,
                        "scale_min": int(formula_scale_min),
                        "scale_max": int(formula_scale_max),
                        "创建时间": created_date,
                    }])
                    formula_items_rows = []
                    for i, c in enumerate(used_cols):
                        item_num, item_text = _extract_item_num_and_text(c)
                        formula_items_rows.append({
                            "measure_id": formula_measure_id,
                            "item_col": str(c),
                            "item_text": item_text,
                            "item_num": item_num,
                            "reverse": 1 if _is_reverse_coded(c) else 0,
                            "beta_rel": float(beta_star[i]),
                            "beta_abs": float(beta_abs[i]),
                            "sort_order": i + 1,
                            "创建时间": created_date,
                        })
                    formula_items_df = pd.DataFrame(formula_items_rows)
                    # 合并为扁平格式供计分逻辑使用（items + intercept_rel 来自 measure）
                    formula_flat = formula_items_df.copy()
                    formula_flat["intercept_rel"] = intercept_star
                    formula_flat["intercept_abs"] = intercept_abs
                    st.session_state.n2_formula_df = formula_flat
                    st.session_state.n2_formula_measure_df = formula_measure_df
                    st.session_state.n2_formula_items_df = formula_items_df
                    # Excel/CSV 导出：扁平化单表（measure 列合并进 items，维持原格式）
                    export_flat = formula_items_df.copy()
                    for col, val in formula_measure_df.iloc[0].items():
                        if col not in export_flat.columns:
                            export_flat[col] = val
                    cols_order = ["measure_id", "measuregroup_title", "dataset", "CFA_model", "intercept_rel", "intercept_abs",
                                  "reporting_mean", "reporting_sd", "scale_min", "scale_max", "创建时间",
                                  "item_col", "item_text", "item_num", "reverse", "beta_rel", "beta_abs", "sort_order"]
                    export_flat = export_flat[[c for c in cols_order if c in export_flat.columns]]
                    st.session_state.n2_formula_csv_bytes = export_flat.to_csv(index=False).encode("utf-8-sig")
                    buf_formula = io.BytesIO()
                    with pd.ExcelWriter(buf_formula, engine="xlsxwriter") as w_f:
                        export_flat.to_excel(w_f, sheet_name="formula", index=False)
                    buf_formula.seek(0)
                    st.session_state.n2_formula_xlsx_bytes = buf_formula.getvalue()
                    st.success("已生成公式参数表。")
                except Exception as e:
                    st.error(f"生成公式参数表失败: {e}")

            if st.session_state.get("n2_formula_df") is not None:
                st.markdown("##### 公式参数表预览（与 Excel/CSV 导出格式一致）")
                m_df = st.session_state.get("n2_formula_measure_df")
                i_df = st.session_state.get("n2_formula_items_df")
                if m_df is not None and i_df is not None:
                    preview_flat = i_df.copy()
                    for col, val in m_df.iloc[0].items():
                        if col not in preview_flat.columns:
                            preview_flat[col] = val
                    st.dataframe(preview_flat.head(20), use_container_width=True)
                formula_mid = st.session_state.get("n2_measure_id") or "measure"
                cfa_type_f = "prelim_single_cfa" if st.session_state.get("n2_prelim_single_cfa") else "single_cfa"
                safe_mid_f = re.sub(r'[\\/:*?"<>|]+', '_', str(formula_mid)).strip() or "measure"
                user_name_f = st.session_state.get("user_name", "unknown_user")
                safe_user_f = re.sub(r'[\\/:*?"<>|]+', '_', str(user_name_f)).strip() or "unknown_user"
                today_f = date.today().strftime("%Y-%m-%d")
                formula_base = f"{safe_mid_f}_{cfa_type_f}_final_score_formula_{today_f}_{safe_user_f}"
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    if st.session_state.get("n2_formula_csv_bytes"):
                        st.download_button(
                            "⬇️ 下载公式参数表（CSV）",
                            data=st.session_state.n2_formula_csv_bytes,
                            file_name=f"{formula_base}.csv",
                            mime="text/csv",
                            key="n2_download_formula_csv",
                        )
                with f2:
                    if st.session_state.get("n2_formula_xlsx_bytes"):
                        st.download_button(
                            "⬇️ 下载公式参数表（Excel）",
                            data=st.session_state.n2_formula_xlsx_bytes,
                            file_name=f"{formula_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="n2_download_formula_xlsx",
                        )
                with f3:
                    if _DB_SAVE_AVAILABLE and st.button("☁️ 保存公式参数到云端", key="n2_save_formula_cloud"):
                        m_df = st.session_state.get("n2_formula_measure_df")
                        i_df = st.session_state.get("n2_formula_items_df")
                        if m_df is not None and i_df is not None:
                            ok, msg = save_formula_params(m_df, i_df)
                        else:
                            ok, msg = False, "公式表未就绪，请先生成公式参数表。"
                        if ok:
                            st.success(msg)
                        else:
                            st.error(f"保存失败: {msg}")
                with f4:
                    m_df = st.session_state.get("n2_formula_measure_df")
                    i_df = st.session_state.get("n2_formula_items_df")
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
                        json_fn = f"{formula_base}.json"
                        st.download_button(
                            "⬇️ 下载公式参数表（JSON）",
                            data=json_bytes,
                            file_name=json_fn,
                            mime="application/json",
                            key="n2_download_formula_json",
                        )

            # --- 7. 使用公式计分：支持 Dataset4 measure / 上传文件 ---
            has_dual_data = (
                st.session_state.get("dc_dataset_full")
                and st.session_state.get("dc_measures")
                and "Dataset4" in st.session_state.get("dc_dataset_full", {})
            )
            has_sub_dataset_source = (data_source == "💾 来自 Data Cleaning（子数据集）")
            has_upload_file_source = (data_source == "📤 上传新文件 (Excel/CSV)")
            if st.session_state.get("n2_formula_df") is not None and (has_dual_data or has_sub_dataset_source or has_upload_file_source):
                st.markdown("---")
                st.markdown("#### 🧾 使用公式计分")
                if has_dual_data:
                    st.caption("支持两种模式：①使用当前 Dataset4 measure；②上传新的 Excel/CSV 数据集并按当前公式计分。缺失样本记为 -999。")
                else:
                    st.caption("当前可上传新的 Excel/CSV 数据集，并按当前公式计算“relative final score”和“absolute final score”。缺失样本记为 -999。")

                def _clean_col_for_score(name: Any) -> str:
                    return re.sub(r"[^\w\u4e00-\u9fa5]", "_", str(name))

                def _safe_name_part(name: Any) -> str:
                    return re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", str(name)).strip("_") or "scored"

                def _prepare_formula_for_score(formula_df_src: pd.DataFrame):
                    formula_df_local = formula_df_src.copy()
                    formula_df_local["beta_rel"] = pd.to_numeric(
                        formula_df_local.get("beta_rel", formula_df_local.get("beta")), errors="coerce"
                    )
                    formula_df_local["intercept_rel"] = pd.to_numeric(
                        formula_df_local.get("intercept_rel"), errors="coerce"
                    )
                    formula_df_local["beta_abs"] = pd.to_numeric(formula_df_local.get("beta_abs"), errors="coerce")
                    formula_df_local["intercept_abs"] = pd.to_numeric(formula_df_local.get("intercept_abs"), errors="coerce")
                    formula_valid_local = formula_df_local[formula_df_local["beta_rel"].notna()].copy().reset_index(drop=True)
                    if formula_valid_local.empty:
                        return None, None, "公式参数表中未找到可用 beta_rel，无法计分。"
                    intercept_vals = formula_valid_local["intercept_rel"].dropna().unique().tolist()
                    if not intercept_vals:
                        return None, None, "公式参数表中 intercept_rel 缺失，无法计分。"
                    return formula_valid_local, float(intercept_vals[0]), ""

                def _normalize_match_text(s: Any) -> str:
                    s_txt = str(s or "").lower().strip()
                    s_txt = s_txt.replace("（", "(").replace("）", ")")
                    s_txt = re.sub(r"\s+", "", s_txt)
                    s_txt = re.sub(r"[^\w\u4e00-\u9fa5]", "", s_txt)
                    return s_txt

                def _build_row_mapping(formula_valid_local: pd.DataFrame, df_clean_cols: list[str], enable_item_num: bool, enable_fuzzy: bool):
                    row_to_col = {}
                    row_reason = {}
                    used_cols = set()
                    num_to_cols = {}
                    if enable_item_num:
                        for c in df_clean_cols:
                            num, _ = _extract_item_num_and_text(c)
                            if isinstance(num, (int, np.integer)) and not pd.isna(num):
                                num_to_cols.setdefault(int(num), []).append(c)

                    normalized_col_map = {c: _normalize_match_text(c) for c in df_clean_cols}
                    for ridx, r in formula_valid_local.iterrows():
                        item_col = str(r.get("item_col", "")).strip()
                        item_num = pd.to_numeric(r.get("item_num"), errors="coerce")
                        item_text = str(r.get("item_text", "")).strip()
                        chosen_col = None
                        reason = ""

                        if item_col and item_col in df_clean_cols and item_col not in used_cols:
                            chosen_col = item_col
                            reason = "列名精确匹配"

                        if chosen_col is None and enable_item_num and not pd.isna(item_num):
                            cands = [c for c in num_to_cols.get(int(item_num), []) if c not in used_cols]
                            if len(cands) == 1:
                                chosen_col = cands[0]
                                reason = "题号唯一匹配"

                        if chosen_col is None and enable_fuzzy:
                            target_text = _normalize_match_text(item_text or item_col)
                            best_col = None
                            best_score = 0.0
                            for c in df_clean_cols:
                                if c in used_cols:
                                    continue
                                score = SequenceMatcher(None, target_text, normalized_col_map.get(c, "")).ratio()
                                if score > best_score:
                                    best_score = score
                                    best_col = c
                            if best_col is not None and best_score >= 0.72:
                                chosen_col = best_col
                                reason = f"文本相似匹配({best_score:.2f})"

                        if chosen_col is not None:
                            row_to_col[ridx] = chosen_col
                            row_reason[ridx] = reason
                            used_cols.add(chosen_col)
                    return row_to_col, row_reason

                def _score_dataframe_with_mapping(
                    df_raw: pd.DataFrame,
                    df_clean: pd.DataFrame,
                    formula_valid_local: pd.DataFrame,
                    intercept_local: float,
                    row_to_col: dict[int, str],
                ):
                    ordered_cols = []
                    ordered_betas = []
                    ordered_betas_abs = []
                    unresolved_items = []

                    for ridx, r in formula_valid_local.iterrows():
                        chosen_col = row_to_col.get(ridx)
                        if chosen_col is None or chosen_col not in df_clean.columns:
                            unresolved_items.append(str(r.get("item_text", r.get("item_col", ridx))))
                            continue
                        ordered_cols.append(chosen_col)
                        ordered_betas.append(float(r.get("beta_rel", r.get("beta"))))
                        ordered_betas_abs.append(pd.to_numeric(r.get("beta_abs"), errors="coerce"))

                    if unresolved_items:
                        return None, unresolved_items, 0, "以下公式题目未匹配到数据列"
                    if not ordered_cols:
                        return None, ["未匹配到任何题目列"], 0, "未匹配到可用于计分的题目列"

                    x_all = df_clean[ordered_cols].apply(pd.to_numeric, errors="coerce")
                    missing_mask = x_all.isna().any(axis=1)
                    n_missing_local = int(missing_mask.sum())

                    scored_df_local = df_raw.copy()
                    scored_df_local["relative_final_score"] = -999.0
                    if (~missing_mask).any():
                        x_valid = x_all.loc[~missing_mask]
                        beta_arr = np.asarray(ordered_betas, dtype=float).reshape(-1, 1)
                        final_valid = intercept_local + (x_valid.values @ beta_arr).flatten()
                        scored_df_local.loc[x_valid.index, "relative_final_score"] = final_valid

                    intercept_abs_vals = pd.to_numeric(formula_valid_local.get("intercept_abs"), errors="coerce").dropna().unique().tolist()
                    has_abs_formula = (len(intercept_abs_vals) > 0) and all(pd.notna(v) for v in ordered_betas_abs)
                    if has_abs_formula:
                        scored_df_local["absolute_final_score"] = -999.0
                        if (~missing_mask).any():
                            x_valid = x_all.loc[~missing_mask]
                            beta_abs_arr = np.asarray(ordered_betas_abs, dtype=float).reshape(-1, 1)
                            abs_valid = float(intercept_abs_vals[0]) + (x_valid.values @ beta_abs_arr).flatten()
                            scored_df_local.loc[x_valid.index, "absolute_final_score"] = abs_valid

                    return scored_df_local, [], n_missing_local, ""

                def _read_uploaded_data(uploaded_file_obj):
                    name_lower = str(uploaded_file_obj.name).lower()
                    if name_lower.endswith(".csv"):
                        try:
                            return pd.read_csv(uploaded_file_obj)
                        except Exception:
                            uploaded_file_obj.seek(0)
                            return pd.read_csv(uploaded_file_obj, encoding="utf-8-sig")
                    return pd.read_excel(uploaded_file_obj)

                score_source_options = ["上传新的 Excel/CSV 数据集"]
                if has_dual_data:
                    score_source_options = ["使用当前 Dataset4 measure", "上传新的 Excel/CSV 数据集"]
                score_source_mode = st.radio(
                    "选择计分数据来源",
                    score_source_options,
                    horizontal=True,
                    key="n2_formula_score_source_mode",
                )

                formula_df_for_hint = st.session_state.get("n2_formula_df")
                formula_measure_ids = []
                if formula_df_for_hint is not None and "measure_id" in formula_df_for_hint.columns:
                    formula_measure_ids = (
                        formula_df_for_hint["measure_id"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .unique()
                        .tolist()
                    )
                if formula_measure_ids:
                    st.caption(f"当前公式 measure_id：{', '.join(formula_measure_ids[:5])}")

                formula_valid, intercept, formula_err = _prepare_formula_for_score(st.session_state.n2_formula_df)
                if formula_err:
                    st.error(formula_err)
                else:
                    if score_source_mode == "使用当前 Dataset4 measure" and has_dual_data:
                        measure_names_for_score = list(st.session_state.dc_measures.keys())
                        if not measure_names_for_score:
                            st.warning("当前未检测到可用 measure，请先在 Data Cleaning 模块完成 Measure 划分。")
                        else:
                            selected_measure_for_score = st.selectbox(
                                "选择要计分的 Dataset4 measure",
                                measure_names_for_score,
                                key="n2_formula_score_measure",
                            )
                            if formula_measure_ids and selected_measure_for_score not in formula_measure_ids:
                                st.warning(
                                    "⚠️ 公式表中的 measure_id 与当前选择的 measure 可能不一致："
                                    f"公式 measure_id={', '.join(formula_measure_ids[:3])}；"
                                    f"当前 measure={selected_measure_for_score}。"
                                    "请确认公式与数据属于同一量表后再计分。"
                                )

                            d4_df = st.session_state.dc_dataset_full.get("Dataset4")
                            selected_measure_cols = []
                            if d4_df is not None:
                                selected_measure_cols = [
                                    c for c in st.session_state.dc_measures.get(selected_measure_for_score, [])
                                    if c in d4_df.columns
                                ]
                            if d4_df is None or not selected_measure_cols:
                                st.error("在 Dataset4 中未找到该 measure 的可用列，无法计分。")
                            else:
                                st.info(
                                    f"将对 Dataset4 的 **{selected_measure_for_score}** 计分："
                                    f"{len(d4_df)} 行样本 × {len(selected_measure_cols)} 列题目。"
                                )
                                if st.button("按公式计算并生成 Dataset4 Measure 得分数据", key="n2_btn_score_dataset4_measure"):
                                    try:
                                        df_measure_raw = d4_df[selected_measure_cols].copy()
                                        cleaned_map = {c: _clean_col_for_score(c) for c in df_measure_raw.columns}
                                        df_measure_clean = df_measure_raw.rename(columns=cleaned_map)
                                        row_to_col, _ = _build_row_mapping(
                                            formula_valid,
                                            list(df_measure_clean.columns),
                                            enable_item_num=True,
                                            enable_fuzzy=False,
                                        )
                                        scored_df, unresolved_items, n_missing, err_msg = _score_dataframe_with_mapping(
                                            df_measure_raw, df_measure_clean, formula_valid, intercept, row_to_col
                                        )
                                        if err_msg:
                                            st.error(f"{err_msg}：{', '.join(unresolved_items[:8])}")
                                        else:
                                            st.session_state.n2_formula_scored_measure_df = scored_df
                                            st.session_state.n2_formula_scored_measure_name = selected_measure_for_score
                                            st.session_state.n2_formula_scored_measure_file_prefix = f"{_safe_name_part(selected_measure_for_score)}_dataset4_scored"
                                            st.session_state.n2_formula_scored_measure_csv = scored_df.to_csv(index=False).encode("utf-8-sig")
                                            xlsx_buf = io.BytesIO()
                                            with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w_x:
                                                scored_df.to_excel(w_x, sheet_name="ScoredMeasure", index=False)
                                            xlsx_buf.seek(0)
                                            st.session_state.n2_formula_scored_measure_xlsx = xlsx_buf.getvalue()
                                            if n_missing > 0:
                                                st.warning(f"检测到 {n_missing} 行样本存在缺失值，这些样本的“最终得分”已记为 -999。")
                                            else:
                                                st.info("检测到 0 行样本存在缺失值。")
                                            st.success(f"计分完成：共 {len(scored_df)} 行样本。")
                                    except Exception as e:
                                        st.error(f"按公式为 Dataset4 measure 计分失败: {e}")
                    else:
                        uploaded_score_file = st.file_uploader(
                            "上传用于计分的数据文件（Excel/CSV）",
                            type=["xlsx", "xls", "csv"],
                            key="n2_formula_score_upload_file",
                        )
                        if uploaded_score_file is not None:
                            try:
                                uploaded_df = _read_uploaded_data(uploaded_score_file)
                                if uploaded_df is None or uploaded_df.empty:
                                    st.error("上传文件为空，无法计分。")
                                else:
                                    st.info(f"上传数据已读取：{len(uploaded_df)} 行 × {len(uploaded_df.columns)} 列。")
                                    cleaned_map_upload = {c: _clean_col_for_score(c) for c in uploaded_df.columns}
                                    upload_df_clean = uploaded_df.rename(columns=cleaned_map_upload)

                                    match_mode = st.selectbox(
                                        "题目匹配方式",
                                        ["自动匹配（列名精确）", "智能匹配（题号+文本）", "手动匹配（逐题选择）"],
                                        index=1,
                                        key="n2_upload_match_mode",
                                    )

                                    if match_mode == "自动匹配（列名精确）":
                                        auto_row_to_col, auto_reason = _build_row_mapping(
                                            formula_valid,
                                            list(upload_df_clean.columns),
                                            enable_item_num=False,
                                            enable_fuzzy=False,
                                        )
                                    else:
                                        auto_row_to_col, auto_reason = _build_row_mapping(
                                            formula_valid,
                                            list(upload_df_clean.columns),
                                            enable_item_num=True,
                                            enable_fuzzy=(match_mode == "智能匹配（题号+文本）"),
                                        )

                                    if match_mode != "手动匹配（逐题选择）":
                                        matched_n = len(auto_row_to_col)
                                        total_n = len(formula_valid)
                                        if matched_n < total_n:
                                            st.warning(f"当前已自动匹配 {matched_n}/{total_n} 道题目。")
                                        else:
                                            st.success(f"当前已自动匹配 {matched_n}/{total_n} 道题目。")
                                        with st.expander("查看自动匹配详情", expanded=False):
                                            rows_preview = []
                                            for ridx, r in formula_valid.iterrows():
                                                rows_preview.append({
                                                    "item_num": r.get("item_num"),
                                                    "item_text": str(r.get("item_text", "")),
                                                    "匹配列": auto_row_to_col.get(ridx, ""),
                                                    "匹配方式": auto_reason.get(ridx, ""),
                                                })
                                            st.dataframe(pd.DataFrame(rows_preview), use_container_width=True, hide_index=True)
                                        final_row_to_col = auto_row_to_col
                                    else:
                                        st.caption("请为每道公式题目手动选择上传文件中的对应列。")
                                        smart_suggest, _ = _build_row_mapping(
                                            formula_valid,
                                            list(upload_df_clean.columns),
                                            enable_item_num=True,
                                            enable_fuzzy=True,
                                        )
                                        final_row_to_col = dict(smart_suggest)
                                        unmatched_ridx = [ridx for ridx in formula_valid.index if ridx not in smart_suggest]
                                        show_unmatched_only = st.checkbox(
                                            "仅显示未匹配题目（基于智能建议）",
                                            value=True,
                                            key="n2_upload_manual_only_unmatched",
                                        )
                                        if show_unmatched_only:
                                            rows_to_render = unmatched_ridx
                                        else:
                                            rows_to_render = list(formula_valid.index)

                                        st.info(
                                            f"智能建议已匹配 {len(smart_suggest)}/{len(formula_valid)} 道题；"
                                            f"待你确认/补齐 {len(unmatched_ridx)} 道题。"
                                        )
                                        if show_unmatched_only and not rows_to_render:
                                            st.success("当前无未匹配题目，可直接点击“按公式计算并生成上传数据最终得分”。")

                                        col_options = ["（不选择）"] + list(upload_df_clean.columns)
                                        for ridx in rows_to_render:
                                            r = formula_valid.loc[ridx]
                                            label = f"题号 {r.get('item_num', '')} | {str(r.get('item_text', ''))[:30]}"
                                            default_col = smart_suggest.get(ridx)
                                            default_idx = col_options.index(default_col) if default_col in col_options else 0
                                            sel_col = st.selectbox(
                                                label,
                                                options=col_options,
                                                index=default_idx,
                                                key=f"n2_upload_manual_map_{ridx}",
                                            )
                                            if sel_col == "（不选择）":
                                                if ridx in final_row_to_col:
                                                    del final_row_to_col[ridx]
                                            else:
                                                final_row_to_col[ridx] = sel_col
                                        st.info(f"手动匹配完成：{len(final_row_to_col)}/{len(formula_valid)} 道题目。")

                                    if st.button("按公式计算并生成上传数据最终得分", key="n2_btn_score_uploaded_measure"):
                                        scored_df, unresolved_items, n_missing, err_msg = _score_dataframe_with_mapping(
                                            uploaded_df, upload_df_clean, formula_valid, intercept, final_row_to_col
                                        )
                                        if err_msg:
                                            st.error(f"{err_msg}：{', '.join(unresolved_items[:10])}")
                                        else:
                                            base_name = re.sub(r"\.[^.]+$", "", str(uploaded_score_file.name))
                                            file_prefix = f"{_safe_name_part(base_name)}_scored"
                                            st.session_state.n2_formula_scored_measure_df = scored_df
                                            st.session_state.n2_formula_scored_measure_name = base_name
                                            st.session_state.n2_formula_scored_measure_file_prefix = file_prefix
                                            st.session_state.n2_formula_scored_measure_csv = scored_df.to_csv(index=False).encode("utf-8-sig")
                                            xlsx_buf = io.BytesIO()
                                            with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w_x:
                                                scored_df.to_excel(w_x, sheet_name="ScoredMeasure", index=False)
                                            xlsx_buf.seek(0)
                                            st.session_state.n2_formula_scored_measure_xlsx = xlsx_buf.getvalue()
                                            if n_missing > 0:
                                                st.warning(f"检测到 {n_missing} 行样本存在缺失值，这些样本的“最终得分”已记为 -999。")
                                            else:
                                                st.info("检测到 0 行样本存在缺失值。")
                                            st.success(f"上传数据计分完成：共 {len(scored_df)} 行样本。")
                            except Exception as e:
                                st.error(f"上传文件解析或计分失败: {e}")

                if st.session_state.get("n2_formula_scored_measure_df") is not None:
                    st.markdown("##### 计分结果预览（前20行）")
                    preview_scored = st.session_state.n2_formula_scored_measure_df.head(20).copy()
                    if "relative_final_score" in preview_scored.columns:
                        preview_scored["relative_final_score"] = pd.to_numeric(preview_scored["relative_final_score"], errors="coerce").round(3)
                    if "absolute_final_score" in preview_scored.columns:
                        preview_scored["absolute_final_score"] = pd.to_numeric(preview_scored["absolute_final_score"], errors="coerce").round(3)
                    st.dataframe(preview_scored, use_container_width=True)

                    dl_prefix = st.session_state.get("n2_formula_scored_measure_file_prefix") or (
                        f"{_safe_name_part(st.session_state.get('n2_formula_scored_measure_name') or 'measure')}_scored"
                    )
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        if st.session_state.get("n2_formula_scored_measure_csv"):
                            st.download_button(
                                "⬇️ 下载含最终得分数据（CSV）",
                                data=st.session_state.n2_formula_scored_measure_csv,
                                file_name=f"{dl_prefix}.csv",
                                mime="text/csv",
                                key="n2_download_formula_scored_measure_csv",
                            )
                    with d2:
                        if st.session_state.get("n2_formula_scored_measure_xlsx"):
                            st.download_button(
                                "⬇️ 下载含最终得分数据（Excel）",
                                data=st.session_state.n2_formula_scored_measure_xlsx,
                                file_name=f"{dl_prefix}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="n2_download_formula_scored_measure_xlsx",
                            )
                    with d3:
                        if _DB_SAVE_AVAILABLE and st.button("☁️ 保存最终得分到云端", key="n2_save_scores_cloud"):
                            scored_df = st.session_state.n2_formula_scored_measure_df
                            measure_id = st.session_state.get("n2_formula_scored_measure_name") or ""
                            ok, msg = save_score_records(scored_df, measure_id=measure_id)
                            if ok:
                                st.success(msg)
                            else:
                                st.error(f"保存失败: {msg}")




# ==============================================================================
# 🌟 顶层三大板块隔离调度中心
# ==============================================================================
def render_n2_analysis():
    st.title("模块 2: N2数据分析")

    # 使用 st.tabs 将三大核心分析板块在水平方向彻底隔离
    tab_single_cfa, tab_multi_cfa = st.tabs([
        "1. 自动删题 single factor CFA 板块", 
        "2. 自动删题 multi factor CFA 板块", 
        "3. 最终不删题 EFA 板块"
    ])

    # 板块一：single factor CFA
    with tab_single_cfa:
       render_single_cfa_clean()

    # 板块二：multi factor CFA
    # with tab_multi_cfa:
    #    render_multi_cfa_clean()

