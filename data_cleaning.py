# 模块1:数据清洗(核心逻辑都在这)
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import zipfile
import hashlib

# 导入通用工具函数
from utils import smart_multiselect, normalize_item_text, get_item_columns, sort_item_cols_by_number
import data_cleaning_dual as dc_dual


def _render_dual_mode_cleaning():
    """EFA & CFA 双数据集模式：双上传、分别仅做反应时长筛选与一致性筛选、合并、在合并表上做其余清洗、生成四数据集。"""
    for key in ("df_efa_raw", "df_efa_current", "df_cfa_raw", "df_cfa_current", "dc_merge_done",
                "dc_dataset_full", "dc_measures", "dc_cfa_only_cols", "dc_merged_dataset"):
        if key not in st.session_state:
            st.session_state[key] = None if "df_" in key or key in ("dc_merge_done", "dc_merged_dataset") else {} if key in ("dc_dataset_full", "dc_measures") else []
    for key in ("dc_efa_item_cols", "dc_cfa_item_cols"):
        if key not in st.session_state:
            st.session_state[key] = []
    if "dc_dual_tab" not in st.session_state:
        st.session_state.dc_dual_tab = "EFA 清洗"
    # 应用上一轮设置的下一步骤（必须在 radio 创建前完成，避免 "cannot be modified after widget is instantiated"）
    if "dc_dual_tab_next" in st.session_state:
        st.session_state.dc_dual_tab = st.session_state.dc_dual_tab_next
        del st.session_state.dc_dual_tab_next
    st.subheader("1. 数据导入 (EFA & CFA)")
    col_efa, col_cfa = st.columns(2)
    with col_efa:
        up_efa = st.file_uploader("上传 EFA 数据文件", type=["xlsx", "xls", "csv"], key="upload_efa")
    with col_cfa:
        up_cfa = st.file_uploader("上传 CFA 数据文件", type=["xlsx", "xls", "csv"], key="upload_cfa")

    def _read_file(f):
        if f is None:
            return None
        try:
            data_bytes = f.getvalue()
            bio = io.BytesIO(data_bytes)
            if f.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(bio)
                df = df.drop(df.index[0]).reset_index(drop=True)
            else:
                df = pd.read_csv(bio)
            
            # 导入后立即修剪表头两侧的隐形空格，防止诸如“说过谎。 ”和“说过谎。”匹配失败报错
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            st.error(f"读取失败: {e}")
            return None

    def _file_fp(f):
        b = f.getvalue() if f is not None else b""
        return hashlib.sha1(b).hexdigest()

    def _reset_efa_downstream():
        st.session_state.dc_merged_dataset = None
        st.session_state.dc_merge_done = False
        st.session_state.dc_dataset_full = {}
        st.session_state.dc_measures = {}
        st.session_state.dc_cfa_only_cols = []
        st.session_state.dc_cfa_cols_at_merge = set()
        st.session_state.dc_cfa_item_cols_for_d2d4 = []

    def _reset_cfa_downstream():
        st.session_state.dc_merged_dataset = None
        st.session_state.dc_merge_done = False
        st.session_state.dc_dataset_full = {}
        st.session_state.dc_measures = {}
        st.session_state.dc_cfa_only_cols = []
        st.session_state.dc_cfa_cols_at_merge = set()
        st.session_state.dc_cfa_item_cols_for_d2d4 = []

    # 用户清空 EFA 上传框时：重置 EFA 相关状态，确保后续重新上传会被识别为新文件
    if not up_efa and st.session_state.get("dc_upload_efa_fp") is not None:
        st.session_state.dc_upload_efa_fp = None
        st.session_state.df_efa_raw = None
        st.session_state.df_efa_current = None
        _reset_efa_downstream()

    # 用户清空 CFA 上传框时：同上
    if not up_cfa and st.session_state.get("dc_upload_cfa_fp") is not None:
        st.session_state.dc_upload_cfa_fp = None
        st.session_state.df_cfa_raw = None
        st.session_state.df_cfa_current = None
        _reset_cfa_downstream()

    if up_efa:
        fp_efa = _file_fp(up_efa)
        if st.session_state.get("dc_upload_efa_fp") != fp_efa:
            df = _read_file(up_efa)
            if df is not None:
                st.session_state.dc_upload_efa_fp = fp_efa
                st.session_state.df_efa_raw = df
                st.session_state.df_efa_current = df.copy()
                _reset_efa_downstream()
                st.success(f"EFA 数据已更新导入，共 {len(df)} 行")
    if up_cfa:
        fp_cfa = _file_fp(up_cfa)
        if st.session_state.get("dc_upload_cfa_fp") != fp_cfa:
            df = _read_file(up_cfa)
            if df is not None:
                st.session_state.dc_upload_cfa_fp = fp_cfa
                st.session_state.df_cfa_raw = df
                st.session_state.df_cfa_current = df.copy()
                _reset_cfa_downstream()
                st.success(f"CFA 数据已更新导入，共 {len(df)} 行")

    if st.session_state.df_efa_current is None or st.session_state.df_cfa_current is None:
        st.info("请同时上传 EFA 和 CFA 数据文件后继续。")
        return

    # 步骤条样式：虚线连接、均匀分布、间距加大，与 subheader 字号一致（仅作用于 5 个步骤的 radio）
    st.markdown("""
    <style>
    .stRadio:has(div[role="radiogroup"] > label:nth-child(5)) > div[role="radiogroup"] {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
        gap: 0 !important;
        padding: 0 0.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: none !important;
    }
    .stRadio:has(div[role="radiogroup"] > label:nth-child(5)) > div[role="radiogroup"] > label {
        flex: 1 1 0 !important;
        min-width: 120px !important;
        max-width: 200px !important;
        text-align: center !important;
        position: relative !important;
        padding: 0.6rem 2rem !important;
        margin: 0 0.5rem !important;
        font-size: 1.25rem !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        color: #6b7280 !important;
        cursor: pointer !important;
        white-space: nowrap !important;
    }
    .stRadio:has(div[role="radiogroup"] > label:nth-child(5)) > div[role="radiogroup"] > label:hover {
        color: #374151 !important;
    }
    .stRadio:has(div[role="radiogroup"] > label:nth-child(5)) > div[role="radiogroup"] > label:has(input:checked) {
        color: #ff4b4b !important;
        font-weight: 600 !important;
        border-bottom-color: #ff4b4b !important;
    }
    .stRadio:has(div[role="radiogroup"] > label:nth-child(5)) > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    tab_options = ["EFA 清洗", "CFA 清洗", "数据集合并&清洗", "Measure 划分", "下载"]
    tab_choice = st.radio(
        "步骤",
        tab_options,
        key="dc_dual_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    if tab_choice == "EFA 清洗":
        _render_dual_tab_preview_and_time("EFA", "df_efa_current")
    elif tab_choice == "CFA 清洗":
        _render_dual_tab_preview_and_time("CFA", "df_cfa_current")
    elif tab_choice == "数据集合并&清洗":
        _render_merge_and_four_datasets()
    elif tab_choice == "Measure 划分":
        if not st.session_state.get("dc_merge_done"):
            st.warning("请先在「数据集合并&清洗」中执行合并。")
        else:
            d3 = st.session_state.dc_dataset_full["Dataset3"]
            all_item_cols = get_item_columns(d3)
            if not all_item_cols:
                all_item_cols = [c for c in d3.columns if c not in ("source", "数据来源") and re.match(r"^(EFA|CFA)\d+_", str(c))]
            measure_name = st.text_input("Measure 名称", key="dc_measure_name", placeholder="例如：考试焦虑")
            selected = smart_multiselect(
                options=all_item_cols,
                label="选择属于该 Measure 的题目",
                key_suffix="dc_measure_items",
                default_selected=[],
                show_selection_controls=True,
            )
            if st.button("添加 Measure", key="btn_add_measure") and measure_name and selected:
                st.session_state.dc_measures[measure_name.strip()] = selected
                st.success(f"已添加 Measure「{measure_name}」，共 {len(selected)} 题。")
            if st.session_state.dc_measures:
                st.write("已定义的 Measure：")
                for m, cols in st.session_state.dc_measures.items():
                    st.caption(f"**{m}**: {len(cols)} 题")
                del_m = st.selectbox("删除 Measure", ["(不删除)"] + list(st.session_state.dc_measures.keys()), key="dc_del_measure")
                if del_m != "(不删除)" and st.button("确认删除", key="dc_btn_del_measure"):
                    del st.session_state.dc_measures[del_m]
                    st.rerun()
    else:
        # 下载
        if not st.session_state.get("dc_merge_done"):
            st.warning("请先完成合并与 Measure 划分后再下载。")
        else:
            full = st.session_state.dc_dataset_full
            measures = st.session_state.get("dc_measures") or {}

            def _safe_filename_part(v: str) -> str:
                s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(v or "")).strip(" .")
                return s or "measure"

            dataset_options = [n for n in ["Dataset1", "Dataset2", "Dataset3", "Dataset4"] if n in full]
            if not dataset_options:
                st.warning("未找到可下载的数据集。")
                return

            st.markdown("**下载全部列（Excel）**")
            selected_dataset_all = st.selectbox(
                "选择要下载全部列的数据集",
                dataset_options,
                key="dc_dl_allcols_dataset",
            )
            df_all = full[selected_dataset_all]
            buf_all = io.BytesIO()
            with pd.ExcelWriter(buf_all, engine="xlsxwriter") as w_all:
                df_all.to_excel(w_all, index=False, sheet_name="Data")
            st.download_button(
                f"📥 下载 {selected_dataset_all}（全部列）",
                data=buf_all.getvalue(),
                file_name=f"{_safe_filename_part(selected_dataset_all)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_allcols_{selected_dataset_all}",
            )

            # 按 Measure 拆分下载（Excel）：先选 dataset，再选 measure，只一个下载按钮
            st.markdown("**按 Measure 下载拆分数据集（Excel）**")
            if not measures:
                st.caption("尚未定义 Measure，请先在「Measure 划分」中完成配置。")
            else:
                selected_measure_dataset = st.selectbox(
                    "选择要按 Measure 拆分的数据集",
                    dataset_options,
                    key="dc_dl_measure_dataset",
                )
                df_selected = full[selected_measure_dataset]
                item_cols_selected = get_item_columns(df_selected)
                aux_cols = [c for c in df_selected.columns if c not in item_cols_selected]
                measure_candidates_split = []
                for mname, mcols in measures.items():
                    in_d = [c for c in mcols if c in df_selected.columns]
                    if in_d:
                        measure_candidates_split.append(mname)
                if not measure_candidates_split:
                    st.caption(f"{selected_measure_dataset} 中暂无可拆分的 Measure 题目列。")
                else:
                    selected_split_measure = st.selectbox(
                        "选择要下载的 Measure",
                        measure_candidates_split,
                        key="dc_dl_measure_split_measure",
                    )
                    mcols = measures[selected_split_measure]
                    in_d = [c for c in mcols if c in df_selected.columns]
                    export_cols = list(dict.fromkeys(aux_cols + in_d))
                    sub_df = df_selected[export_cols].copy()
                    m_buf = io.BytesIO()
                    with pd.ExcelWriter(m_buf, engine="xlsxwriter") as w_m:
                        sub_df.to_excel(w_m, sheet_name="Data", index=False)
                    m_buf.seek(0)
                    m_bytes = m_buf.getvalue()
                    d_safe = _safe_filename_part(selected_measure_dataset)
                    m_safe = _safe_filename_part(selected_split_measure)
                    st.download_button(
                        f"📥 下载 {selected_measure_dataset} - {selected_split_measure}（Excel）",
                        data=m_bytes,
                        file_name=f"{d_safe}_{m_safe}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_measure_excel_{selected_measure_dataset}_{m_safe}",
                    )
                    zip_measure_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_measure_buf, "w", zipfile.ZIP_DEFLATED) as zf_m:
                        for mname in measure_candidates_split:
                            mcols_z = measures[mname]
                            in_d_z = [c for c in mcols_z if c in df_selected.columns]
                            export_cols_z = list(dict.fromkeys(aux_cols + in_d_z))
                            sub_df_z = df_selected[export_cols_z].copy()
                            m_buf_z = io.BytesIO()
                            with pd.ExcelWriter(m_buf_z, engine="xlsxwriter") as w_mz:
                                sub_df_z.to_excel(w_mz, sheet_name="Data", index=False)
                            m_buf_z.seek(0)
                            zf_m.writestr(
                                f"{d_safe}/{_safe_filename_part(mname)}.xlsx",
                                m_buf_z.getvalue(),
                            )
                    zip_measure_buf.seek(0)
                    st.download_button(
                        f"📦 打包下载 {selected_measure_dataset} 全部 Measure（Excel）",
                        data=zip_measure_buf.getvalue(),
                        file_name=f"{_safe_filename_part(selected_measure_dataset)}_measures.xlsx.zip",
                        mime="application/zip",
                        key=f"dl_measure_excel_zip_{selected_measure_dataset}",
                    )

            st.markdown("**按 Measure 统计下载（Dataset2 / Dataset4）**")
            if not measures:
                st.caption("尚未定义 Measure，请先在「Measure 划分」中完成配置。")
            else:
                stat_dataset_options = [n for n in ["Dataset2", "Dataset4"] if n in full]
                if not stat_dataset_options:
                    st.caption("当前没有可用于统计下载的 Dataset2 / Dataset4。")
                else:
                    selected_stat_dataset = st.selectbox(
                        "选择统计下载的数据集（仅 Dataset2 / Dataset4）",
                        stat_dataset_options,
                        key="dc_dl_stats_dataset",
                    )
                    df_stat = full[selected_stat_dataset]
                    item_cols_stat = get_item_columns(df_stat)
                    measure_candidates = []
                    for mname, mcols in measures.items():
                        in_d = [c for c in mcols if c in df_stat.columns and c in item_cols_stat]
                        if in_d:
                            measure_candidates.append(mname)
                    if not measure_candidates:
                        st.caption(f"{selected_stat_dataset} 中暂无可统计的 Measure 题目列。")
                    else:
                        selected_stat_measure = st.selectbox(
                            "选择要下载统计表的 Measure",
                            measure_candidates,
                            key="dc_dl_stats_measure",
                        )
                        stat_cols = [
                            c for c in measures.get(selected_stat_measure, [])
                            if c in df_stat.columns and c in item_cols_stat
                        ]
                        stat_cols = sort_item_cols_by_number(stat_cols)
                        mean_sd = dc_dual.item_mean_sd_table(df_stat, stat_cols)
                        cov_m = dc_dual.item_covariance_matrix(df_stat, stat_cols)
                        d_safe = _safe_filename_part(selected_stat_dataset)
                        m_safe = _safe_filename_part(selected_stat_measure)
                        col_ms, col_cov = st.columns(2)
                        with col_ms:
                            st.download_button(
                                f"📥 下载 {selected_stat_dataset} - {selected_stat_measure} 题目均值/标准差",
                                data=mean_sd.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"{d_safe}_{m_safe}_item_mean_sd.csv",
                                mime="text/csv",
                                key=f"dl_{selected_stat_dataset}_{m_safe}_ms_single",
                            )
                        with col_cov:
                            st.download_button(
                                f"📥 下载 {selected_stat_dataset} - {selected_stat_measure} 协方差矩阵",
                                data=cov_m.to_csv().encode("utf-8-sig"),
                                file_name=f"{d_safe}_{m_safe}_covariance.csv",
                                mime="text/csv",
                                key=f"dl_{selected_stat_dataset}_{m_safe}_cov_single",
                            )

            if st.button("📦 下载全部（ZIP 文件夹结构）", key="btn_zip_all"):
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # 1) 各 Dataset 全部列（Excel）
                    for name, d in full.items():
                        all_buf = io.BytesIO()
                        with pd.ExcelWriter(all_buf, engine="xlsxwriter") as w_all:
                            d.to_excel(w_all, index=False, sheet_name="Data")
                        all_buf.seek(0)
                        zf.writestr(f"all_columns/{_safe_filename_part(name)}.xlsx", all_buf.getvalue())

                    # 2) 各 Dataset 按 Measure 拆分（Excel，保留辅助列）
                    for name, d in full.items():
                        item_cols_d = get_item_columns(d)
                        aux_cols_d = [c for c in d.columns if c not in item_cols_d]
                        for mname, mcols in measures.items():
                            in_d = [c for c in mcols if c in d.columns]
                            if not in_d:
                                continue
                            export_cols = list(dict.fromkeys(aux_cols_d + in_d))
                            sub_df = d[export_cols].copy()
                            m_buf = io.BytesIO()
                            with pd.ExcelWriter(m_buf, engine="xlsxwriter") as w_m:
                                sub_df.to_excel(w_m, index=False, sheet_name="Data")
                            m_buf.seek(0)
                            zf.writestr(
                                f"measure_split/{_safe_filename_part(name)}/{_safe_filename_part(mname)}.xlsx",
                                m_buf.getvalue(),
                            )

                    # 3) 按 Measure 统计（仅 Dataset2 / Dataset4，CSV）
                    for name in ["Dataset2", "Dataset4"]:
                        if name not in full:
                            continue
                        d = full[name]
                        item_cols_d = get_item_columns(d)
                        for mname, mcols in measures.items():
                            in_d = [c for c in mcols if c in d.columns and c in item_cols_d]
                            if not in_d:
                                continue
                            in_d = sort_item_cols_by_number(in_d)
                            mean_sd = dc_dual.item_mean_sd_table(d, in_d)
                            cov_m = dc_dual.item_covariance_matrix(d, in_d)
                            base = f"measure_stats/{_safe_filename_part(name)}/{_safe_filename_part(mname)}"
                            zf.writestr(f"{base}_item_mean_sd.csv", mean_sd.to_csv(index=False).encode("utf-8-sig"))
                            zf.writestr(f"{base}_covariance.csv", cov_m.to_csv().encode("utf-8-sig"))
                zip_buf.seek(0)
                st.download_button(
                    "📥 保存 ZIP",
                    data=zip_buf.getvalue(),
                    file_name="EFA_CFA_datasets.zip",
                    mime="application/zip",
                    key="dl_zip",
                )

    st.markdown("---")
    def _go_n1():
        st.session_state.nav_selection = "2. N1 EFA数据分析"
    st.button("前往 N1 模块进行分析 ->", type="primary", use_container_width=True, on_click=_go_n1)


@st.fragment
def _render_dual_tab_preview_and_time(role: str, state_key: str):
    """双数据集模式下 EFA/CFA 页签：仅保留数据预览与反应时长筛选，一致性筛选。用 @st.fragment 隔离 widget 交互。"""
    df = getattr(st.session_state, state_key, None)
    if df is None:
        return
    feedback_key = f"dc_time_filter_feedback_dual_{role}"
    if st.session_state.get(feedback_key):
        st.success(st.session_state.get(feedback_key))
        del st.session_state[feedback_key]
    st.write(f"**{role} 数据**：{df.shape[0]} 行 × {df.shape[1]} 列")
    with st.expander("数据预览"):
        st.dataframe(
            _safe_df_for_preview(df.head()),
            key=f"preview_dual_{role}_{len(df.columns)}_{hash(tuple(df.columns))}"
        )
    st.markdown("###### 反应时长筛选 (3SD)")
    time_col_name = "作答总时长(秒)"
    time_col = st.selectbox(
        f"选择作答时长列（{role}）",
        df.columns,
        index=df.columns.get_loc(time_col_name) if time_col_name in df.columns else 0,
        key=f"time_col_dual_{role}",
    )
    if st.button(f"执行反应时长筛选（{role}）", key=f"btn_time_dual_{role}"):
        try:
            ser = pd.to_numeric(
                st.session_state[state_key][time_col].astype(str).str.replace("秒", "", regex=False),
                errors="coerce",
            )
            valid = ser.dropna()
            if len(valid) > 0:
                avg, sd = valid.mean(), valid.std()
                lb = max(0, avg - 1.96 * sd)
                ub = avg + 3 * sd
                before_count = len(st.session_state[state_key])
                mask = (ser >= lb) & (ser <= ub)
                st.session_state[state_key] = st.session_state[state_key].loc[mask].reset_index(drop=True)
                after_count = len(st.session_state[state_key])
                st.session_state[feedback_key] = f"筛选完成：已删除 {before_count - after_count} 行，保留区间 [{lb:.2f}, {ub:.2f}]，剩余 {after_count} 行。"
            st.rerun(scope="app")
        except Exception as e:
            st.error(f"处理出错: {e}")

    # ==========================================================
    # 一致性筛选
    # ==========================================================
    st.markdown("###### 作答一致性筛选 (Straight-lining)")

    col1, col2 = st.columns(2)

    with col1:
        start_col = st.selectbox(
            f"选择第一个题项列（{role}）",
            options=df.columns.tolist(),
            key=f"cons_start_col_{role}",
        )

    with col2:
        start_idx_default = df.columns.get_loc(start_col)

        end_col = st.selectbox(
            f"选择最后一个题项列（{role}）",
            options=df.columns.tolist(),
            index=max(start_idx_default, len(df.columns) - 1),
            key=f"cons_end_col_{role}",
        )

    ratio = st.slider(
        f"一致性阈值（{role}）",
        min_value=0.50,
        max_value=1.00,
        value=0.90,
        step=0.05,
        key=f"cons_ratio_{role}",
    )

    try:
        start_idx = df.columns.get_loc(start_col)
        end_idx = df.columns.get_loc(end_col)

        if start_idx > end_idx:
            st.warning("起始列不能位于结束列之后")
        else:
            item_cols = df.columns[start_idx:end_idx + 1].tolist()

            st.caption(
                f"当前用于一致性检测的题项范围："
                f"{item_cols[0]} ～ {item_cols[-1]}"
                f"（共 {len(item_cols)} 列）"
            )

    except Exception:
        item_cols = []

    if st.button(
        f"执行一致性筛选（{role}）",
        key=f"btn_consistency_dual_{role}",
    ):
        try:

            start_idx = df.columns.get_loc(start_col)
            end_idx = df.columns.get_loc(end_col)

            if start_idx > end_idx:
                st.error("起始列不能位于结束列之后")
                return

            item_cols = df.columns[start_idx:end_idx + 1].tolist()

            num_df = (
                st.session_state[state_key][item_cols]
                .apply(pd.to_numeric, errors="coerce")
            )

            non_null_counts = num_df.notna().sum(axis=1)

            max_counts = num_df.apply(
                lambda x: (
                    x.value_counts(dropna=True).max()
                    if x.notna().any()
                    else 0
                ),
                axis=1,
            )

            mask_drop = (
                (non_null_counts > 0)
                & ((max_counts / non_null_counts) >= ratio)
            )

            before_count = len(st.session_state[state_key])

            st.session_state[state_key] = (
                st.session_state[state_key]
                .loc[~mask_drop]
                .reset_index(drop=True)
            )

            after_count = len(st.session_state[state_key])

            st.session_state[cons_feedback_key] = (
                f"一致性筛选完成："
                f"已删除 {before_count - after_count} 行，"
                f"剩余 {after_count} 行。"
            )

            st.rerun(scope="app")

        except Exception as e:
            st.error(f"处理出错: {e}")





def _sync_smart_multiselect_after_rename(key_suffix: str, rename_map: dict):
    """列名重命名后，同步 smart_multiselect 缓存，避免 checkbox 显示旧列名。"""
    if not rename_map:
        return
    # 删除所有 checkbox key，强制下次渲染用新列名重建
    cb_prefix = f"cb_{key_suffix}_"
    for k in list(st.session_state.keys()):
        if k.startswith(cb_prefix):
            del st.session_state[k]
    # 保留用户已选项：按 rename_map 映射到新列名
    last_selected_key = f"{key_suffix}_last_selected"
    prev_selected = st.session_state.get(last_selected_key, [])
    if isinstance(prev_selected, list):
        mapped = [rename_map.get(c, c) for c in prev_selected]
        st.session_state[last_selected_key] = list(dict.fromkeys(mapped))


def _clear_editor_cache():
    """清理 session_state 中所有的 smart_multiselect 缓存 (cb_ checkbox keys)，防止列名变更后显示旧状态。
    应在 DataFrame 的列名（题干）发生任何结构性变更且调用 st.rerun() 前执行它。
    """
    keys_to_delete = [
        k for k in st.session_state.keys()
        if k.startswith("cb_") or k.endswith("_control_action")
    ]
    for k in keys_to_delete:
        st.session_state.pop(k, None)


def _ensure_unique_columns(df):
    """【列名重复保险】检测并修复重名列，给重复项加后缀。确保 PyArrow 渲染和一系列分析函数不报错。"""
    if df is None: return None
    if len(df.columns) == len(set(df.columns)):
        return df
    
    new_cols = []
    counts = {}
    for col in df.columns:
        c_str = str(col)
        if c_str in counts:
            counts[c_str] += 1
            new_cols.append(f"{c_str}_{counts[c_str]}")
        else:
            counts[c_str] = 0
            new_cols.append(c_str)
    
    df_fixed = df.copy()
    df_fixed.columns = new_cols
    return df_fixed


def _safe_df_for_preview(df):
    """确保 DataFrame 预览时没有重复的列名，防止 Streamlit (PyArrow) 抛出 Duplicate column names 报错。"""
    return _ensure_unique_columns(df)


def _is_missing_val(v):
    """检测是否为缺失值（NaN, 空字符串, None, 'nan' 等）。"""
    if pd.isna(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _normalize_answer(v):
    """将答案尽可能转数字以兼容不同的数据类型（1 与 1.0 的安全匹配）。"""
    if _is_missing_val(v):
        return ""
    s = str(v).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return s.lower()


def _render_merge_and_four_datasets():
    """合并与四数据集：合并(加数据来源) → 在合并表上做其余清洗 → 生成 d1～d4 → 数据预览。"""
    efa_df = st.session_state.df_efa_current
    cfa_df = st.session_state.df_cfa_current
    if efa_df is None or cfa_df is None:
        st.warning("请先在 EFA 清洗、CFA 清洗中完成数据预览与反应时长筛选。")
        return

    st.markdown("##### 1. 合并为 merged_dataset（新增列「数据来源」）")
    if st.button("执行合并", key="btn_merge_dual"):
        df_efa = st.session_state.df_efa_current.copy()
        df_cfa = st.session_state.df_cfa_current.copy()
        source_col_name = "数据来源"
        # 必须先记录 CFA 原始列（在添加数据来源列和对齐列之前）
        cfa_original_cols = list(df_cfa.columns)
        df_efa[source_col_name] = "EFA"
        df_cfa[source_col_name] = "CFA"
        all_cols = list(dict.fromkeys(list(df_efa.columns) + list(df_cfa.columns)))
        for c in all_cols:
            if c not in df_efa.columns:
                df_efa[c] = np.nan
            if c not in df_cfa.columns:
                df_cfa[c] = np.nan
        merged = pd.concat([df_efa[all_cols], df_cfa[all_cols]], axis=0, ignore_index=True)
        st.session_state.dc_merged_dataset = merged
        st.session_state.dc_merge_done = False
        # 记录 CFA 原始列名（排除数据来源列），用于生成 D2/D4 时只保留 CFA 题目列
        st.session_state.dc_cfa_cols_at_merge = set(cfa_original_cols)
        st.session_state.dc_cfa_item_cols_for_d2d4 = [c for c in cfa_original_cols if c in merged.columns]
        st.success(f"已合并为 merged_dataset：{merged.shape[0]} 行 × {merged.shape[1]} 列，含「数据来源」列。")

    merged = st.session_state.get("dc_merged_dataset")
    if merged is not None:
        # 将预览功能放在这里，常驻在合并按钮下方
        st.write(f"**merged_dataset (预览)**：{merged.shape[0]} 行 × {merged.shape[1]} 列")
        with st.expander("点击展开/折叠合并后的预览", expanded=True):
            st.dataframe(
                _safe_df_for_preview(merged.head()),
                key=f"preview_merge_top_{len(merged.columns)}_{hash(tuple(merged.columns))}"
            )

        st.markdown("##### 2. 对 merged_dataset 进行其余数据清洗与筛选")
        st.caption("在合并后的数据上执行：批量删字、给题目添加序号、注意力/诚实性、IP、一致性、反向计分等。")
        _render_cleaning_on_merged()

    if st.session_state.get("dc_merge_done") and st.session_state.get("dc_dataset_full"):
        st.markdown("##### 3. 四份数据集预览")
        for name in ["Dataset1", "Dataset2", "Dataset3", "Dataset4"]:
            if name not in st.session_state.dc_dataset_full:
                continue
            d = st.session_state.dc_dataset_full[name]
            with st.expander(f"**{name}**：{d.shape[0]} 行 × {d.shape[1]} 列（点击展开预览）"):
                st.dataframe(
                    _safe_df_for_preview(d.head()),
                    key=f"preview_four_{name}_{len(d.columns)}_{hash(tuple(d.columns))}"
                )


def _render_cleaning_on_merged():
    """在 dc_merged_dataset 上执行：批量删字、注意力/诚实性、IP、一致性、反向计分。生成四数据集后写入 dc_dataset_full。"""
    merged = st.session_state.dc_merged_dataset
    if merged is None:
        return
    state_key = "dc_merged_dataset"
    current_df = st.session_state[state_key]
    
    st.markdown("###### 批量删除指定文字")
    text_remove = st.text_input("批量删除文字", key="merge_text_remove")
    if st.button("执行删除", key="btn_merge_remove") and text_remove:
        rename = {c: c.replace(text_remove, "").strip() or c for c in st.session_state[state_key].columns if text_remove in str(c)}
        if rename:
            # 执行重命名并立即进行“重名保险”检查
            st.session_state[state_key] = _ensure_unique_columns(st.session_state[state_key].rename(columns=rename))
            # 同步更新「CFA 题目列」列表
            _dc = st.session_state.get("dc_cfa_item_cols_for_d2d4") or []
            _cols = st.session_state[state_key].columns.tolist()
            st.session_state.dc_cfa_item_cols_for_d2d4 = [rename.get(c, c) for c in _dc if rename.get(c, c) in _cols]
            st.success(f"已从 {len(rename)} 个列中删除指定文字。")
            _clear_editor_cache()
            st.rerun()

    st.markdown("###### 给题目添加序号（EFA_数字_item text）")
    df = st.session_state[state_key]
    source_col_name = "数据来源"
    cols_excl_source = [c for c in df.columns if c != source_col_name]
    # 已符合 EFA_数字_ 或 EFA数字_ 的列不再重新编号
    cols_to_number = [c for c in cols_excl_source if not re.match(r"^EFA_?\d+_", str(c))]
    if cols_to_number:
        start_col = st.selectbox(
            "从哪一列开始编号（该列及之后的列将改为 EFA_1_xxx, EFA_2_xxx, ...）",
            options=df.columns.tolist(),
            key="merge_start_col_efa",
        )
        if st.button("为题目添加序号", key="btn_merge_assign_efa"):
            all_cols = df.columns.tolist()
            start_idx = all_cols.index(start_col) if start_col in all_cols else 0
            rename_map = {}
            num = 1
            for col in all_cols[start_idx:]:
                if col == source_col_name:
                    continue
                rename_map[col] = f"EFA_{num}_{col}"
                num += 1
            if rename_map:
                # 执行重命名并立即进行“重名保险”检查
                st.session_state[state_key] = _ensure_unique_columns(st.session_state[state_key].rename(columns=rename_map))
                # 同步更新「CFA 题目列」列表，供生成 D2/D4 时只保留 CFA 列
                _dc = st.session_state.get("dc_cfa_item_cols_for_d2d4") or []
                _cols = st.session_state[state_key].columns.tolist()
                st.session_state.dc_cfa_item_cols_for_d2d4 = [rename_map.get(c, c) for c in _dc if rename_map.get(c, c) in _cols]
                st.success(f"已为 {len(rename_map)} 个题目列添加序号。")
                _clear_editor_cache()
                st.rerun()
    else:
        st.caption("当前列名已均为 EFA_数字_ 格式，或无可编号列。")


    # ==========================================================
    # 注意力与诚实性检查
    # 规则：
    # 1. 正确 -> 保留
    # 2. 空值（结构性缺失）-> 保留
    # 3. 非空且答错 -> 删除
    # ==========================================================

    st.markdown("###### 注意力与诚实性检查")

    df = st.session_state[state_key]

    # ---------- 注意力题 ----------
    num_attention = st.number_input(
        "注意力检查题数量",
        min_value=0,
        value=0,
        step=1,
        key="merge_num_att",
    )

    attention_configs = []

    for i in range(num_attention):

        item_name = st.selectbox(
            f"注意力题 {i+1}",
            df.columns,
            key=f"merge_att_q_{i}",
        )

        ans_str = st.text_input(
            f"正确答案 {i+1}（多个答案用 / 分隔）",
            key=f"merge_att_a_{i}",
        )

        attention_configs.append(
            {
                "name": item_name,
                "answers": [
                    a.strip()
                    for a in (ans_str or "").split("/")
                    if a.strip()
                ],
            }
        )

    # ---------- 诚实性题 ----------
    num_honesty = st.number_input(
        "诚实性检查题数量",
        min_value=0,
        value=0,
        step=1,
        key="merge_num_hon",
    )

    honesty_configs = []

    for i in range(num_honesty):

        item_name = st.selectbox(
            f"诚实题 {i+1}",
            df.columns,
            key=f"merge_hon_q_{i}",
        )

        ans_str = st.text_input(
            f"正确答案 {i+1}（多个答案用 / 分隔）",
            key=f"merge_hon_a_{i}",
        )

        honesty_configs.append(
            {
                "name": item_name,
                "answers": [
                    a.strip()
                    for a in (ans_str or "").split("/")
                    if a.strip()
                ],
            }
        )


    def _vectorized_check(df_in, check_list):
        """
        检查规则：

        正确 -> 保留
        空值 -> 保留
        非空且错误 -> 删除

        适用于 EFA/CFA 合并后的结构性缺失。
        """

        mask = pd.Series(True, index=df_in.index)

        for check in check_list:

            name = check.get("name")

            if name not in df_in.columns:
                continue

            normalized_answers = [
                _normalize_answer(a)
                for a in check.get("answers", [])
            ]

            is_missing = df_in[name].apply(_is_missing_val)

            normalized_col = df_in[name].apply(_normalize_answer)

            is_correct = normalized_col.isin(normalized_answers)

            # 空值 OR 正确 => 保留
            mask &= (is_missing | is_correct)

        return mask


    if st.button("执行题目筛选清洗"):

        try:

            df_check = st.session_state[state_key]

            before_count = len(df_check)

            # 注意力检查
            att_mask = _vectorized_check(
                df_check,
                attention_configs,
            )

            # 诚实性检查
            hon_mask = _vectorized_check(
                df_check,
                honesty_configs,
            )

            # 两者同时满足
            mask_keep = att_mask & hon_mask

            st.session_state[state_key] = (
                df_check.loc[mask_keep]
                .reset_index(drop=True)
            )

            after_count = len(st.session_state[state_key])

            st.session_state["fb_merge_att"] = (
                f"筛选完成：已删除 {before_count - after_count} 行，"
                f"剩余 {after_count} 行。"
            )

            st.rerun()

        except Exception as e:
            st.error(f"处理出错：{e}")


    if st.session_state.get("fb_merge_att"):
        st.success(st.session_state.pop("fb_merge_att"))


    st.markdown("###### IP 地址筛选")
    df = st.session_state[state_key]
    ip_col = st.selectbox("选择 IP 列", df.columns, index=df.columns.get_loc("IP") if "IP" in df.columns else 0, key="merge_ip_col")
    if st.button("执行相同 IP 去重"):
        before_count = len(st.session_state[state_key])
        st.session_state[state_key] = st.session_state[state_key].drop_duplicates(subset=ip_col, keep="first")
        after_count = len(st.session_state[state_key])
        st.session_state["fb_merge_ip"] = f"筛选完成：已删除 {before_count - after_count} 行，剩余 {after_count} 行。"
        st.rerun()
    if st.session_state.get("fb_merge_ip"):
        st.success(st.session_state.pop("fb_merge_ip"))

    

    st.markdown("###### 反向计分 (Reverse Coding)")
    need_rc = st.checkbox("是否需要进行反向计分", key="merge_rc")
    if need_rc:
        rc_cols = smart_multiselect(
            options=st.session_state[state_key].columns.tolist(),
            label="选择需要反向计分的题目",
            key_suffix="merge_rc",
        )
        scale = st.radio("问卷量表类型", (5, 7), index=1, key="merge_scale")
        if st.button("执行反向计分") and rc_cols:
            m = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1} if scale == 5 else {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
            m.update({str(k): v for k, v in m.items()})
            rename_after_rc = {}
            for col in rc_cols:
                st.session_state[state_key][col] = pd.to_numeric(st.session_state[state_key][col], errors="coerce").replace(m)
                if str(col).rstrip().endswith("r"):
                    rename_after_rc[col] = str(col).rstrip()[:-1]
                else:
                    rename_after_rc[col] = f"{col}r"
            if rename_after_rc:
                st.session_state[state_key] = _ensure_unique_columns(st.session_state[state_key].rename(columns=rename_after_rc))
                _dc = st.session_state.get("dc_cfa_item_cols_for_d2d4") or []
                _cols = st.session_state[state_key].columns.tolist()
                st.session_state.dc_cfa_item_cols_for_d2d4 = [rename_after_rc.get(c, c) for c in _dc if rename_after_rc.get(c, c) in _cols]
                _sync_smart_multiselect_after_rename("merge_rc", rename_after_rc)
            st.success("反向计分完成。")
            _clear_editor_cache()
            st.rerun()


    st.markdown("##### 生成四份数据集")
    st.caption("Dataset1: cleaned EFA with all items")
    st.caption("Dataset2: cleaned CFA with CFA items only")
    st.caption("Dataset3: cleaned merged, all items")
    st.caption("Dataset4: cleaned merged, CFA items only")

    source_col_name = "数据来源"
    merged_df = st.session_state[state_key]
    all_item_cols = [c for c in get_item_columns(merged_df) if c in merged_df.columns]
    if all_item_cols and source_col_name in merged_df.columns:
        if st.button("生成 Dataset1～4", key="btn_build_four_from_merged"):
            try:
                # Dataset2/D4 只保留「CFA 题目列」：必须是原始 CFA 列且 CFA 行中有实际数据
                cfa_mask = (merged_df[source_col_name].astype(str).str.strip() == "CFA")
                # 获取原始 CFA 列（随重命名更新后的）
                original_cfa_cols = st.session_state.get("dc_cfa_item_cols_for_d2d4") or []
                # 严格判断：列必须在原始 CFA 列表中，且 CFA 行中有非空数据
                def _cfa_has_real_data(col):
                    if col not in original_cfa_cols:
                        return False  # 不是原始 CFA 列，排除
                    s = merged_df.loc[cfa_mask, col]
                    if s.empty:
                        return False
                    # 检查是否有非空且非 NaN 的值（排除填充的 NaN）
                    s = s.dropna()
                    if len(s) == 0:
                        return False
                    # 进一步检查是否都是空字符串或"nan"
                    s_str = s.astype(str).str.strip()
                    has_real = (s_str != "").any() and (s_str.str.lower() != "nan").any()
                    return has_real
                cfa_item_cols = [c for c in all_item_cols if _cfa_has_real_data(c)]
                # 调试信息（开发时可显示）
                st.caption(f"识别到 CFA 题目列：{len(cfa_item_cols)} 列")
                d1, d2, d3, d4 = dc_dual.build_four_datasets_from_merged(
                    st.session_state[state_key],
                    all_item_cols,
                    cfa_item_cols,
                    source_col=source_col_name,
                )
                st.session_state.dc_dataset_full = {
                    "Dataset1": d1, "Dataset2": d2, "Dataset3": d3, "Dataset4": d4,
                }
                st.session_state.dc_merge_done = True
                # 用临时变量记录下一步骤，下一轮运行开头再应用到 dc_dual_tab，避免 widget key 冲突
                st.session_state.dc_dual_tab_next = "数据集合并&清洗"
                st.success("已生成 Dataset1～4，请到「Measure 划分」定义量表。")
                _clear_editor_cache()
                st.rerun()
            except Exception as e:
                st.error(str(e))


def _render_one_dataset_cleaning(role: str, state_key: str, item_cols_key: str, assign_efa_after: bool):
    """单侧（EFA 或 CFA）清洗：删文字、添加序号(仅EFA)、注意力/诚实性、IP、时长、一致性、反向计分(列名+r)；EFA 可赋 EFA 题号，CFA 可对齐到 EFA。"""
    df = getattr(st.session_state, state_key, None)
    if df is None:
        return
    feedback_key = f"dc_time_filter_feedback_single_{role}"
    if st.session_state.get(feedback_key):
        st.success(st.session_state.get(feedback_key))
        del st.session_state[feedback_key]
    st.write(f"**{role} 数据**：{df.shape[0]} 行 × {df.shape[1]} 列")
    with st.expander("数据预览"):
        st.dataframe(
            _safe_df_for_preview(df.head()),
            key=f"preview_single_{role}_{len(df.columns)}_{hash(tuple(df.columns))}"
        )

    # ---------- 1. 批量删除文字 ----------
    st.markdown("###### 🧹 预处理 - 批量删除指定文字")
    text_remove = st.text_input(f"批量删除文字（{role}）", key=f"text_remove_{role}")
    if st.button(f"执行删除", key=f"btn_remove_{role}") and text_remove:
        rename = {c: c.replace(text_remove, "").strip() or c for c in df.columns if text_remove in c}
        if rename:
            # 执行重命名并立即进行“重名保险”检查
            st.session_state[state_key] = _ensure_unique_columns(st.session_state[state_key].rename(columns=rename))
            st.success(f"已从 {len(rename)} 个列中删除指定文字。")
            _clear_editor_cache()
            st.rerun()

    # ---------- 2. 添加序号 / CFA 对齐 ----------
    if assign_efa_after:
        # EFA：指定起始题目 + 为 EFA 添加题号（放在一起）
        st.markdown("###### 🔢 添加 EFA 题号")
        st.caption("可选择从哪一列开始编号；不勾选则从第一列开始。")
        do_start = st.checkbox(f"是否指定起始题目再添加 EFA 题号？（{role}）", value=False, key=f"do_start_{role}")
        start_col = None
        if do_start:
            all_cols = st.session_state[state_key].columns.tolist()
            start_col = st.selectbox(f"请选择起始题目的名称（{role}）", all_cols, key=f"start_item_{role}")
        use_start = do_start and start_col
        if st.button("为 EFA 题目添加 EFA 题号 (EFA1_, EFA2_, ...)", key="btn_assign_efa"):
            current = st.session_state[state_key]
            item_cols = [c for c in current.columns if isinstance(c, str) and not re.match(r"^(EFA|CFA)\d+_", c)]
            if not item_cols:
                st.warning("未检测到可编号的题目列（可能已有 EFA/CFA 前缀）。")
            else:
                st.session_state[state_key] = dc_dual.assign_efa_item_numbers(current, item_cols, start_col=start_col if use_start else None)
                st.session_state[item_cols_key] = get_item_columns(st.session_state[state_key])
                st.success("已添加 EFA 题号。")
                _clear_editor_cache()
                st.rerun()
    else:
        # CFA：将 CFA 题目对齐到 EFA 题号（放在注意力诚实性检查前面）
        st.markdown("###### 将 CFA 题目对齐到 EFA 题号")
        st.caption("可选择从哪一列开始对齐；不勾选则从第一列开始。")
        do_start_cfa = st.checkbox(f"是否指定起始题目再对齐？（{role}）", value=False, key=f"do_start_cfa_{role}")
        start_col_cfa = None
        if do_start_cfa and get_item_columns(st.session_state.df_efa_current):
            all_cols_cfa = st.session_state[state_key].columns.tolist()
            start_col_cfa = st.selectbox(f"请选择起始题目的名称（{role}）", all_cols_cfa, key=f"start_item_cfa_{role}")
        if get_item_columns(st.session_state.df_efa_current):
            if st.button("将 CFA 题目对齐到 EFA 题号（未匹配的用 CFA1_, CFA2_, ...）", key="btn_align_cfa"):
                efa_cols = get_item_columns(st.session_state.df_efa_current)
                efa_text_to_col = dc_dual.build_efa_text_to_col(efa_cols)
                df_cfa, matched, cfa_only = dc_dual.align_cfa_to_efa(
                    st.session_state[state_key], efa_cols, efa_text_to_col,
                    start_col=start_col_cfa if (do_start_cfa and start_col_cfa) else None,
                )
                st.session_state[state_key] = df_cfa
                st.session_state[item_cols_key] = get_item_columns(df_cfa)
                st.session_state["dc_cfa_only_cols"] = cfa_only
                st.success(f"已对齐：匹配 EFA {len(matched)} 题，CFA 独有 {len(cfa_only)} 题。")
                st.rerun()
        else:
            st.caption("请先在 EFA 清洗中执行「为 EFA 题目添加 EFA 题号」后再对齐 CFA。")

    # ---------- 3. 注意力与诚实性检查 ----------
    st.markdown("###### 筛选：注意力与诚实性检查")
    fb_key_att = f"fb_single_att_{role}"
    df = st.session_state[state_key]
    num_attention = st.number_input(f"注意力检查题数量（{role}）", min_value=0, value=0, step=1, key=f"num_att_{role}")
    attention_configs = []
    if num_attention > 0:
        for i in range(num_attention):
            col_a, col_b = st.columns(2)
            with col_a:
                item_name = st.selectbox(f"注意力题 {i+1}（{role}）", df.columns, key=f"att_q_{role}_{i}")
            with col_b:
                ans_str = st.text_input(f"正确答案 {i+1}（用 / 分隔）", key=f"att_a_{role}_{i}")
            attention_configs.append({"name": item_name, "answers": [a.strip() for a in (ans_str or "").split("/") if a.strip()]})
    num_honesty = st.number_input(f"诚实性检查题数量（{role}）", min_value=0, value=0, step=1, key=f"num_hon_{role}")
    honesty_configs = []
    if num_honesty > 0:
        for i in range(num_honesty):
            col_a, col_b = st.columns(2)
            with col_a:
                item_name = st.selectbox(f"诚实题 {i+1}（{role}）", df.columns, key=f"hon_q_{role}_{i}")
            with col_b:
                ans_str = st.text_input(f"正确答案 {i+1}（用 / 分隔）", key=f"hon_a_{role}_{i}")
            honesty_configs.append({"name": item_name, "answers": [a.strip() for a in (ans_str or "").split("/") if a.strip()]})
    if st.button(f"执行题目筛选清洗（{role}）", key=f"btn_filter_{role}"):
        # 向量化检查：对每道注意力/诚实题，构建列级布尔掩码后合并
        df_check = st.session_state[state_key]
        before_count = len(df_check)
        mask_keep = pd.Series(True, index=df_check.index)

        for check in attention_configs + honesty_configs:
            name = check.get("name")
            if name not in df_check.columns:
                continue
            is_missing = df_check[name].apply(_is_missing_val)
            normalized_col = df_check[name].apply(_normalize_answer)
            normalized_answers = [_normalize_answer(a) for a in check["answers"]]
            is_correct = normalized_col.isin(normalized_answers)
            mask_keep &= (~is_missing & is_correct)

        st.session_state[state_key] = df_check[mask_keep].reset_index(drop=True)
        after_count = len(st.session_state[state_key])
        st.rerun()
    if st.session_state.get(fb_key_att):
        st.success(st.session_state.pop(fb_key_att))

    # ---------- 4. IP 地址筛选 ----------
    st.markdown("###### IP 地址筛选")
    fb_key_ip = f"fb_single_ip_{role}"
    df = st.session_state[state_key]
    ip_col = st.selectbox(f"选择 IP 列（{role}）", df.columns, index=df.columns.get_loc("IP") if "IP" in df.columns else 0, key=f"ip_col_{role}")
    if st.button(f"执行相同 IP 去重（{role}）", key=f"btn_ip_{role}"):
        before_count = len(st.session_state[state_key])
        st.session_state[state_key] = st.session_state[state_key].drop_duplicates(subset=ip_col, keep="first")
        after_count = len(st.session_state[state_key])
        st.rerun()
    if st.session_state.get(fb_key_ip):
        st.success(st.session_state.pop(fb_key_ip))

    # ---------- 5. 反应时长筛选 ----------
    st.markdown("###### 反应时长筛选 (3SD)")
    df = st.session_state[state_key]
    time_col_name = "作答总时长(秒)"
    time_col = st.selectbox(f"选择作答时长列（{role}）", df.columns, index=df.columns.get_loc(time_col_name) if time_col_name in df.columns else 0, key=f"time_col_{role}")
    if st.button(f"执行反应时长筛选（{role}）", key=f"btn_time_{role}"):
        try:
            ser = pd.to_numeric(st.session_state[state_key][time_col].astype(str).str.replace("秒", "", regex=False), errors="coerce")
            valid = ser.dropna()
            if len(valid) > 0:
                avg, sd = valid.mean(), valid.std()
                lb = max(0, avg - 1.96 * sd)
                ub = avg + 3 * sd
                before_count = len(st.session_state[state_key])
                mask = (ser >= lb) & (ser <= ub)
                st.session_state[state_key] = st.session_state[state_key].loc[mask].reset_index(drop=True)
                after_count = len(st.session_state[state_key])
                st.session_state[feedback_key] = f"筛选完成：已删除 {before_count - after_count} 行，保留区间 [{lb:.2f}, {ub:.2f}]，剩余 {after_count} 行。"
            st.rerun()
        except Exception as e:
            st.error(f"处理出错: {e}")

    # ---------- 6. 作答一致性筛选 (Straight-lining) ----------
    st.markdown("###### 作答一致性筛选 (Straight-lining)")
    fb_key_cons = f"fb_single_cons_{role}"
    ratio = st.slider(f"一致性阈值（{role}）", 0.5, 1.0, 0.9, key=f"ratio_{role}")
    if st.button(f"执行一致性筛选（{role}）", key=f"btn_consistency_{role}"):
        df_temp = st.session_state[state_key].copy()
        # 逐列转换为数值类型，errors='coerce' 将无法转换的值设为 NaN
        for col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        num_df = df_temp.select_dtypes(include=[np.number])
        before_count = len(st.session_state[state_key])
        # 向量化一致性筛选：计算每行众数占比
        non_null_counts = num_df.notna().sum(axis=1)
        mode_vals = num_df.mode(axis=1).iloc[:, 0] if not num_df.empty else pd.Series(dtype=float)
        mode_counts = num_df.eq(mode_vals, axis=0).sum(axis=1)
        mask_drop = (non_null_counts > 0) & ((mode_counts / non_null_counts) >= ratio)
        mask_keep = ~mask_drop
        st.session_state[state_key] = st.session_state[state_key].loc[mask_keep].reset_index(drop=True)
        after_count = len(st.session_state[state_key])
        st.rerun()
    if st.session_state.get(fb_key_cons):
        st.success(st.session_state.pop(fb_key_cons))

    # ---------- 7. 反向计分（执行后列名更新为 题目名称+r）----------
    st.markdown("###### 反向计分 (Reverse Coding)")
    need_rc = st.checkbox(f"是否需要进行反向计分（{role}）", key=f"rc_{role}")
    if need_rc:
        rc_cols = smart_multiselect(
            options=st.session_state[state_key].columns.tolist(),
            label=f"选择需要反向计分的题目（{role}）",
            key_suffix=f"rc_{role}",
        )
        scale = st.radio(f"问卷量表类型（{role}）", (5, 7), index=1, key=f"scale_{role}")
        if st.button(f"执行反向计分（{role}）", key=f"btn_rc_{role}") and rc_cols:
            m = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1} if scale == 5 else {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
            m.update({str(k): v for k, v in m.items()})
            rename_after_rc = {}
            for col in rc_cols:
                st.session_state[state_key][col] = pd.to_numeric(st.session_state[state_key][col], errors="coerce").replace(m)
                # 执行反向计分后：若原列名已以 r 结尾则去掉 r，否则在列名后加 r
                if str(col).endswith("r"):
                    rename_after_rc[col] = str(col)[:-1]
                else:
                    rename_after_rc[col] = f"{col}r"
            if rename_after_rc:
                st.session_state[state_key] = _ensure_unique_columns(st.session_state[state_key].rename(columns=rename_after_rc))
                _sync_smart_multiselect_after_rename(f"rc_{role}", rename_after_rc)
            if assign_efa_after and item_cols_key == "dc_efa_item_cols":
                st.session_state.dc_efa_item_cols = get_item_columns(st.session_state[state_key])
            st.success("反向计分完成，已对反向题列名添加后缀「r」。")
            st.rerun()


def render_data_cleaning():
    st.title("模块 0: 数据清洗 (Data Cleaning)")
    st.markdown("""
    <style>
    /* 针对多选框选定值的区域 */
    span[data-baseweb="tag"] {
        background-color: #f0f2f6;
    }
    /* 尝试增加下拉列表的最大高度 */
    ul[data-baseweb="menu"] {
        max-height: 400px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- Session State 初始化 (用于存储处理过程中的数据) ---
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_current' not in st.session_state:
        st.session_state.df_current = None
    if 'step_history' not in st.session_state:
        st.session_state.step_history = []
    if 'current_step_index' not in st.session_state:
        st.session_state.current_step_index = -1

    # 数据导入模式：单数据集 或 EFA & CFA 双数据集
    dc_mode = st.radio("数据导入模式", ["单数据集", "EFA & CFA 双数据集"], horizontal=True, key="dc_import_mode")
    if dc_mode == "EFA & CFA 双数据集":
        _render_dual_mode_cleaning()
        return

    # ==========================================
    # 以下为单数据集模式
    # ==========================================
    
    # 辅助函数：保存当前状态到历史记录
    def save_state(step_name):
        """保存当前数据状态到历史记录"""
        if st.session_state.df_current is not None:
            # 保存当前状态的深拷贝
            state_copy = st.session_state.df_current.copy()
            # 如果当前不是最新状态，删除之后的所有状态（分支操作）
            if st.session_state.current_step_index < len(st.session_state.step_history) - 1:
                st.session_state.step_history = st.session_state.step_history[:st.session_state.current_step_index + 1]
            # 添加新状态
            st.session_state.step_history.append({
                'step_name': step_name,
                'data': state_copy,
                'row_count': len(state_copy),
                'col_count': len(state_copy.columns)
            })
            # 限制历史记录最多 10 步，防止内存无限增长
            MAX_HISTORY = 10
            if len(st.session_state.step_history) > MAX_HISTORY:
                st.session_state.step_history = st.session_state.step_history[-MAX_HISTORY:]
            st.session_state.current_step_index = len(st.session_state.step_history) - 1
    
    # 辅助函数：还原到上一个状态
    def restore_previous_state():
        """还原到上一个状态"""
        if st.session_state.current_step_index > 0:
            # 获取上一个状态
            previous_state = st.session_state.step_history[st.session_state.current_step_index - 1]
            # 恢复数据
            st.session_state.df_current = previous_state['data'].copy()
            # 更新索引
            st.session_state.current_step_index -= 1
            # 删除当前及之后的状态
            st.session_state.step_history = st.session_state.step_history[:st.session_state.current_step_index + 1]
            return True
        return False

    # ==========================================
    # 1. 导入数据 (对应 Cell 1 & 2)
    # ==========================================
    st.subheader("1. 数据导入")
    uploaded_file = st.file_uploader("请上传数据文件 (.xlsx 或 .csv)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_fp = hashlib.sha1(file_bytes).hexdigest()
            if st.session_state.get("dc_upload_single_fp") != file_fp:
                bio = io.BytesIO(file_bytes)
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # 逻辑：统一删除第一行（index=0）- 保持原逻辑
                    df = pd.read_excel(bio)
                    df = df.drop(df.index[0]).reset_index(drop=True)
                else:
                    df = pd.read_csv(bio)

                # 导入后立刻修剪表头前后的隐形空格，防止出现看似重名但由于空格不同而引发的报错或冗余列
                df.columns = [str(c).strip() for c in df.columns]

                st.session_state.dc_upload_single_fp = file_fp
                st.session_state.df_raw = df
                st.session_state.df_current = df.copy()  # 创建副本用于处理
                # 初始化历史记录，保存初始状态
                st.session_state.step_history = [{
                    'step_name': '数据导入',
                    'data': df.copy(),
                    'row_count': len(df),
                    'col_count': len(df.columns)
                }]
                st.session_state.current_step_index = 0
                # 单数据集更新时清理子数据集缓存，避免沿用旧数据拆分结果
                st.session_state.sub_datasets = {}
                st.session_state.sub_datasets_updated = {}
                st.success(f"成功更新导入数据！共 {len(df)} 行")
        except Exception as e:
            st.error(f"读取文件出错: {e}")
    
    # 如果数据已加载，显示后续步骤
    if st.session_state.df_current is not None:
        df = st.session_state.df_current

        # 数据预览 (对应 Cell 2)
        with st.expander("查看原始数据概览", expanded=True):
            st.write("数据前5行预览:")
            st.dataframe(df.head())
            st.write(f"**数据集摘要**: 总字段数: {len(df.columns)} | 总行数: {len(df)}")

        # ==========================================
        # 2. 题目重命名与预处理 (对应 Cell 4)
        # ==========================================
        st.subheader("2. 题目重命名与预处理")

        # 还原按钮
        if st.session_state.current_step_index > 0:
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button("↩️ 还原到上一个状态", key="restore_rename", help="还原到执行重命名之前的状态"):
                    if restore_previous_state():
                        st.success("已还原到上一个状态！")
                        st.rerun()
                    else:
                        st.warning("没有可还原的状态")

        # ==========================================================
        # 🧹 第一步：预处理 - 批量删除指定文字
        # ==========================================================
        st.markdown("###### 🧹 预处理 - 批量删除指定文字")
        st.caption("先清理题目名称中不需要的文字片段（例如冗长的指导语），让题目名称更简洁。")

        text_to_remove = st.text_input("请输入要批量删除的文字片段:", key="text_to_remove_cleaning")

        # 创建状态反馈容器
        remove_status = st.empty()

        if st.button("🚀 执行批量删除", key="btn_remove_text_cleaning"):
            if not text_to_remove:
                st.warning("请先输入要删除的文字。")
            else:
                rename_map = {}
                history_data = []
                count = 0

                for old_name in st.session_state.df_current.columns:
                    if text_to_remove in old_name:
                        # 将指定文字替换为空字符串
                        new_name = old_name.replace(text_to_remove, "")
                        # 如果替换后名字变空了（或者只剩空格），加个保护，防止列名消失
                        if not new_name.strip():
                            new_name = old_name # 撤销修改

                        rename_map[old_name] = new_name
                        history_data.append({"原始题目名称": old_name, "更新后名称": new_name})
                        count += 1

                if count > 0:
                    st.session_state.df_current = _ensure_unique_columns(st.session_state.df_current.rename(columns=rename_map))
                    st.success(f"✅ 成功从 {count} 个题目中移除了指定文字！")
                    st.write("📋 **改名对照表：**")
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True)

                    # 在按钮下方显示持久状态反馈
                    remove_status.info(f"📝 已删除指定文字，共处理了 {count} 个题目")

                    # 刷新页面显示新列名
                    st.rerun()
                else:
                    st.warning(f"⚠️ 在所有题目中未找到文字片段：“{text_to_remove}”")

                    # 在按钮下方显示持久状态反馈
                    remove_status.info("📝 未找到指定的文字片段，无需删除")


        # ==========================================================
        # 🔢 第二步：添加序号
        # ==========================================================
        st.markdown("###### 🔢 添加序号")
        st.caption("为题目添加序号标识，便于后续分析和管理。")

        col1, col2 = st.columns([1, 3])
        with col1:
            do_rename = st.checkbox("是否需要给题目加序号?", value=False)

        if do_rename:
            all_cols = df.columns.tolist()
            start_item = st.selectbox("请选择起始题目的名称：", all_cols)

            # 创建状态反馈容器
            rename_status = st.empty()

            if st.button("执行添加序号"):
                try:
                    # 保存当前状态
                    save_state("题目重命名")
                    start_idx = all_cols.index(start_item)
                    new_cols = (
                        all_cols[:start_idx] +
                        [f"{i+1}_{c}" for i, c in enumerate(all_cols[start_idx:], start=0)]
                    )
                    st.session_state.df_current.columns = new_cols
                    st.success("列名添加序号完成！")

                    # 在按钮下方显示持久状态反馈
                    rename_status.info(f"📝 已为题目添加序号标识，从第 {start_idx + 1} 个题目开始编号")

                    st.rerun() # 刷新页面显示新列名
                except Exception as e:
                    st.error(f"添加序号失败: {e}")

                    # 在按钮下方显示错误状态反馈
                    rename_status.error("❌ 添加序号操作失败")

    
        st.markdown("---")
        # ==========================================
        # 3. 筛选：注意力检查与诚实性检查 (对应 Cell 5)
        # ==========================================
        @st.fragment
        def _fragment_attention_honesty():
            st.subheader("3. 筛选：注意力与诚实性检查")

            # 显示上次操作的反馈消息
            _fb_key = "_fb_attention"
            if _fb_key in st.session_state:
                st.success(st.session_state.pop(_fb_key))

            # 还原按钮
            if st.session_state.current_step_index > 0:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("↩️ 还原到上一个状态", key="restore_filter", help="还原到执行筛选之前的状态"):
                        if restore_previous_state():
                            st.session_state[_fb_key] = "已还原到上一个状态！"
                        else:
                            st.warning("没有可还原的状态")
                        st.rerun(scope="app")

            st.info("请配置筛选题规则，点击下方按钮进行清洗。")

            # 注意力题配置
            num_attention = st.number_input("注意力检查题数量 (Attention Check)", min_value=0, value=0, step=1)
            attention_configs = []
            if num_attention > 0:
                for i in range(num_attention):
                    st.markdown(f"**注意力题 {i+1} 配置**")
                    col_a, col_b = st.columns(2)
                    item_name = col_a.selectbox(f"选择题目 {i+1}", st.session_state.df_current.columns, key=f"att_q_{i}")
                    ans_str = col_b.text_input(f"正确答案 {i+1} (多个答案用 / 分隔)", key=f"att_a_{i}")
                    attention_configs.append({'name': item_name, 'answers': [a.strip() for a in (ans_str or "").split('/') if a.strip()]})

            # 诚实题配置
            num_honesty = st.number_input("诚实性检查题数量 (Honesty Check)", min_value=0, value=0, step=1)
            honesty_configs = []
            if num_honesty > 0:
                for i in range(num_honesty):
                    st.markdown(f"**诚实题 {i+1} 配置**")
                    col_a, col_b = st.columns(2)
                    item_name = col_a.selectbox(f"选择题目 {i+1}", st.session_state.df_current.columns, key=f"hon_q_{i}")
                    ans_str = col_b.text_input(f"正确答案 {i+1} (多个答案用 / 分隔)", key=f"hon_a_{i}")
                    honesty_configs.append({'name': item_name, 'answers': [a.strip() for a in (ans_str or "").split('/') if a.strip()]})

            if st.button("执行题目筛选清洗"):
                current_len = len(st.session_state.df_current)
                df_check = st.session_state.df_current
                mask_keep = pd.Series(True, index=df_check.index)

                for check in attention_configs + honesty_configs:
                    item_name = check['name']
                    if item_name not in df_check.columns:
                        continue
                    is_missing = df_check[item_name].apply(_is_missing_val)
                    normalized_col = df_check[item_name].apply(_normalize_answer)
                    normalized_answers = [_normalize_answer(a) for a in check['answers']]
                    is_correct = normalized_col.isin(normalized_answers)
                    mask_keep &= (~is_missing & is_correct)

                st.session_state.df_current = df_check[mask_keep].reset_index(drop=True)
                new_len = len(st.session_state.df_current)
                deleted_count = current_len - new_len
                st.session_state[_fb_key] = f"筛选完成！筛选前: {current_len} -> 筛选后: {new_len}. 删除了 {deleted_count} 行。"
                st.rerun(scope="app")

        _fragment_attention_honesty()
        st.markdown("---")

        # ==========================================
        # 4. IP 地址筛选 (对应 Cell 80 - 之前是7)
        # ==========================================
        @st.fragment
        def _fragment_ip_filter():
            st.subheader("4. IP 地址筛选")

            _fb_key = "_fb_ip"
            if _fb_key in st.session_state:
                st.success(st.session_state.pop(_fb_key))

            if st.session_state.current_step_index > 0:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("↩️ 还原到上一个状态", key="restore_ip", help="还原到执行IP筛选之前的状态"):
                        if restore_previous_state():
                            st.session_state[_fb_key] = "已还原到上一个状态！"
                        else:
                            st.warning("没有可还原的状态")
                        st.rerun(scope="app")

            _df_cols = st.session_state.df_current.columns
            ip_col = st.selectbox("选择代表IP地址的列", _df_cols, index=_df_cols.get_loc("IP") if "IP" in _df_cols else 0)

            if st.button("执行相同IP去重"):
                save_state("IP地址筛选")
                before_len = len(st.session_state.df_current)
                st.session_state.df_current = st.session_state.df_current.drop_duplicates(subset=ip_col, keep='first')
                after_len = len(st.session_state.df_current)
                st.session_state[_fb_key] = f"IP筛选完成。剩余样本量: {after_len} (删除了 {before_len - after_len} 行)"
                st.rerun(scope="app")

        _fragment_ip_filter()
        st.markdown("---")

        # ==========================================
        # 5. 反应时长筛选 (对应 Cell 6)
        # ==========================================
        @st.fragment
        def _fragment_time_filter():
            st.subheader("5. 反应时长筛选")

            _fb_key = "_fb_time"
            if _fb_key in st.session_state:
                st.success(st.session_state.pop(_fb_key))

            if st.session_state.current_step_index > 0:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("↩️ 还原到上一个状态", key="restore_time", help="还原到执行时长筛选之前的状态"):
                        if restore_previous_state():
                            st.session_state[_fb_key] = "已还原到上一个状态！"
                        else:
                            st.warning("没有可还原的状态")
                        st.rerun(scope="app")

            time_col_name = "作答总时长(秒)"
            _df_cols = st.session_state.df_current.columns
            actual_time_col = st.selectbox("选择作答时长列", _df_cols, index=_df_cols.get_loc(time_col_name) if time_col_name in _df_cols else 0)

            if st.button("执行反应时长筛选 (3SD原则)"):
                save_state("反应时长筛选")
                try:
                    temp_series = pd.to_numeric(
                        st.session_state.df_current[actual_time_col].astype(str).str.replace("秒", "", regex=False),
                        errors='coerce'
                    ).astype('Int64')
                    st.session_state.df_current[actual_time_col] = temp_series

                    valid_times = st.session_state.df_current[actual_time_col].dropna()
                    avg_time = valid_times.mean()
                    sd_time = valid_times.std()
                    lower_bound = max(0, avg_time - 1.96 * sd_time)
                    upper_bound = avg_time + 3 * sd_time

                    mask_keep = (st.session_state.df_current[actual_time_col] >= lower_bound) & (st.session_state.df_current[actual_time_col] <= upper_bound)
                    rows_to_drop = st.session_state.df_current[~mask_keep]
                    st.session_state.df_current = st.session_state.df_current[mask_keep].copy().reset_index(drop=True)

                    st.session_state[_fb_key] = f"时长筛选完成。删除 {len(rows_to_drop)} 行。剩余: {len(st.session_state.df_current)}（均值={avg_time:.2f}, SD={sd_time:.2f}, 区间=[{lower_bound:.2f}, {upper_bound:.2f}]）"
                    st.rerun(scope="app")
                except Exception as e:
                    st.error(f"处理时长列时出错: {e}")

        _fragment_time_filter()
        st.markdown("---")

        # ==========================================
        # 6. 作答一致性筛选 (Straight-lining) (对应 Cell 7)
        # ==========================================
        @st.fragment
        def _fragment_consistency_check():
            st.subheader("6. 作答一致性筛选 (Straight-lining)")

            # 显示上次操作的反馈消息
            _fb_key = "_fb_consistency"
            if _fb_key in st.session_state:
                st.success(st.session_state.pop(_fb_key))

            # 还原按钮
            if st.session_state.current_step_index > 0:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("↩️ 还原到上一个状态", key="restore_consistency", help="还原到执行一致性筛选之前的状态"):
                        if restore_previous_state():
                            st.session_state[_fb_key] = "已还原到上一个状态！"
                        else:
                            st.warning("没有可还原的状态")
                        st.rerun(scope="app")

            ratio = st.slider("选择一致性阈值 (例如 0.9 代表 90% 的题目答案相同)", 0.5, 1.0, 0.9)

            if st.button("执行一致性筛选"):
                save_state("作答一致性筛选")
                df_temp = st.session_state.df_current.copy()

                for col in df_temp.columns:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

                num_df = df_temp.select_dtypes(include=[np.number])

                non_null_counts = num_df.notna().sum(axis=1)
                mode_vals = num_df.mode(axis=1).iloc[:, 0] if not num_df.empty else pd.Series(dtype=float)
                mode_counts = num_df.eq(mode_vals, axis=0).sum(axis=1)
                mask_drop = (non_null_counts > 0) & ((mode_counts / non_null_counts) >= ratio)

                rows_dropped = df_temp[mask_drop]
                st.session_state.df_current = st.session_state.df_current[~mask_drop].reset_index(drop=True)

                st.session_state[_fb_key] = f"一致性筛选完成。删除 {len(rows_dropped)} 行 (答案重复率 >= {ratio*100}%)。"
                st.rerun(scope="app")

        _fragment_consistency_check()
        st.markdown("---")

        # ==========================================
        # 7. 反向计分 (Reverse Coding) (对应 Cell 10/11)
        # ==========================================
        st.subheader("7. 反向计分 (Reverse Coding)")
        
        # 还原按钮
        if st.session_state.current_step_index > 0:
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button("↩️ 还原到上一个状态", key="restore_reverse", help="还原到执行反向计分之前的状态"):
                    if restore_previous_state():
                        st.success("已还原到上一个状态！")
                        st.rerun()
                    else:
                        st.warning("没有可还原的状态")
        
        need_rc = st.checkbox("是否需要进行反向计分?")
        if need_rc:
          #  rc_cols = st.multiselect("请选择需要反向计分的题目", st.session_state.df_current.columns)
            rc_cols = smart_multiselect(
                options=st.session_state.df_current.columns.tolist(),
                label="请选择需要反向计分的题目",
                key_suffix="reverse_coding"
            )
            scale_type = st.radio("问卷量表类型", (5, 7), index=1)
            
            if st.button("执行反向计分"):
                # 保存当前状态
                save_state("反向计分")
                mapping = {}
                if scale_type == 5:
                    mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, '1': 5, '2': 4, '3': 3, '4': 2, '5': 1}
                else:
                    mapping = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, '1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1}
                
                try:
                    # 确保列是数值型或可转换，并执行反向计分
                    rename_after_rc = {}
                    for col in rc_cols:
                        st.session_state.df_current[col] = pd.to_numeric(st.session_state.df_current[col], errors='coerce')
                        st.session_state.df_current[col] = st.session_state.df_current[col].replace(mapping)
                        # 反向计分后更新题目名称：若列名以 "r" 结尾则去掉 "r"，否则加上 "r"
                        if str(col).rstrip().endswith("r"):
                            rename_after_rc[col] = str(col).rstrip()[:-1]
                        else:
                            rename_after_rc[col] = f"{col}r"
                    if rename_after_rc:
                        st.session_state.df_current = _ensure_unique_columns(st.session_state.df_current.rename(columns=rename_after_rc))
                        _sync_smart_multiselect_after_rename("reverse_coding", rename_after_rc)
                    st.success(f"已完成对 {len(rc_cols)} 个题目的反向计分，并已在对应列名后添加后缀「r」。")
                    st.rerun()
                except Exception as e:
                    st.error(f"反向计分出错: {e}")

        st.markdown("---")

        st.markdown("---")

        # ==========================================
        # 7. 反向计分 (Reverse Coding)
        # ==========================================
        st.subheader("8. 拆分数据集 (构建子数据集)")
        st.info("在这里，你可以从清洗好的总表中挑选特定的题目（列），组成新的子数据集（例如：只包含量表题的数据集），供后续 N1/N2 分析使用。")

        # 初始化 sub_datasets 字典
        if 'sub_datasets' not in st.session_state:
            st.session_state.sub_datasets = {}

        # 1. 输入子数据集名称
        sub_name = st.text_input("给新数据集起个名字 (例如: EFA_Items)", value="Sub_Dataset_1", key="split_sub_name")

        # 2. 选择包含的列
        # 默认全选太乱，默认不选，让用户自己挑
        all_columns = st.session_state.df_current.columns.tolist()

        # 初始化选择状态管理
        if 'split_data_selection' not in st.session_state:
            st.session_state.split_data_selection = []

        selected_cols = smart_multiselect(
            options=all_columns,
            label="请选择包含在该数据集中的列 (题目)",
            key_suffix="split_data",
            default_selected=st.session_state.split_data_selection,
            show_selection_controls=True
        )

        # 同步选择状态
        st.session_state.split_data_selection = selected_cols

        # 3. 创建按钮 + 数据集列表 — 用 fragment 隔离，避免全页 rerun 闪烁
        @st.fragment
        def _dataset_actions():
            _sub_name = st.session_state.get("split_sub_name", "Sub_Dataset_1")
            _selected = st.session_state.get("split_data_last_selected", [])

            if st.button("创建并保存子数据集", key="btn_create_sub"):
                if not _sub_name:
                    st.error("请输入数据集名称！")
                elif not _selected:
                    st.error("请至少选择一列！")
                else:
                    df_sub = st.session_state.df_current[_selected].copy()
                    st.session_state.sub_datasets[_sub_name] = df_sub
                    st.success(f"成功创建数据集: 【{_sub_name}】，包含 {len(_selected)} 列，{len(df_sub)} 行。")

            # 4. 显示已创建的数据集列表
            if len(st.session_state.sub_datasets) > 0:
                st.write("📊 **当前已保存的待分析数据集:**")

                dataset_info = []
                for name, data in st.session_state.sub_datasets.items():
                    dataset_info.append({"数据集名称": name, "样本量": data.shape[0], "题目数": data.shape[1]})

                st.table(pd.DataFrame(dataset_info))

                del_name = st.selectbox("选择要删除的数据集", ["(不删除)"] + list(st.session_state.sub_datasets.keys()), key="del_sub_select")
                if del_name != "(不删除)":
                    if st.button("确认删除选中数据集", key="btn_del_sub"):
                        del st.session_state.sub_datasets[del_name]
                        st.rerun(scope="app")  # 删除需要全页刷新以更新第9节

        _dataset_actions()

        

        # ==========================================
        #    跳转按钮
        # ==========================================
        st.markdown("### 🚀 下一步")
        col1, col2 = st.columns([1, 2])
        with col1:
            # 1. 定义一个回调函数，专门用来改状态
            def go_to_n1():
                st.session_state.nav_selection = "2. N1 EFA数据分析"
            
            # 2. 在按钮里使用 on_click 参数调用这个函数
            st.button(
                "前往 N1 模块进行分析 ->", 
                type="primary", 
                use_container_width=True,
                on_click=go_to_n1  # <--- 关键在这里
            )
        
        with col2:
            st.caption("点击此按钮将直接携带当前保存的子数据集，跳转至 N1 分析页面。")

        st.markdown("---")


        
        # ==========================================
        # 9. 结果导出
        # ==========================================
        st.subheader("10. 导出清洗后的数据")
        st.write(f"当前最终数据集行数: {len(st.session_state.df_current)}")
        
        # 将 DataFrame 转换为 Excel 字节流供下载
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.df_current.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 下载清洗后的 Excel 文件",
            data=processed_data,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
