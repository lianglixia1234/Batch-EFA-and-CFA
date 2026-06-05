# 通用工具函数模块
import streamlit as st
import pandas as pd
import re
import unicodedata


def normalize_item_text(text):
    """题目文本规范化，用于 EFA/CFA 题目匹配：去首尾空格、统一标点（全角转半角）。"""
    if not text or not isinstance(text, str):
        return ""
    s = text.strip()
    # 全角标点、空格转半角
    s = "".join(
        unicodedata.normalize("NFKC", c) if unicodedata.category(c).startswith("P") or c.isspace() else c
        for c in s
    )
    return s


def get_item_columns(df):
    """检测题目列：列名符合 EFA数字_、EFA_数字_、CFA数字_ 或 CFA_数字_ 的格式。"""
    if df is None or df.empty:
        return []
    item_cols = []
    for c in df.columns:
        if isinstance(c, str) and (
            re.match(r"^EFA\d+_", c) or re.match(r"^EFA_\d+_", c)
            or re.match(r"^CFA\d+_", c) or re.match(r"^CFA_\d+_", c)
        ):
            item_cols.append(c)
    return item_cols


def parse_item_col(col_name):
    """解析题目列名，返回 (prefix, num, text)。如 EFA12_题目 或 EFA_12_题目 -> ('EFA', 12, '题目')。"""
    if not isinstance(col_name, str):
        return None, None, ""
    m = re.match(r"^(EFA|CFA)(\d+)_(.+)$", col_name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3).strip()
    m2 = re.match(r"^(EFA|CFA)_(\d+)_(.+)$", col_name)
    if m2:
        return m2.group(1), int(m2.group(2)), m2.group(3).strip()
    return None, None, col_name


def sort_item_cols_by_number(cols):
    """按题目编号排序：先按 prefix (EFA 在前), 再按数字升序。"""
    def key(c):
        pre, num, _ = parse_item_col(c)
        if pre is None:
            return (1, 0)
        return (0 if pre == "EFA" else 1, num)
    return sorted(cols, key=key)


def smart_multiselect(options, label, key_suffix, default_selected=None, show_selection_controls=False):
    """
    使用 st.checkbox 列表实现的多选功能，用 @st.fragment 隔离渲染。
    每个选项是独立的 checkbox，Streamlit 原生管理状态，全选/全不选/反选通过 on_click 直接操作。
    checkbox 在 fragment 内工作可靠（不像 st.data_editor 有缓存不同步问题）。

    参数:
    options: 选项列表
    label: 标签
    key_suffix: 键后缀
    default_selected: 默认选中的选项
    show_selection_controls: 是否显示选择控制按钮（全选、全不选、反选）
    """
    if default_selected is None:
        default_selected = []

    last_selected_key = f"{key_suffix}_last_selected"
    cb_prefix = f"cb_{key_suffix}_"

    @st.fragment
    def _render():
        # 处理控制按钮信号（on_click 回调在 fragment 重跑前已设好）
        control_action = st.session_state.pop(f"{key_suffix}_control_action", None)
        if control_action == "select_all":
            for i in range(len(options)):
                st.session_state[f"{cb_prefix}{i}"] = True
        elif control_action == "select_none":
            for i in range(len(options)):
                st.session_state[f"{cb_prefix}{i}"] = False
        elif control_action == "select_inverse":
            for i in range(len(options)):
                k = f"{cb_prefix}{i}"
                st.session_state[k] = not st.session_state.get(k, options[i] in default_selected)

        with st.expander(f"👇 点击展开/收起：{label}", expanded=False):
            if show_selection_controls:
                def _set_action(action, ks=key_suffix):
                    st.session_state[f"{ks}_control_action"] = action

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.button("✅ 全选", key=f"select_all_{key_suffix}", help="选择所有题目",
                              on_click=_set_action, args=("select_all",))
                with col2:
                    st.button("❌ 全不选", key=f"select_none_{key_suffix}", help="清除所有选择",
                              on_click=_set_action, args=("select_none",))
                with col3:
                    st.button("🔄 反选", key=f"select_inverse_{key_suffix}", help="反转当前选择",
                              on_click=_set_action, args=("select_inverse",))

            container = st.container(height=300)
            selected = []
            with container:
                for i, item in enumerate(options):
                    default_val = item in default_selected
                    checked = st.checkbox(item, value=default_val, key=f"{cb_prefix}{i}")
                    if checked:
                        selected.append(item)

        st.session_state[last_selected_key] = selected
        st.info(f"已选择 {len(selected)} 个题目: {', '.join(selected[:3])}{'...' if len(selected) > 3 else ''}")

    _render()

    # 从 session_state 读取最新选择结果返回给调用方
    result = st.session_state.get(last_selected_key, list(default_selected))
    return [x for x in result if x in options]