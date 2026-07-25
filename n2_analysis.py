import streamlit as st
from single_factor_cfa import render_singlefactor_cfa
from multi_factor_cfa import render_multifactor_cfa


def render_n2_analysis():
    st.title("模块 3: N2 分析")

    # 使用 st.tabs 将三大核心分析板块在水平方向彻底隔离
    tab_single_cfa, tab_multi_cfa = st.tabs([
        "1. Single factor CFA", 
        "2. Multi factor CFA"
    ])

    # 板块一：直接渲染原逻辑改名后的核心 EFA
    with tab_single_cfa:
        render_singlefactor_cfa()

    # 板块二：自动删题 CFA (读取 Stage 1 的 N1_preEFA 资产进行分析)
    with tab_multi_cfa:
        render_multifactor_cfa()

