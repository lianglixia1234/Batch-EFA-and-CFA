import streamlit as st
from data_cleaning import render_data_cleaning
from n1_analysis import render_n1_analysis
from n2_analysis import render_n2_analysis

st.set_page_config(layout="wide")

def main():
    st.sidebar.title("导航")

    # 初始化状态
    if "page" not in st.session_state:
        st.session_state.page = "数据清洗"

    # sidebar 永远存在（关键）
    st.sidebar.radio(
        "选择模块",
        ["数据清洗", "N1分析","N2分析"],
        key="page"
    )

    # 页面路由（永远执行）
    if st.session_state.page == "数据清洗":
        render_data_cleaning()

    elif st.session_state.page == "N1分析":
        render_n1_analysis()

    elif st.session_state.page == "N2分析":
        render_n2_analysis()



if __name__ == "__main__":
    main()
