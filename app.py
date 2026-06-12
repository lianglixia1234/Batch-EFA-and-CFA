import streamlit as st
from data_cleaning import render_data_cleaning
from n1_analysis import render_n1_analysis  # 假设你有这个函数

st.set_page_config(layout="wide")

def main():
    st.sidebar.title("导航")

    # 注意这里把所有选项放在一个列表里
    page = st.sidebar.radio(
        "选择模块",
        ["数据清洗", "N1分析"]
    )

    if page == "数据清洗":
        render_data_cleaning()
    elif page == "N1分析":
        render_n1_analysis()  # 点击 N1分析 时运行这个函数

if __name__ == "__main__":
    main()
