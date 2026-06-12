import streamlit as st
from data_cleaning import render_data_cleaning
from n1_analysis import render_n1_analysis  # 假设你有这个函数

st.set_page_config(layout="wide")

def main():
    st.sidebar.title("导航")

    if "page" not in st.session_state:
        st.session_state.page = "数据清洗"
        
        page = st.sidebar.radio(
            "选择模块",
            ["数据清洗", "N1分析"],
            key="page"
        )
    
        if st.session_state.page == "数据清洗":
            render_data_cleaning()
        
        elif st.session_state.page == "N1分析":
            render_n1_analysis()

if __name__ == "__main__":
    main()
