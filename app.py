import streamlit as st

from data_cleaning import run_data_cleaning

st.set_page_config(
    page_title="数据清洗系统",
    layout="wide"
)

def main():
    st.title("📊 数据清洗模块")

    # 直接调用数据清洗UI
    run_data_cleaning()

if __name__ == "__main__":
    main()
