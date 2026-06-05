import streamlit as st

from data_cleaning import run_data_cleaning
from n1_analysis import run_n1
from n2_analysis import run_n2

st.title("Batch EFA & CFA System")

menu = st.sidebar.selectbox(
    "选择模块",
    ["数据清洗", "N1分析", "N2分析"]
)

if menu == "数据清洗":
    run_data_cleaning()

elif menu == "N1分析":
    run_n1()

elif menu == "N2分析":
    run_n2()
