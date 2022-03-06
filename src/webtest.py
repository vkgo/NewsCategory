import streamlit as st
import forecast as fc
# 基本组件
forecast = fc.forecast()


# 网页配置
st.set_page_config(page_title="中文新闻识别分类", page_icon="random")#页面基本设置
st.title('中文新闻识别分类')

st.subheader("关于:")
st.write("华南理工大学计算机科学与工程学院")
st.write("2020级网络工程黄韦锦")
st.write("数据结构课程大作业")

st.subheader("注意:")
st.write("由于新闻训练样本原因，新闻放数段或全篇效果最佳")
st.write("分类数10")
st.write("'体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经'")

rawnews = st.text_area(label='输入要预测的新闻:')

if st.button('确认提交'):
    ans = forecast.get(rawnews)
    st.write("分类结果:" + ans)