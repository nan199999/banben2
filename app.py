# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="Piuma Soft Material Analysis", page_icon="💧", layout="wide")

# --- 样式 ---
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("💧 软物质压痕分析 (Hertz Model)")
st.markdown("专为 **Piuma / Optics11** 设备设计，适用于细胞、水凝胶、生物组织分析。")

# --- 侧边栏：参数设置 ---
with st.sidebar:
    st.header("1. 探针与材料参数")
    
    # 探针半径 (关键参数)
    tip_radius_um = st.number_input("探针半径 R (um)", value=23.0, help="查看实验记录，图片显示为 23.0")
    
    # 泊松比
    nu = st.number_input("样品泊松比 v", value=0.5, help="生物材料/水凝胶通常取 0.5")
    
    st.divider()
    
    st.header("2. 原始单位选择")
    # Piuma 通常是 uN 和 um
    force_unit = st.selectbox("载荷单位 (Load)", ["uN (微牛)", "mN (毫牛)", "N (牛顿)"], index=0)
    disp_unit = st.selectbox("位移单位 (Disp)", ["um (微米)", "nm (纳米)", "m (米)"], index=0)

# --- 核心函数 ---
@st.cache_data
def load_tdms(file):
    try:
        tdms = TdmsFile.read(file)
        data = {}
        for group in tdms.groups():
            for channel in group.channels():
                data[channel.name] = channel[:]
        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    except:
        return None

def hertz_model(h, E_star, R):
    """
    Hertz Contact Model for Sphere:
    F = (4/3) * E* * sqrt(R) * h^(1.5)
    """
    # 强制 h >= 0，避免复数错误
    h = np.maximum(h, 0)
    return (4.0 / 3.0) * E_star * np.sqrt(R) * np.power(h, 1.5)

# --- 主界面 ---
uploaded_file = st.file_uploader("📂 上传 Piuma 生成的 .tdms 文件", type=["tdms"])

if uploaded_file:
    df = load_tdms(uploaded_file)
    
    if df is not None:
        st.success("文件读取成功")
        
        # 1. 通道映射
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            col_load = st.selectbox("选择载荷列 (Load/Force)", cols, index=0)
        with c2:
            col_disp = st.selectbox("选择位移列 (Disp/Indentation)", cols, index=1 if len(cols)>1 else 0)

        # 2. 数据转换 (全部转为 SI 单位: N, m)
        raw_F = df[col_load].dropna().values
        raw_D = df[col_disp].dropna().values
        
        # 单位换算系数
        scale_F = 1e-6 if "uN" in force_unit else (1e-3 if "mN" in force_unit else 1.0)
        scale_D = 1e-6 if "um" in disp_unit else (1e-9 if "nm" in disp_unit else 1.0)
        
        F_si = raw_F * scale_F  # 单位: N
        D_si = raw_D * scale_D  # 单位: m
        R_si = tip_radius_um * 1e-6 # 单位: m

        # 3. 寻找接触点 (最重要的步骤)
        st.subheader("🔍 接触点校准 (Contact Point)")
        st.info("拖动滑块，使红线对准**力开始上升**的瞬间。左侧通常是基线噪音。")
        
        # 创建滑块用于找零
        start_idx = st.slider("选择接触起始点 (Index)", 0, len(F_si)-1, 0)
        
        # 归零后的数据
        F_zeroed = F_si[start_idx:] - F_si[start_idx]
        D_zeroed = D_si[start_idx:] - D_si[start_idx]
        
        # 确保只要正值 (压入部分)
        mask = (F_zeroed > 0) & (D_zeroed > 0)
        F_fit = F_zeroed[mask]
        D_fit = D_zeroed[mask]

        # 绘图：帮助找零
        fig_calib = go.Figure()
        # 全局数据
        fig_calib.add_trace(go.Scatter(y=F_si, mode='lines', name='原始数据', line=dict(color='gray')))
        # 选中的接触点
        fig_calib.add_trace(go.Scatter(x=[start_idx], y=[F_si[start_idx]], mode='markers', marker=dict(color='red', size=10), name='接触点'))
        fig_calib.update_layout(title="调整滑块直到红点位于曲线起飞处", xaxis_title="数据点索引", yaxis_title="载荷 (N)")
        st.plotly_chart(fig_calib, use_container_width=True)

        # 4. Hertz 拟合与计算
        if len(F_fit) > 10:
            if st.button("🚀 计算杨氏模量 (Young's Modulus)", type="primary"):
                try:
                    # 定义拟合函数 wrapper，固定 R
                    def fit_func(h, E_star):
                        return hertz_model(h, E_star, R_si)
                    
                    # 拟合 E*
                    # 初始猜测 10 kPa = 10000 Pa
                    popt, pcov = curve_fit(fit_func, D_fit, F_fit, p0=[10000], bounds=(0, np.inf))
                    E_star = popt[0] # 单位 Pa
                    
                    # 计算样品模量 E_sample
                    # 假设探针无限硬: 1/E* = (1-v_s^2)/E_s + (1-v_i^2)/E_i
                    # 简化为: E_s = E* * (1 - v_s^2)
                    E_sample_Pa = E_star * (1 - nu**2)
                    E_sample_kPa = E_sample_Pa / 1000.0
                    
                    # 结果展示
                    st.divider()
                    st.markdown("### 📊 分析结果")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("杨氏模量 (E)", f"{E_sample_kPa:.2f} kPa")
                    col_res2.metric("最大载荷", f"{np.max(F_fit)*1e6:.2f} uN")
                    col_res3.metric("最大压入深度", f"{np.max(D_fit)*1e6:.2f} um")
                    
                    # 拟合效果图
                    fig_fit = go.Figure()
                    # 实验数据
                    fig_fit.add_trace(go.Scatter(x=D_fit*1e6, y=F_fit*1e6, mode='lines', name='实验数据 (归零后)', line=dict(color='#2E86C1')))
                    # 拟合曲线
                    D_sim = np.linspace(0, np.max(D_fit), 100)
                    F_sim = hertz_model(D_sim, E_star, R_si)
                    fig_fit.add_trace(go.Scatter(x=D_sim*1e6, y=F_sim*1e6, mode='lines', name=f'Hertz 拟合 (E={E_sample_kPa:.1f} kPa)', line=dict(color='red', dash='dash')))
                    
                    fig_fit.update_layout(
                        title="F-D 曲线与 Hertz 拟合",
                        xaxis_title="Indentation Depth (um)",
                        yaxis_title="Force (uN)",
                        hovermode="x"
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"拟合失败: {e}。可能是数据噪音太大或未正确归零。")
        else:
            st.warning("⚠️ 选定的接触区域有效数据太少，请调整接触点滑块。")
