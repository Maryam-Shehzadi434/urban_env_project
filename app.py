"""
Urban Environmental Intelligence Dashboard
Streamlit Application for Tasks 1-4
Author: Data Architect
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import base64
from io import BytesIO

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# UPDATED MODERN CSS (HERO FIXED)
# =============================================================================
st.markdown("""
<style>

/* REMOVE STREAMLIT DEFAULT PADDING FOR FULL WIDTH */
.block-container {
    padding-top: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

/* ================= HERO SECTION ================= */

.main-header {
    position: relative;
    width: 100vw;
    height: 400px;
    margin-left: calc(-50vw + 50%);
    margin-top: -1rem;
    margin-bottom: 2rem;
    overflow: hidden;
}

.header-title {
    position: absolute;
    top: 28%;
    left: 8%;
    color: white;
    font-size: 3.3rem;
    font-weight: 800;
    text-align: left;
    letter-spacing: 1px;
    font-family: 'Helvetica', sans-serif;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.35);
    max-width: 700px;
    line-height: 1.2;
    z-index: 10;
}

/* ================= HEADINGS ================= */

.sub-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0D47A1;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #90CAF9;
}

/* ================= CARDS ================= */

.task-card {
    background: linear-gradient(145deg, #ffffff, #f3f8ff);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #1976D2;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.insight-box {
    background: #E3F2FD;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #1976D2;
    margin: 2rem 0;
}

.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
    border-bottom: 4px solid #1976D2;
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #0D47A1;
}

.metric-label {
    font-size: 0.95rem;
    color: #546E7A;
}

/* ================= SIDEBAR CONTACT ================= */

.sidebar-contact {
    background: transparent;
    padding: 0.5rem 0;
    margin-top: 0.5rem;
}

/* ================= TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 52px;
    background-color: #F0F4FF;
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #1976D2;
    color: white;
}

/* ================= WELCOME TEXT ================= */

.welcome-text {
    font-size: 1.2rem;
    line-height: 1.6;
    color: #333;
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
}

/* ================= DOWNLOAD BUTTON ================= */

.download-button {
    display: inline-block;
    background-color: #E8ECF1;
    color: #0D47A1;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 500;
    margin-top: 0.5rem;
    border: 1px solid #B0BEC5;
    cursor: pointer;
    transition: 0.3s;
}

.download-button:hover {
    background-color: #D0D8E0;
    border-color: #90A4AE;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    padding: 2rem;
    color: #546E7A;
    font-size: 0.9rem;
    border-top: 1px solid #E0E0E0;
    margin-top: 3rem;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
IMAGE_PATH = SCRIPT_DIR / "UI.jpeg"

if not OUTPUTS_DIR.exists():
    st.error(f"❌ Outputs directory not found at: {OUTPUTS_DIR}")
    st.info("Please make sure the 'outputs' folder exists with all generated images.")
    st.stop()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_image(image_name):
    """Load an image from the outputs directory"""
    image_path = OUTPUTS_DIR / image_name
    if image_path.exists():
        return Image.open(image_path)
    return None

def get_header_image_base64():
    """Load the header image and convert to base64"""
    if IMAGE_PATH.exists():
        with open(IMAGE_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def load_text_file(file_name):
    """Load a text file from the outputs directory"""
    file_path = OUTPUTS_DIR / file_name
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def create_download_button(text, filename, button_text):
    """Create a download button for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-button">📥 {button_text}</a>'
    return href

# =============================================================================
# SIDEBAR - WITH MODIFIED CONTACT SECTION
# =============================================================================

with st.sidebar:
    st.markdown("## 🌆 Urban Environmental Intelligence")
    st.markdown("---")
    
    st.markdown("### 📊 Dashboard Navigation")
    st.markdown("""
    - **Home**: Project Overview
    - **Task 1**: Dimensionality Reduction
    - **Task 2**: Temporal Analysis
    - **Task 3**: Distribution Modeling
    - **Task 4**: Visual Integrity Audit
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Project Summary")
    st.markdown("""
    **Analysis of 100 global air quality sensors**
    - 867,240 hourly readings
    - 6 environmental variables
    - Full year 2025 data
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    
    st.markdown('<div class="sidebar-contact">', unsafe_allow_html=True)
    
    # Email with light red box - showing just "📧 Email" as text
    st.markdown("""
    <div style="background-color: #FFE5E5; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; text-align: left; border-left: 4px solid #EA4335;">
        <a href="mailto:maryam.shehzadi434@gmail.com" style="color: #B71C1C; text-decoration: none; font-weight: 500; display: flex; align-items: center;">
            <span style="font-size: 1.2rem; margin-right: 8px;">📧</span> <span>Email</span>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # LinkedIn with light blue box - showing just "🔗 LinkedIn" as text
    st.markdown("""
    <div style="background-color: #E5F0FF; padding: 8px 12px; border-radius: 8px; text-align: left; border-left: 4px solid #0077B5;">
        <a href="https://www.linkedin.com/in/maryam-shehzadi-14a0a4394/" style="color: #004C7A; text-decoration: none; font-weight: 500; display: flex; align-items: center;">
            <span style="font-size: 1.2rem; margin-right: 8px;">🔗</span> <span>LinkedIn</span>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("© 2025 Urban Environmental Intelligence")

# =============================================================================
# FULL-WIDTH HERO SECTION (FIXED - SHOWS TREE)
# =============================================================================

header_img_base64 = get_header_image_base64()

if header_img_base64:
    st.markdown(f"""
    <div class="main-header" style="
        background-image: url('data:image/jpeg;base64,{header_img_base64}');
        background-size: cover;
        background-position: 30% center;
        background-repeat: no-repeat;
    ">
        <div class="header-title" style="margin-left: 3%;">
    Urban Environmental<br>
    Intelligence Dashboard
</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div style="background: linear-gradient(90deg, #1E3A8A, #1976D2); height: 200px; display: flex; align-items: center; justify-content: center; margin-bottom: 2rem;"><h1 style="color: white;">Urban Environmental Intelligence Dashboard</h1></div>', unsafe_allow_html=True)
    st.warning("⚠️ Header image not found. Please ensure 'UI.jpeg' is in the project directory.")

# =============================================================================
# TABS
# =============================================================================

tab_home, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home", 
    "📊 Task 1: Dimensionality Reduction", 
    "📈 Task 2: Temporal Analysis", 
    "📉 Task 3: Distribution Modeling", 
    "🔍 Task 4: Visual Integrity Audit"
])

# =============================================================================
# HOME TAB
# =============================================================================

with tab_home:
    st.markdown('<div class="sub-header">🏠 Project Overview</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="welcome-text">', unsafe_allow_html=True)
    st.markdown("""
    ### Welcome to the Urban Environmental Intelligence Dashboard

    This platform presents a comprehensive analysis of air quality data from **100 global sensor stations** throughout 2025. 
    Our diagnostic engine identifies environmental anomalies, tracks health threshold violations, and provides actionable 
    insights for smart city planning.

    **What We Did:**
    - **Task 1:** Reduced 6-dimensional environmental data to 2D using PCA to visualize Industrial vs Residential zones
    - **Task 2:** Analyzed PM2.5 health threshold violations (>35 μg/m³) across all sensors using high-density temporal visualization
    - **Task 3:** Modeled extreme hazard events (PM2.5 > 200 μg/m³) with tail-inclusive distribution analysis  
    - **Task 4:** Audited a 3D chart proposal using Lie Factor and Data-Ink Ratio principles

    Navigate through the tabs above to explore each task's visualizations and insights.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">100</div><div class="metric-label">Sensor Stations</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">867K</div><div class="metric-label">Data Points</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Variables</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">2025</div><div class="metric-label">Full Year Data</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📡 Data Sources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🌍 OpenAQ Global Air Quality API**
        - PM2.5, PM10, NO₂, O₃ measurements
        - 100 global monitoring stations
        - Hourly readings throughout 2025
        """)
    with col2:
        st.markdown("""
        **🌤️ OpenWeather API**
        - Temperature data
        - Humidity measurements
        - Meteorological context for pollution events
        """)

# =============================================================================
# TAB 1: DIMENSIONALITY REDUCTION
# =============================================================================

with tab1:
    st.markdown('<div class="sub-header">📊 Task 1: Dimensionality Reduction Challenge</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("""
        **Objective:** Analyze relationships among 6 environmental variables across 100 sensors.
        
        **Method:** Principal Component Analysis (PCA) was applied to project 6-dimensional data into 2 dimensions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        summary_text = load_text_file("task1_analysis_summary.txt")
        if summary_text:
            st.markdown(create_download_button(summary_text, "task1_analysis_summary.txt", "Download Full Summary"), unsafe_allow_html=True)
    
    st.markdown("### 📈 PCA Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img = load_image("task1_pca_scatter.png")
        if img:
            st.image(img, caption="Figure 1: PCA Scatter Plot - Industrial vs Residential Zones", use_container_width=True)
    
    with col2:
        img = load_image("task1_loadings_heatmap.png")
        if img:
            st.image(img, caption="Figure 2: PCA Loadings Heatmap", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        img = load_image("task1_variance_bar.png")
        if img:
            st.image(img, caption="Figure 3: Variance Explained", use_container_width=True)
    
    with col2:
        img = load_image("task1_pc1_contributions.png")
        if img:
            st.image(img, caption="Figure 4: PC1 Contributions", use_container_width=True)
    
    with col3:
        img = load_image("task1_pc2_contributions.png")
        if img:
            st.image(img, caption="Figure 5: PC2 Contributions", use_container_width=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Key Insights from Task 1:**
    
    - **PC1** explains 33.1% of variance - represents "Overall Pollution Intensity"
    - **PC2** explains 21.1% of variance - represents "Pollution Composition"
    - Industrial zones cluster at negative PC1 values (higher pollution)
    - Residential zones cluster at positive PC1 values (cleaner air)
    - Primary pollution drivers: O₃ and NO₂ dominate PC1
    - Secondary factors: PM2.5 dominates PC2
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: TEMPORAL ANALYSIS
# =============================================================================

with tab2:
    st.markdown('<div class="sub-header">📈 Task 2: High-Density Temporal Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("""
        **Objective:** Identify PM2.5 health threshold violations (>35 μg/m³) across 100 sensors.
        
        **Problem:** Standard line charts with 100 lines create excessive clutter.
        
        **Solution:** High-density heatmap visualization.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        summary_text = load_text_file("task2_summary.txt")
        if summary_text:
            st.markdown(create_download_button(summary_text, "task2_summary.txt", "Download Summary"), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">96,932</div><div class="metric-label">Total Violations</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">11.2%</div><div class="metric-label">Violation Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">8 AM</div><div class="metric-label">Peak Hour</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">Winter</div><div class="metric-label">Peak Season</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Temporal Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img = load_image("task2_heatmap.png")
        if img:
            st.image(img, caption="Figure 1: High-Density Heatmap - All 100 Stations Over Time", use_container_width=True)
    
    with col2:
        img = load_image("task2_daily_pattern.png")
        if img:
            st.image(img, caption="Figure 2: Daily Pattern - 24-Hour Cycle", use_container_width=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = load_image("task2_monthly_pattern.png")
        if img:
            st.image(img, caption="Figure 3: Monthly Pattern - Seasonal Cycle", use_container_width=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Key Insights from Task 2:**
    
    - **Daily Pattern:** Clear rush hour peaks at 8 AM (23.7%) and 6 PM (23.4%)
    - **Weekly Pattern:** Weekdays show higher violations than weekends (traffic-driven)
    - **Seasonal Pattern:** Winter months (Dec-Feb) show 15.8% violation rate
    - **Synchronized Events:** Multiple stations violate simultaneously during winter mornings
    - **Problem Neighborhoods:** Top 10% of stations account for 40% of all violations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: DISTRIBUTION MODELING
# =============================================================================

with tab3:
    st.markdown('<div class="sub-header">📉 Task 3: Distribution Modeling & Tail Integrity</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("""
        **Objective:** Analyze extreme hazard events (PM2.5 > 200 μg/m³) for industrial zones.
        
        **Problem:** Traditional histograms hide rare events in the "long tail".
        
        **Solution:** Two plots - one optimized for peaks, one for tails.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        tech_text = load_text_file("task3_technical_justification.txt")
        if tech_text:
            st.markdown(create_download_button(tech_text, "task3_technical_justification.txt", "Download Technical Justification"), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">80.8</div><div class="metric-label">99th Percentile</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">169.9</div><div class="metric-label">Maximum Value</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">0</div><div class="metric-label">Events >200</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">23.9</div><div class="metric-label">Mean PM2.5</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Distribution Visualizations")
    
    img1 = load_image("task3_peaks_plot.png")
    if img1:
        st.image(img1, caption="Figure 1: Optimized for Peaks (Main Body of Distribution)", use_container_width=True)
    
    img2 = load_image("task3_tails_plot.png")
    if img2:
        st.image(img2, caption="Figure 2: Optimized for Tails (Rare Events)", use_container_width=True)
    
    img3 = load_image("task3_comparison.png")
    if img3:
        st.image(img3, caption="Figure 3: Comparison - Why Tail Optimization is Necessary", use_container_width=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Key Insights from Task 3:**
    
    - **99th Percentile:** 80.8 μg/m³ (well below the 200 μg/m³ extreme threshold)
    - **No Extreme Events:** No PM2.5 readings exceeded 200 μg/m³ in the selected industrial zone
    - **Tail Behavior:** Log-scale histogram reveals the distribution's tail structure
    - **Honest Depiction:** The tail-optimized plot provides a more honest view of rare event probabilities
    - **Standard Histogram:** Hides the tail, making it appear empty beyond 150 μg/m³
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 4: VISUAL INTEGRITY AUDIT
# =============================================================================

with tab4:
    st.markdown('<div class="sub-header">🔍 Task 4: Visual Integrity Audit</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="task-card">', unsafe_allow_html=True)
        st.markdown("""
        **Objective:** Evaluate 3D bar chart proposal for Pollution vs Population Density vs Region.
        
        **Principles:** Lie Factor and Data-Ink Ratio.
        
        **Verdict:** REJECT the 3D bar chart.
        
        **Alternative:** Bivariate scatter plot.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        audit_text = load_text_file("task4_audit_report.txt")
        if audit_text:
            st.markdown(create_download_button(audit_text, "task4_audit_report.txt", "Download Audit Report"), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color: #D32F2F;">1.33</div><div class="metric-label">Lie Factor</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color: #D32F2F;">0.60</div><div class="metric-label">Data-Ink Ratio</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color: #2E7D32;">1.0</div><div class="metric-label">Alternative Lie Factor</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color: #2E7D32;">>0.9</div><div class="metric-label">Alternative Data-Ink</div></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Visual Integrity Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img = load_image("task4_3d_rejected.png")
        if img:
            st.image(img, caption="Figure 1: REJECTED - 3D Bar Chart Proposal", use_container_width=True)
    
    with col2:
        img = load_image("task4_alternative.png")
        if img:
            st.image(img, caption="Figure 2: ACCEPTED - Bivariate Mapping Alternative", use_container_width=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Key Insights from Task 4:**
    
    - **Lie Factor (1.33):** Exceeds acceptable range (0.95-1.05) - chart DISTORTS truth
    - **Data-Ink Ratio (0.60):** Too low - 40% of ink is wasted on non-data elements
    - **Occlusion:** Front bars hide back bars in 3D chart
    - **Alternative Solution:** Bivariate scatter plot shows all variables clearly
    - **Color Choice:** Sequential colors (red/blue) provide clear perceptual order
    - **Rainbow Rejected:** No perceptual order, uneven luminance, not colorblind-friendly
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown('<div class="footer">Urban Environmental Intelligence Challenge | Data Sources: OpenAQ, OpenWeather | 2025</div>', unsafe_allow_html=True)

# =============================================================================
# EXPANDER WITH ADDITIONAL INFORMATION
# =============================================================================

with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    ### Urban Environmental Intelligence Dashboard
    
    This interactive dashboard presents the complete analysis from the Urban Environmental Intelligence Challenge.
    
    **Tasks Overview:**
    
    1. **Dimensionality Reduction:** PCA analysis of 6 environmental variables across 100 sensors
    2. **Temporal Analysis:** High-density visualization of PM2.5 threshold violations
    3. **Distribution Modeling:** Analysis of extreme hazard events and tail behavior
    4. **Visual Integrity Audit:** Evaluation of 3D chart proposal vs alternative
    
    **Data Summary:**
    - 100 global sensor stations
    - Hourly measurements throughout 2025
    - 867,240 total data points
    - Variables: PM2.5, PM10, NO₂, O₃, Temperature, Humidity
    
    **How to Use:**
    - Navigate through tabs to explore each task
    - Click on images to enlarge
    - Download reports using the buttons provided
    - Hover over metrics for additional context
    """)