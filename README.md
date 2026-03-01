<div align="center">
  <img src="UI.jpeg" alt="Urban Environmental Intelligence Banner" width="100%" style="border-radius: 10px; margin-bottom: 20px;">
  
  # 🌆 Urban Environmental Intelligence Dashboard
  
  ### A Comprehensive Air Quality Analysis Platform for Smart Cities
  

  <br>
  
  **[Live Demo](https://urbanenvproject-ffoycfafpgr9nvhtq3zdnm.streamlit.app/)** • 
 
  <br>
  
  <img src="https://img.shields.io/github/last-commit/yourusername/urban-environmental-intelligence?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/yourusername/urban-environmental-intelligence?style=flat-square" />
  <img src="https://img.shields.io/github/issues/yourusername/urban-environmental-intelligence?style=flat-square" />
</div>

---

## 📋 **TABLE OF CONTENTS**
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Data Sources](#-data-sources)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Usage Instructions](#-usage-instructions)
- [Task Breakdown](#-task-breakdown)
- [Visualizations Gallery](#-visualizations-gallery)
- [Key Findings](#-key-findings)
- [Results & Insights](#-results--insights)
- [Dashboard Preview](#-dashboard-preview)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 **PROJECT OVERVIEW**

The **Urban Environmental Intelligence Dashboard** is a comprehensive data analysis platform designed for smart city initiatives. It processes and visualizes air quality data from **100 global sensor stations** throughout 2025 to identify environmental anomalies, track health threshold violations, and provide actionable insights for urban planning.

### **The Challenge**
- **Data Volume:** 867,240 hourly readings from 100 sensors
- **Variables:** PM2.5, PM10, NO₂, O₃, Temperature, Humidity
- **Time Period:** Full year 2025
- **Objective:** Build a diagnostic engine to identify environmental anomalies and pollution patterns

### **What We Accomplished**
| Task | Description | Key Technique |
|------|-------------|---------------|
| **Task 1** | Reduced 6-dimensional data to 2D using PCA | Principal Component Analysis |
| **Task 2** | Analyzed PM2.5 health threshold violations | High-density heatmap visualization |
| **Task 3** | Modeled extreme hazard events (>200 μg/m³) | Tail-inclusive distribution analysis |
| **Task 4** | Audited 3D chart proposal | Lie Factor & Data-Ink Ratio principles |

---

## ✨ **KEY FEATURES**

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/fluency/48/000000/statistics.png" width="40"/>
        <br><b>Dimensionality Reduction</b>
        <br><small>PCA analysis of 6 variables</small>
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/fluency/48/000000/time-machine.png" width="40"/>
        <br><b>Temporal Analysis</b>
        <br><small>24h & 30d pattern detection</small>
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/fluency/48/000000/bar-chart.png" width="40"/>
        <br><b>Distribution Modeling</b>
        <br><small>Tail-inclusive analysis</small>
      </td>
      <td align="center" width="25%">
        <img src="https://img.icons8.com/fluency/48/000000/inspection.png" width="40"/>
        <br><b>Visual Integrity Audit</b>
        <br><small>Lie Factor evaluation</small>
      </td>
    </tr>
  </table>
</div>

### **🌟 Highlight Features**
- **📊 Interactive Dashboard** - 5-tab Streamlit interface with all visualizations
- **📈 High-Density Visualizations** - Heatmaps showing 100 stations simultaneously
- **📉 Tail-Inclusive Analysis** - Log-scale histograms revealing rare events
- **🔍 Visual Integrity Auditing** - Lie Factor and Data-Ink Ratio calculations
- **📁 Export Functionality** - Downloadable summaries, reports, and high-resolution images

---

## 🔗 **DATA SOURCES**

This project uses two primary data sources:

<div align="center">
  <table>
    <tr>
      <th width="50%">
        <img src="https://img.icons8.com/fluency/48/000000/globe.png" width="30"/>
        <br>🌍 OpenAQ Global Air Quality API
      </th>
      <th width="50%">
        <img src="https://img.icons8.com/fluency/48/000000/partly-cloudy-day.png" width="30"/>
        <br>🌤️ OpenWeather API
      </th>
    </tr>
    <tr>
      <td>
        • PM2.5, PM10, NO₂, O₃ measurements<br>
        • 100 global monitoring stations<br>
        • Hourly readings throughout 2025
      </td>
      <td>
        • Temperature data<br>
        • Humidity measurements<br>
        • Meteorological context for pollution events
      </td>
    </tr>
  </table>
</div>

---

## 🛠️ **TECHNOLOGY STACK**

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
</div>

### **Core Dependencies**
```txt
streamlit==1.28.0      # Interactive web dashboard
pandas==2.0.3          # Data manipulation and analysis
numpy==1.24.3          # Numerical computing
matplotlib==3.7.2      # Data visualization
seaborn==0.12.2        # Statistical visualizations
scikit-learn==1.3.0    # Machine learning (PCA, KMeans)
scipy==1.11.1          # Scientific computing (statistics)
pillow==10.0.0         # Image processing
