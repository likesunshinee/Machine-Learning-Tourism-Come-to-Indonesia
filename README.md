# Nusantara Transit Intelligence

## Indonesian Tourism Analytics Dashboard

A comprehensive Streamlit application for analyzing and forecasting Indonesian tourism data through multiple entry points and transportation modes.

### 🌟 Features

1. **Dashboard** - Executive overview with key insights and trends
2. **Cluster Explorer** - K-Means clustering analysis of entry points
3. **Forecasting** - SARIMA time-series forecasting (12-month ahead)
4. **Data Explorer** - Interactive data filtering and exploration
5. **Transport Insights** - Transportation mode analysis (Air, Land, Sea)
6. **Model Documentation** - Comprehensive model cards and methodology

### 🚀 Quick Start

#### Prerequisites
- Python 3.8+
- All dependencies in `requirements.txt`

#### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Data Structure
```
tourism_indonesia_forecasting/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── dataset/
│   └── tourism_indonesia_2016-2026_clean.csv
├── model/
│   ├── kmeans_clustering_final.pkl
│   ├── scaler_clustering.pkl
│   └── tourism_sarima_model.pkl
├── data_cleaning.ipynb         # Data preprocessing
├── clustering.ipynb            # K-Means clustering
└── forecasting.ipynb          # SARIMA forecasting
```

### 🐛 Bug Fixes

#### Critical Fix: Black `</div>` Artifact
**Root Cause:** HTML injection issue in transport illustration rendering
**Solution:** Replaced unsafe `st.markdown()` with isolated `components.html()`
- Moved SVG content to dedicated function `get_transport_svg()`
- Used safe HTML component rendering instead of f-string injection
- Eliminated stray closing tags through proper HTML isolation

### 🎨 Visual Design

**Theme:** Claymorphism Tropical
- Soft pastel backgrounds with tropical gradient
- Rounded clay-style cards with neumorphic shadows
- Consistent green typography (#1F5137)
- Professional transportation elements (✈🚌⛴)

### 📊 Models & Analytics

#### Clustering Model
- **Algorithm:** K-Means (k=2)
- **Features:** 14 engineered features (seasonal indices + volume + recovery)
- **Evaluation:** Silhouette=0.6295, Davies-Bouldin=0.7270
- **Results:** Cluster 0 (2 entry points), Cluster 1 (24 entry points)

#### Forecasting Model  
- **Algorithm:** SARIMA(0,1,1)(0,1,1,12)
- **Training:** 108 months (2016-2024)
- **Testing:** 12 months (2025)
- **Performance:** MAE=79,871, RMSE=88,965, MAPE=6.36%
- **Forecast:** 12 months ahead (Aug 2026 - Jul 2027)

### 🔄 Data Processing

#### Monthly Aggregation Logic
```python
# Matches forecasting.ipynb exactly
df_detail = df[df["Jenis Data"].isin(["Pintu Masuk", "Pintu Masuk Lainnya"])]
monthly = df_detail.groupby("Tanggal")["Jumlah"].sum(min_count=1)
monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp("MS")
```

#### Feature Engineering
- **Seasonal Indices:** 12 monthly patterns (2016-2019 baseline)
- **Log Mean Volume:** Log-transformed average visitor volume
- **Recovery 2025:** Percentage recovery vs 2019 baseline
- **Standardization:** MinMaxScaler for clustering input

### 📈 Key Insights

1. **Tourism Recovery:** 2025 shows significant recovery vs 2019 baseline
2. **Seasonal Patterns:** Clear monthly variations across entry points  
3. **Transportation Modes:** Air dominates, with land and sea variations
4. **Clustering:** Two distinct behavioral patterns identified
5. **Forecasting:** SARIMA provides reliable 12-month predictions

### ⚠️ Important Data Notes

- **Historical Data:** Available through July 2026
- **Forecast Data:** August 2026 - July 2027 are SARIMA predictions
- **Data Quality:** Cleaned dataset with proper missing value handling
- **No Data Leakage:** Strict train/test split methodology

### 🛠️ Technical Implementation

#### Safe HTML Rendering
- Eliminated malformed HTML injection
- Used `streamlit.components.v1.html()` for isolated rendering
- Proper SVG encapsulation without stray tags

#### Performance Optimization
- `@st.cache_data` for dataset loading
- `@st.cache_resource` for model loading
- Efficient pandas operations with proper indexing

#### Responsive Design
- `use_container_width=True` for all charts
- Flexible grid layouts with `st.columns()`
- Mobile-friendly clay card styling

### 🎯 Usage Instructions

1. **Dashboard:** Overview of tourism trends and key metrics
2. **Cluster Explorer:** Search/explore entry points by behavioral patterns
3. **Forecasting:** View SARIMA predictions with confidence intervals
4. **Data Explorer:** Filter and examine raw cleaned data
5. **Transport Insights:** Analyze air/land/sea transportation patterns
6. **Model Documentation:** Technical details and methodology

### 📝 Export Instructions

To export SARIMA model from `forecasting.ipynb`:
```python
import joblib
from pathlib import Path

model_artifact = {
    "model": final_fit,
    "model_name": "SARIMA(0,1,1)(0,1,1,12)",
    "target": "Jumlah wisatawan mancanegara",
    "frequency": "MS",
    "seasonal_period": 12,
    "training_start": str(monthly_known.index.min()),
    "training_end": str(monthly_known.index.max()),
}

joblib.dump(model_artifact, "model/tourism_sarima_model.pkl")
```

### 🎨 Color Palette

- **Primary Green:** #1F5137
- **Dark Green:** #133426  
- **Secondary Green:** #2D4D33
- **Background:** #E9F3DC
- **Cards:** #F7F7E9
- **Accent Colors:** #F1CC6D (yellow), #E8995D (orange), #8DC3D5 (blue)

### 🔍 Troubleshooting

**Model Loading Issues:**
- Ensure all `.pkl` files exist in `model/` directory
- Check Python version compatibility with joblib/sklearn versions
- Verify statsmodels version for SARIMA model loading

**Display Issues:**
- Clear Streamlit cache: `streamlit cache clear`
- Check browser compatibility (Chrome/Firefox recommended)
- Ensure proper UTF-8 encoding for Indonesian text

**Data Issues:**
- Verify dataset path and CSV format
- Check date parsing for "Tanggal" column
- Ensure consistent data types across columns

---

### 👥 Contributors

**Project:** Tourism Indonesia Forecasting Analysis
**Theme:** Nusantara Transit Intelligence  
**Technology:** Streamlit + SARIMA + K-Means
**Design:** Claymorphism Tropical