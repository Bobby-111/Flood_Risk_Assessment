# 🌊 Flood Risk Intelligence — East Godavari

An **ML-powered flood risk analytics dashboard** for the East Godavari District, Andhra Pradesh. Built with Plotly Dash, GeoPandas, and the Google Gemini AI API.

![Dashboard Preview](assets/hero.png)

---

## ✨ Features

- **Scroll-driven cinematic hero** — Apple-style frame-by-frame animation on scroll
- **Interactive spatial risk map** — choropleth map showing flood probability per grid zone
- **Live KPI cards** — average probability, high/medium/low-risk zone counts
- **AI Executive Insights** — Gemini-powered analyst summaries for each time period
- **Temporal analytics** — monthly timeline, year-over-year comparison, risk heatmap
- **Feature analysis** — rainfall vs flood probability scatter, feature importance bars
- **Top 15 risk zones table** — sortable data table with colour-coded risk levels
- **CSV export** — download filtered data with one click

---

## 📁 Project Structure

```
Flood/
├── flood_dashboard.py      # Main Dash application
├── assets/
│   ├── style.css           # Global dark-theme stylesheet
│   ├── hero.png            # Hero image for README / OG image
│   └── frames/             # Scroll animation frames (generated locally)
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

> **Note:** The large data files (`Flood_ML_Dataset_2015_2023.csv`, `flood_predictions.csv`) and the source video (`D_Animation_Video_Generation.mp4`) are **not included** in this repository due to size. Download them separately (see below).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/flood-risk-dashboard.git
cd flood-risk-dashboard
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real [Google Gemini API key](https://aistudio.google.com/app/apikey):

```
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 5. Add the data files

Place the following files in the project root (not included in the repo):

| File | Description |
|------|-------------|
| `Flood_ML_Dataset_2015_2023.csv` | Raw ML dataset with geometry (2015–2023) |
| `flood_predictions.csv` | Model prediction outputs |

### 6. Generate scroll animation frames *(optional)*

The hero section uses ~150 JPEG frames extracted from `D_Animation_Video_Generation.mp4`. Place your frames in `assets/frames/` named `frame_0000.jpg` … `frame_0149.jpg`.

### 7. Run the dashboard

```bash
python flood_dashboard.py
```

Open your browser at **http://127.0.0.1:8050**

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI insights. Dashboard runs without it (uses fallback logic). |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Plotly Dash](https://dash.plotly.com/) |
| Geospatial | [GeoPandas](https://geopandas.org/), [Shapely](https://shapely.readthedocs.io/) |
| Visualisation | [Plotly](https://plotly.com/python/) |
| AI Insights | [Google Gemini](https://ai.google.dev/) |
| Styling | Vanilla CSS (dark theme) |

---

## 📊 Data

- **Source:** East Godavari District flood monitoring data (2015–2023)
- **Coverage:** 538 grid zones across the district
- **Features:** Rainfall, soil moisture, elevation, slope, TWI, HAND, river distance, drainage density

---

## 📄 License

MIT © Bharath Chilaka
