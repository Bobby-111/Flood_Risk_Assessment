import pandas as pd
import json

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# ── Gemini Configuration ──────────────────────────────────────────
# Set your API key in environment variable 'GEMINI_API_KEY'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gemini Configuration Error: {e}")
else:
    print("Gemini API Key missing. AI insights will use fallback logic.")

# ── Palette ───────────────────────────────────────────────────────
BG       = "#0f1117"
CARD_BG  = "#1a1d27"
BORDER   = "#2a2d3e"
ACCENT   = "#4f8ef7"
RED      = "#ef4444"
AMBER    = "#f59e0b"
GREEN    = "#10b981"
TEXT     = "#e2e8f0"
SUBTEXT  = "#94a3b8"

# Base layout (NO xaxis/yaxis — those are overridden per chart to avoid
# "got multiple values for keyword argument" TypeError when spreading **CHART_LAYOUT)
CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    title_font=dict(size=14, color=TEXT),
    margin=dict(t=48, b=36, l=48, r=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBTEXT))
)

# Axis defaults reused in each chart
AXIS = dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT)

def L(**overrides):
    """Merge CHART_LAYOUT with per-chart overrides safely."""
    return {**CHART_LAYOUT, **overrides}


MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}
MONTH_ORDER = [MONTH_NAMES[i] for i in range(1, 13)]

# ── Load Data ─────────────────────────────────────────────────────
df  = pd.read_csv("flood_predictions.csv")
raw = pd.read_csv("Flood_ML_Dataset_2015_2023.csv")

# Clean types for merging/filtering
df["year"]    = df["year"].astype(int)
df["month"]   = df["month"].astype(int)
df["grid_id"] = df["grid_id"].astype(str).str.replace(",", "", regex=False)

# ── PRE-PROCESS GRID GEOMETRIES (GeoJSON vs GeoDataFrame) ──────────
# We only need 538 unique grid shapes, not 58,000 duplicates!
# This avoids Geopandas/GDAL dependencies and saves ~100MB RAM.
raw_grids = raw.drop_duplicates("grid_id").copy()
raw_grids["grid_id"] = raw_grids["grid_id"].astype(str).str.replace(",", "", regex=False)

features = []
for _, row in raw_grids.iterrows():
    features.append({
        "type": "Feature",
        "id": row["grid_id"],
        "geometry": json.loads(row[".geo"]),
        "properties": {"grid_id": row["grid_id"]}
    })
GRIDS_GEOJSON = {"type": "FeatureCollection", "features": features}

# Clear raw from memory as we only need df now
del raw
del raw_grids

FEATURES = ["rainfall", "soil_moisture", "elevation",
            "slope", "TWI", "HAND", "river_distance", "drainage_density"]

YEARS  = sorted(df.year.unique())
MONTHS = sorted(df.month.unique())

# ── Palette (used in Plotly charts only) ──────────────────────────
CHART_BG   = "rgba(0,0,0,0)"
CARD_PLOT  = "#0d0d0d"
BORDER_PLT = "#1e1e1e"
TEXT_PLT   = "#f5f5f7"
SUB_PLT    = "#6e6e73"
AMBER      = "#f0a842"
RED        = "#ef4444"
GREEN      = "#10b981"
BLUE       = "#3b82f6"

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_PLOT, plot_bgcolor=CARD_PLOT,
    font=dict(color=TEXT_PLT, family="Inter, -apple-system, sans-serif", size=12),
    title_font=dict(size=13, color=TEXT_PLT, family="Inter"),
    margin=dict(t=44, b=32, l=44, r=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUB_PLT))
)
AXIS = dict(gridcolor=BORDER_PLT, zerolinecolor=BORDER_PLT, color=SUB_PLT)

def L(**overrides):
    return {**CHART_LAYOUT, **overrides}

# ── App ────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html lang="en">
<head>
    {%metas%}
    <title>Flood Risk Intelligence — Andhra Pradesh</title>
    {%favicon%}
    {%css%}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      /* ── Canvas Hero ──────────────────────────────────────── */
      #canvas-hero {
        position: relative;
        height: 600vh;          /* tall scroll container */
        background: #000;
      }
      #canvas-sticky {
        position: sticky;
        top: 0;
        height: 100vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }
      #hero-canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scale(1.12);      /* Zoom 12% to hide watermark */
        transform-origin: top;       /* Crop from the bottom */
      }
      /* overlay gradient */
      #canvas-sticky::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(
          to bottom,
          rgba(0,0,0,0.18) 0%,
          rgba(0,0,0,0.0)  40%,
          rgba(0,0,0,0.0)  60%,
          rgba(0,0,0,0.75) 100%
        );
        z-index: 1;
        pointer-events: none;
      }
      /* Text overlays that change with scroll */
      .canvas-text {
        position: absolute;
        z-index: 2;
        text-align: center;
        padding: 0 24px;
        pointer-events: none;
        transition: opacity 0.6s ease, transform 0.6s ease;
      }
      #ct-intro {
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 800px;
      }
      #ct-intro .eyebrow {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(240,168,66,0.9);
        margin-bottom: 16px;
        display: block;
      }
      #ct-intro h1 {
        font-size: clamp(42px, 7vw, 84px);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.02;
        color: #fff;
        margin-bottom: 20px;
      }
      #ct-intro h1 span {
        background: linear-gradient(135deg, #f0a842 0%, #f7d26b 50%, #e8916a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      #ct-intro p {
        font-size: clamp(16px, 2.2vw, 21px);
        font-weight: 300;
        color: rgba(255,255,255,0.75);
        line-height: 1.55;
      }
      #ct-phrase {
        bottom: 120px;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
      }
      #ct-phrase p {
        font-size: clamp(24px, 4vw, 52px);
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #fff;
        text-shadow: 0 2px 20px rgba(0,0,0,0.5);
      }
      #ct-phrase span { color: #f0a842; }
      /* scroll cue */
      #cv-scroll-cue {
        position: absolute;
        bottom: 36px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 3;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        animation: cvCueFade 2s ease 1.8s forwards;
        opacity: 0;
      }
      @keyframes cvCueFade {
        0%   { opacity: 0; transform: translateX(-50%) translateY(10px); }
        40%  { opacity: 1; transform: translateX(-50%) translateY(0); }
        80%  { opacity: 0.8; }
        100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
      }
      #cv-scroll-cue span {
        font-size: 10px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.45);
      }
      .cv-line {
        width: 1px;
        height: 38px;
        background: linear-gradient(to bottom, rgba(255,255,255,0.5), transparent);
        animation: cvLinePulse 1.8s ease-in-out infinite;
      }
      @keyframes cvLinePulse {
        0%, 100% { opacity: 0.5; }
        50%       { opacity: 1; }
      }
      /* Progress bar */
      #scroll-progress {
        position: fixed;
        top: 0; left: 0;
        height: 2px;
        background: linear-gradient(90deg, #f0a842, #f7d26b);
        z-index: 9999;
        transition: width 0.05s linear;
        width: 0%;
        display: none;
      }
    </style>
</head>
<body>
    <div id="scroll-progress"></div>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
    <script>
    (function() {
      /* ====================================================
         1. SCROLL-DRIVEN CANVAS FRAME ANIMATION
         Like Apple iPhone scroll on apple.com
      ==================================================== */
      var FRAME_COUNT = 150;
      var FRAME_BASE  = '/assets/frames/frame_';
      var frames      = [];
      var canvas, ctx;

      function padNum(n) {
        return ('000' + n).slice(-4);
      }

      function initCanvas() {
        canvas = document.getElementById('hero-canvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');
        sizeCanvas();
        window.addEventListener('resize', sizeCanvas, { passive: true });
        preloadFrames();
      }

      function sizeCanvas() {
        if (!canvas) return;
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        drawFrame(frames[currentFrame] || null);
      }

      function preloadFrames() {
        var loaded = 0;
        for (var i = 0; i < FRAME_COUNT; i++) {
          (function(idx) {
            var img = new Image();
            img.onload = function() {
              loaded++;
              if (idx === 0) drawFrame(img);
              if (loaded === FRAME_COUNT) onAllLoaded();
            };
            img.src = FRAME_BASE + padNum(idx) + '.jpg';
            frames[idx] = img;
          })(i);
        }
      }

      function onAllLoaded() {
        /* hide intro text and show canvas scroll phrase */
        drawFrame(frames[0]);
      }

      var currentFrame = 0;

      function drawFrame(img) {
        if (!ctx || !img || !img.complete) return;
        var cW = canvas.width,  cH = canvas.height;
        var iW = img.naturalWidth || 1280, iH = img.naturalHeight || 720;
        var scale = Math.max(cW / iW, cH / iH);
        var dW = iW * scale, dH = iH * scale;
        var dx = (cW - dW) / 2, dy = (cH - dH) / 2;
        ctx.clearRect(0, 0, cW, cH);
        ctx.drawImage(img, dx, dy, dW, dH);
      }

      /* ── Phrase overlays at different scroll positions ── */
      var phrases = [
        { start: 0.15, end: 0.45, text: ['Predicting for', '<span>East Godavari</span>', 'District'] },
        { start: 0.45, end: 0.70, text: ['ML calculates risk', 'across every', 'winding river.'] },
        { start: 0.70, end: 0.95, text: ['538 grid zones.', '9 years of history.'] },
      ];

      function updateOverlays(progress) {
        var introEl  = document.getElementById('ct-intro');
        var phraseEl = document.getElementById('ct-phrase');
        if (!introEl || !phraseEl) return;

        /* hide intro text once we scroll */
        var introOpacity = Math.max(0, 1 - progress * 8);
        introEl.style.opacity = introOpacity;
        introEl.style.transform = 'translate(-50%, calc(-50% - ' + (progress * 150) + 'px))';

        /* show scroll phrases with clean cross-fade */
        var phraseOpacity = 0;
        var phraseHtml    = '';
        for (var i = 0; i < phrases.length; i++) {
          var p = phrases[i];
          if (progress >= p.start && progress < p.end) {
            var local = (progress - p.start) / (p.end - p.start);
            // Smooth bell curve fade
            var fade  = Math.sin(local * Math.PI);
            phraseOpacity = fade;
            phraseHtml    = p.text.join('<br>');
            break;
          }
        }
        phraseEl.style.opacity   = phraseOpacity;
        phraseEl.style.transform = 'translate(-50%, calc(' + (10 - phraseOpacity*10) + 'px))';
        phraseEl.innerHTML = '<p>' + phraseHtml + '</p>';
      }

      /* ── Master scroll handler ── */
      var heroSection = null;
      var nav = null;
      var progressBar = null;

      function onScroll() {
        if (!heroSection) {
          heroSection   = document.getElementById('canvas-hero');
          nav           = document.getElementById('top-nav');
          progressBar   = document.getElementById('scroll-progress');
        }

        var scrollY = window.scrollY;

        /* nav glass effect */
        if (nav) nav.classList.toggle('scrolled', scrollY > 60);

        /* total page progress for top bar */
        var pageH = document.body.scrollHeight - window.innerHeight;
        if (progressBar) {
          progressBar.style.display = 'block';
          progressBar.style.width   = (scrollY / pageH * 100) + '%';
        }

        /* canvas hero progress */
        if (!heroSection) return;
        var heroTop    = heroSection.offsetTop;
        var heroHeight = heroSection.offsetHeight - window.innerHeight;
        var progress   = Math.max(0, Math.min(1, (scrollY - heroTop) / heroHeight));

        /* map progress → frame index */
        var idx = Math.min(FRAME_COUNT - 1, Math.floor(progress * FRAME_COUNT));
        if (idx !== currentFrame) {
          currentFrame = idx;
          drawFrame(frames[idx]);
        }

        updateOverlays(progress);
      }

      window.addEventListener('scroll', onScroll, { passive: true });

      /* ── Scroll-reveal ── */
      function initReveals() {
        var io = new IntersectionObserver(function(entries) {
          entries.forEach(function(e) {
            if (e.isIntersecting) {
              e.target.classList.add('visible');
              io.unobserve(e.target);
            }
          });
        }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
        document.querySelectorAll('.reveal, .stat-item').forEach(function(el) {
          io.observe(el);
        });
      }

      /* ── KPI card 3D tilt ── */
      function initTilt() {
        document.querySelectorAll('.kpi-card').forEach(function(card) {
          card.addEventListener('mousemove', function(e) {
            var r = card.getBoundingClientRect();
            var x = (e.clientX - r.left) / r.width  - 0.5;
            var y = (e.clientY - r.top)  / r.height - 0.5;
            card.style.transform = 'perspective(600px) rotateX(' + (-y*10) + 'deg) rotateY(' + (x*10) + 'deg) translateY(-6px) scale(1.01)';
          });
          card.addEventListener('mouseleave', function() {
            card.style.transform = '';
          });
        });
      }

      /* ── Smooth scroll anchors ── */
      document.addEventListener('click', function(e) {
        var a = e.target.closest('[data-scroll]');
        if (a) {
          e.preventDefault();
          var t = document.getElementById(a.getAttribute('data-scroll'));
          if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        var href = e.target.closest('a[href^="#"]');
        if (href) {
          e.preventDefault();
          var id  = href.getAttribute('href').slice(1);
          var el  = document.getElementById(id);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });

      /* ── Init ── */
      function init() {
        initCanvas();
        initReveals();
        initTilt();
      }

      setTimeout(init, 500);
      window._dashPrivate_mGlobalAfterRenderHooks = window._dashPrivate_mGlobalAfterRenderHooks || [];
      window._dashPrivate_mGlobalAfterRenderHooks.push(init);
    })();
    </script>
</body>
</html>
'''

# ── Helpers ────────────────────────────────────────────────────────
def chart_card(label, *children, icon=None, cls=""):
    header = [html.P(label, className="chart-label")]
    if icon:
        header.insert(0, html.Span(icon, className="chart-card-icon"))
    return html.Div([
        html.Div(header, className="chart-card-header"),
        *children
    ], className=f"chart-card {cls}")

def kpi_card(label, val_id, color, icon, klass):
    return html.Div([
        html.Span(icon, className="kpi-icon"),
        html.P(label, className="kpi-label"),
        html.Div(id=val_id, className="kpi-value", style={"color": color})
    ], className=f"kpi-card {klass} reveal")

# ── Layout ─────────────────────────────────────────────────────────
app.layout = html.Div(children=[

    # ── Sticky nav ──────────────────────────────────────────────────
    html.Nav(id="top-nav", children=[
        html.Span("🌊 East Godavari Flood Risk Intelligence", className="nav-brand"),
        html.Ul(className="nav-links", children=[
            html.Li(html.A("Overview",   href="#stat-band")),
            html.Li(html.A("Risk Map",   href="#charts-section")),
            html.Li(html.A("Analytics",  href="#charts-section")),
            html.Li(html.A("Data Table", href="#table-section")),
        ])
    ]),

    # ── Scroll-driven Canvas Hero ────────────────────────────────
    html.Section(id="canvas-hero", children=[
        html.Div(id="canvas-sticky", children=[

            # The canvas that renders frames
            html.Canvas(id="hero-canvas"),

            # Intro overlay — fades out as you start scrolling
            html.Div(id="ct-intro", className="canvas-text", children=[
                html.Span("East Godavari · ML-Powered · 2015 – 2023", className="eyebrow"),
                html.H1(children=[
                    "Flood Risk ", html.Span("Intelligence"), html.Br(), "System"
                ]),
                html.P("Real-time predictive analytics for the East Godavari District.")
            ]),

            # Dynamic scroll phrase — changes as frames progress
            html.Div(id="ct-phrase", className="canvas-text"),

            # Scroll cue
            html.Div(id="cv-scroll-cue", children=[
                html.Span("Scroll to explore"),
                html.Div(className="cv-line")
            ]),

            # CTA pill appears mid-scroll
            html.A("Enter Dashboard", className="hero-cta",
                   id="canvas-cta",
                   style={"position":"absolute","bottom":"48px","left":"50%",
                          "transform":"translateX(-50%)","zIndex":"3",
                          "display":"none"},
                   **{"data-scroll": "kpi-section"})
        ])
    ]),


    # ── Stat band ───────────────────────────────────────────────────
    html.Section(id="stat-band", children=[
        html.Div(className="stat-item", children=[
            html.Div("538", className="stat-num"),
            html.P("Grid Zones Monitored", className="stat-label")
        ]),
        html.Div(className="stat-item", children=[
            html.Div("9", className="stat-num"),
            html.P("Years of Data (2015–2023)", className="stat-label")
        ]),
        html.Div(className="stat-item", children=[
            html.Div("8", className="stat-num"),
            html.P("Predictive Features", className="stat-label")
        ]),
        html.Div(className="stat-item", children=[
            html.Div("ML", className="stat-num"),
            html.P("AI-Powered Predictions", className="stat-label")
        ]),
    ]),

    # ── KPI Section ─────────────────────────────────────────────────
    html.Section(id="kpi-section", children=[
        html.Div(className="kpi-row", children=[
            kpi_card("Avg Flood Probability", "kpi-avg",  AMBER, "📊", "kpi-amber"),
            kpi_card("High-Risk Zones",       "kpi-high", RED,   "⚠️", "kpi-red"),
            kpi_card("Medium-Risk Zones",     "kpi-med",  BLUE,  "📈", "kpi-blue"),
            kpi_card("Low-Risk Zones",        "kpi-low",  GREEN, "✅", "kpi-green"),
        ])
    ]),

    # ── Analyst Insights Section ────────────────────────────────────
    html.Section(id="insights-section", children=[
        chart_card("Executive Analyst Insights",
            dcc.Loading(
                id="loading-insights",
                type="dot",
                color=ACCENT,
                children=html.Div(id="insights-content", className="insights-text")
            ),
            icon="🔬",
            cls="reveal"
        ),
    ]),

    # ── Controls bar (sticky) ────────────────────────────────────────
    html.Div(id="controls-section", children=[
        html.Span("Filter data by period", className="controls-label"),
        html.Div(className="controls-inputs", children=[
            html.Div([
                html.Label("YEAR", className="ctrl-label"),
                dcc.Dropdown(
                    id="year",
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    value=YEARS[-1], clearable=False,
                    style={"width": "120px", "color": "#000000"}  # Forced black text
                )
            ]),
            html.Div([
                html.Label("MONTH", className="ctrl-label"),
                dcc.Dropdown(
                    id="month",
                    options=[{"label": MONTH_NAMES[m], "value": m} for m in MONTHS],
                    value=6, clearable=False,
                    style={"width": "120px", "color": "#000000"}  # Forced black text
                )
            ]),
            html.Button([
                html.Span("📥"), " Export Data"
            ], id="btn-export", className="export-btn"),
            dcc.Download(id="download-dataframe-csv"),
        ])
    ]),

    # ── Charts Section ───────────────────────────────────────────────
    html.Section(id="charts-section", children=[

        # Intro text
        html.Div(className="charts-intro reveal", children=[
            html.Span("Analytics", className="section-chip"),
            html.H2("Every dimension of flood risk,\nin one view.", className="section-title"),
            html.P("Spatial maps, temporal trends, feature correlations and "
                   "historical risk patterns — all updating live with your selection.",
                   className="section-body"),
        ]),

        html.Div(style={"maxWidth":"1400px","margin":"0 auto"}, children=[

            # ── Full-width map ──────────────────────────────────────
            html.Div(className="chart-card-full reveal", children=[
                html.P("Spatial Flood Risk Map", className="chart-label"),
                dcc.Graph(id="flood_map", style={"height":"520px"},
                          config={"displayModeBar": False})
            ]),

            # ── Timeline + YoY ──────────────────────────────────────
            html.Div(className="chart-grid chart-grid-2", children=[
                chart_card("Monthly Flood Probability",
                    dcc.Graph(id="timeline", style={"height":"320px"},
                              config={"displayModeBar": False})
                ),
                html.Div(className="reveal reveal-delay-1", children=[
                    chart_card("Year-over-Year Comparison",
                        dcc.Graph(id="yoy", style={"height":"320px"},
                                  config={"displayModeBar": False})
                    )
                ])
            ]),

            # ── Scatter + Heatmap ────────────────────────────────────
            html.Div(className="chart-grid chart-grid-3-2", style={"marginTop":"16px"}, children=[
                chart_card("Rainfall vs Flood Probability",
                    dcc.Graph(id="scatter", style={"height":"320px"},
                              config={"displayModeBar": False})
                ),
                html.Div(className="reveal reveal-delay-2", children=[
                    chart_card("Historical Risk Heatmap  (Month × Year)",
                        dcc.Graph(id="heatmap", style={"height":"320px"},
                                  config={"displayModeBar": False})
                    )
                ])
            ]),

            # ── Feature bars + Risk donut ────────────────────────────
            html.Div(className="chart-grid chart-grid-feat", style={"marginTop":"16px"}, children=[
                chart_card("Feature Comparison  (0–1 Normalised)",
                    dcc.Graph(id="feature_importance", style={"height":"320px"},
                              config={"displayModeBar": False})
                ),
                html.Div(className="reveal reveal-delay-1", children=[
                    chart_card("Risk-Level Distribution",
                        dcc.Graph(id="risk_dist", style={"height":"320px"},
                                  config={"displayModeBar": False})
                    )
                ])
            ]),
        ])
    ]),

    # ── Table section ────────────────────────────────────────────────
    html.Section(id="table-section", children=[
        html.Div(className="charts-intro reveal", children=[
            html.Span("Detailed Data", className="section-chip"),
            html.H2("Top 15 highest-risk zones.", className="section-title"),
            html.P("Sorted by flood probability. Click any column header to re-sort.",
                   className="section-body"),
        ]),
        html.Div(className="table-card reveal reveal-delay-1", children=[
            dash_table.DataTable(
                id="top_zones",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#000",
                    "color": "#6e6e73",
                    "fontWeight": "600",
                    "fontSize": "10px",
                    "letterSpacing": "0.12em",
                    "border": "0",
                    "borderBottom": "1px solid #1e1e1e",
                    "textTransform": "uppercase",
                    "padding": "14px 20px",
                    "fontFamily": "Inter, sans-serif"
                },
                style_cell={
                    "backgroundColor": "#000",
                    "color": "#f5f5f7",
                    "border": "0",
                    "borderBottom": "1px solid #111",
                    "fontSize": "14px",
                    "padding": "14px 20px",
                    "fontFamily": "Inter, sans-serif"
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{flood_probability} > "0.70"'},
                     "color": RED, "fontWeight": "600"},
                    {"if": {"filter_query": '{flood_probability} > "0.40" && {flood_probability} <= "0.70"'},
                     "color": AMBER},
                    {"if": {"row_index": "even"},
                     "backgroundColor": "#050505"},
                ],
                page_size=15,
                sort_action="native"
            )
        ])
    ]),

    # ── Footer ──────────────────────────────────────────────────────
    html.Footer(id="site-footer", children=[
        html.Span("🌊 Flood Risk Intelligence System", className="footer-brand"),
        html.Span("Andhra Pradesh Flood ML Dashboard · 2015 – 2023 · Built with Dash & Plotly",
                  className="footer-copy")
    ])

])


# ── Callbacks ─────────────────────────────────────────────────────

# KPI values
@app.callback(
    Output("kpi-avg",  "children"),
    Output("kpi-high", "children"),
    Output("kpi-med",  "children"),
    Output("kpi-low",  "children"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_kpis(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)]
    if dff.empty:
        return "—", "—", "—", "—"
    avg_prob  = dff["flood_probability"].mean()
    high_risk = (dff["flood_probability"] > 0.7).sum()
    med_risk  = ((dff["flood_probability"] >= 0.4) & (dff["flood_probability"] <= 0.7)).sum()
    low_risk  = (dff["flood_probability"] < 0.4).sum()
    return f"{avg_prob:.1%}", f"{int(high_risk):,}", f"{int(med_risk):,}", f"{int(low_risk):,}"


# Flood map
@app.callback(
    Output("flood_map", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_map(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)]
    if dff.empty:
        return go.Figure(layout=dict(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                                     font=dict(color=TEXT)))

    fig = px.choropleth_mapbox(
        dff, geojson=GRIDS_GEOJSON, locations="grid_id",
        color="flood_probability",
        mapbox_style="carto-darkmatter",
        center={"lat": 16.7, "lon": 82.0}, zoom=8,
        opacity=0.8,
        color_continuous_scale="YlOrRd",
        range_color=[0, 1],
        hover_data={"flood_probability": ":.2%", "grid_id": True,
                    "rainfall": ":.1f", "elevation": ":.1f"}
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor=CARD_BG,
        coloraxis_colorbar=dict(
            title="Flood Prob.",
            tickformat=".0%",
            bgcolor=CARD_BG,
            tickfont=dict(color=TEXT),
            titlefont=dict(color=TEXT)
        )
    )
    return fig


# Monthly timeline
@app.callback(
    Output("timeline", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_timeline(year, month):
    year, month = int(year), int(month)
    dff = df[df.year == year]
    tl = dff.groupby("month")["flood_probability"].mean().reset_index()
    tl["month_name"] = tl["month"].map(MONTH_NAMES)

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=tl["month_name"], y=tl["flood_probability"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=7, color=ACCENT),
        fill="tozeroy", fillcolor=f"rgba(79,142,247,0.12)",
        name="Avg Probability"
    ))

    # Highlight selected month
    mn = MONTH_NAMES.get(month)
    if mn and mn in tl["month_name"].values:
        sp = tl.loc[tl["month_name"] == mn, "flood_probability"].values[0]
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=mn, x1=mn, y0=0, y1=1,
                      line=dict(color=AMBER, width=1.5, dash="dot"))
        fig.add_trace(go.Scatter(
            x=[mn], y=[sp], mode="markers+text",
            marker=dict(size=14, color=AMBER, symbol="diamond"),
            text=[f"{sp:.1%}"], textposition="top center",
            textfont=dict(color=AMBER, size=11),
            name=mn, showlegend=False
        ))

    fig.update_layout(**L(
        yaxis=dict(tickformat=".0%", range=[0,1], **AXIS),
        xaxis=AXIS,
        title=f"Monthly Profile — {year}"
    ))
    return fig


# Year-over-Year comparison
@app.callback(
    Output("yoy", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_yoy(year, month):
    year, month = int(year), int(month)
    dff = df[df.month == month]
    yoy = dff.groupby("year")["flood_probability"].mean().reset_index()

    colors = [RED if y == year else ACCENT for y in yoy["year"]]

    fig = go.Figure(go.Bar(
        x=yoy["year"].astype(str),
        y=yoy["flood_probability"],
        marker_color=colors,
        text=[f"{v:.1%}" for v in yoy["flood_probability"]],
        textposition="outside",
        textfont=dict(size=10, color=TEXT)
    ))
    mn = MONTH_NAMES.get(month, str(month))
    fig.update_layout(**L(
        yaxis=dict(tickformat=".0%", range=[0, yoy["flood_probability"].max()*1.3], **AXIS),
        xaxis=AXIS,
        title=f"YoY — {mn} (highlighted: {year})",
        showlegend=False
    ))
    return fig


# Scatter: Rainfall vs Flood Probability
@app.callback(
    Output("scatter", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_scatter(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)].copy()
    if dff.empty:
        return go.Figure(layout=L(title="No data"))

    bins   = [0, 0.4, 0.7, 1.0]
    labels = ["Low", "Medium", "High"]
    # astype(str) avoids Categorical dtype crashing plotly trendline grouping
    dff["risk"] = pd.cut(dff["flood_probability"], bins=bins, labels=labels, include_lowest=True).astype(str)

    color_map = {"Low": GREEN, "Medium": AMBER, "High": RED}

    fig = px.scatter(
        dff, x="rainfall", y="flood_probability",
        color="risk", color_discrete_map=color_map,
        opacity=0.55, size_max=6,
        labels={"rainfall": "Rainfall (mm)", "flood_probability": "Flood Probability",
                "risk": "Risk Level"},
        trendline="lowess", trendline_scope="overall",
        trendline_color_override=ACCENT
    )
    mn = MONTH_NAMES.get(month, str(month))
    fig.update_layout(**L(
        yaxis=dict(tickformat=".0%", **AXIS),
        xaxis=AXIS,
        title=f"Rainfall vs Risk — {mn} {year}"
    ))
    return fig


# Calendar heatmap
@app.callback(
    Output("heatmap", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_heatmap(year, month):
    year, month = int(year), int(month)
    pivot = (df.groupby(["year","month"])["flood_probability"]
               .mean()
               .reset_index()
               .pivot(index="month", columns="year", values="flood_probability"))
    pivot.index = [MONTH_NAMES[m] for m in pivot.index]
    pivot = pivot.reindex(MONTH_ORDER)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index,
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="black"),  # Black text for better contrast on yellow/orange
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Avg Risk: %{z:.2%}<extra></extra>",
        colorbar=dict(
            tickformat=".0%", bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color=TEXT_PLT), titlefont=dict(color=TEXT_PLT), title="Risk"
        )
    ))

    # Highlight selected cell with a rectangle shape
    if str(year) in [str(c) for c in pivot.columns] and MONTH_NAMES.get(month) in pivot.index:
        x_idx = [str(c) for c in pivot.columns].index(str(year))
        y_idx = list(pivot.index).index(MONTH_NAMES[month])
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x_idx - 0.5, x1=x_idx + 0.5,
            y0=y_idx - 0.5, y1=y_idx + 0.5,
            line=dict(color=ACCENT, width=3)
        )

    fig.update_layout(**L(
        title="Historical Risk Heatmap",
        xaxis=dict(side="bottom", **AXIS),
        yaxis=dict(autorange="reversed", **AXIS)
    ))
    return fig


# Feature importance bar
@app.callback(
    Output("feature_importance", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_features(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)]
    if dff.empty:
        return go.Figure(layout=L(title="No data"))

    available = [f for f in FEATURES if f in dff.columns]
    means     = dff[available].mean()

    # Normalise to 0–1 using global dataset range so all features are comparable
    # (river_distance ~471 vs drainage_density ~0.54 makes raw bars unreadable)
    f_min = df[available].min()
    f_max = df[available].max()
    normalised = ((means - f_min) / (f_max - f_min).replace(0, 1)).sort_values()

    raw_labels = [f"{means[feat]:.2f}" for feat in normalised.index]
    bar_colors = [RED if v == normalised.max() else ACCENT for v in normalised.values]

    fig = go.Figure(go.Bar(
        x=normalised.values,
        y=normalised.index,
        orientation="h",
        marker_color=bar_colors,
        text=raw_labels,          # show actual value, not the 0-1 score
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate="<b>%{y}</b><br>Raw mean: %{text}<br>Normalised: %{x:.2f}<extra></extra>"
    ))
    mn = MONTH_NAMES.get(month, str(month))
    fig.update_layout(**L(
        title=f"Feature Comparison (0–1 normalised) — {mn} {year}",
        xaxis=dict(range=[0, 1.15], tickformat=".0%", **AXIS),
        yaxis=AXIS,
        showlegend=False
    ))
    return fig


# Risk distribution donut
@app.callback(
    Output("risk_dist", "figure"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_risk_dist(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)].copy()
    if dff.empty:
        return go.Figure(layout=L(title="No data"))

    bins   = [0, 0.4, 0.7, 1.0]
    labels = ["Low (<40%)", "Medium (40–70%)", "High (>70%)"]
    dff["risk_level"] = pd.cut(dff["flood_probability"], bins=bins, labels=labels, include_lowest=True).astype(str)
    counts = dff["risk_level"].value_counts().reindex(labels, fill_value=0)

    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.6,
        marker=dict(colors=[GREEN, BLUE, RED],
                    line=dict(color="#000", width=2)),
        textinfo="percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="<b>%{label}</b><br>%{value:,} zones<br>%{percent}<extra></extra>"
    ))
    mn = MONTH_NAMES.get(month, str(month))
    fig.update_layout(**L(
        title=f"Risk Probabilities — {mn} {year}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    ))
    return fig


# Top 15 zones table
@app.callback(
    Output("top_zones", "data"),
    Output("top_zones", "columns"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_table(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)].copy()
    if dff.empty:
        return [], []

    show_cols  = ["grid_id", "flood_probability", "rainfall",
                  "soil_moisture", "elevation", "river_distance"]
    available  = [c for c in show_cols if c in dff.columns]
    top15      = dff.nlargest(15, "flood_probability")[available].copy()

    top15["flood_probability"] = top15["flood_probability"].map(lambda x: f"{x:.2%}")
    for col in ["rainfall","soil_moisture","elevation","river_distance"]:
        if col in top15.columns:
            top15[col] = top15[col].map(lambda x: f"{x:.2f}")

    rename = {
        "grid_id":          "Grid ID",
        "flood_probability":"Flood Prob.",
        "rainfall":         "Rainfall (mm)",
        "soil_moisture":    "Soil Moisture",
        "elevation":        "Elevation (m)",
        "river_distance":   "River Dist. (m)"
    }
    top15.rename(columns=rename, inplace=True)

    columns = [{"name": c, "id": c} for c in top15.columns]
    return top15.to_dict("records"), columns


# Dynamic Analyst Insights (AI-Powered)
@app.callback(
    Output("insights-content", "children"),
    Input("year",  "value"),
    Input("month", "value")
)
def update_insights(year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)]
    if dff.empty:
        return dcc.Markdown("No data available for the selected period.")

    avg_prob = dff["flood_probability"].mean()
    high_risk_count = len(dff[dff["flood_probability"] > 0.7])
    max_rainfall = dff["rainfall"].max()
    max_rain_grid = dff.loc[dff["rainfall"].idxmax(), "grid_id"]
    mn = MONTH_NAMES.get(month, str(month))

    # Historical comparison
    prev_year_df = df[(df.year == year - 1) & (df.month == month)]
    comparison_text = "No historical baseline for comparison."
    if not prev_year_df.empty:
        prev_avg = prev_year_df["flood_probability"].mean()
        diff = (avg_prob - prev_avg) / (prev_avg if prev_avg != 0 else 1)
        direction = "increase" if diff > 0 else "decrease"
        comparison_text = f"{abs(diff):.1%} {direction} vs {mn} {year-1}"

    # Use Gemini with multi-model fallback
    if GEMINI_API_KEY:
        # Try models in order of speed/availability (matching user's diagnostic)
        for model_name in ["models/gemini-2.0-flash", "models/gemini-flash-latest", "models/gemini-pro-latest", "models/gemini-pro"]:
            try:
                temp_model = genai.GenerativeModel(model_name)
                prompt = f"""
                You are an expert Flood Risk Analyst for the East Godavari District.
                Generate a professional, concise executive 3-line summary for {mn} {year}.
                Metrics:
                - Avg Flood Probability: {avg_prob:.1%}
                - High-Risk Zones (>70%): {high_risk_count}
                - Max Rainfall: {max_rainfall:.1f}mm (Grid {max_rain_grid})
                - Trend: {comparison_text}
                
                Focus on spatial risk and primary drivers. Use bolding for numbers.
                Return ONLY the summary text in Markdown format.
                """
                response = temp_model.generate_content(prompt)
                if response and response.text:
                    return dcc.Markdown(response.text)
            except Exception as e:
                # If it's a 404, we don't want to print the whole stack trace every time
                error_msg = str(e)
                if "404" in error_msg:
                    print(f"Gemini {model_name} not available (404).")
                else:
                    print(f"Gemini {model_name} Error: {error_msg}")
                continue # Try next model

    # Hardcoded Fallback Logic
    markdown_text = f"""
    #### EXECUTIVE RISK SUMMARY — {mn.upper()} {year}
    In the **East Godavari District**, the average flood probability for {mn} {year} is **{avg_prob:.1%}**.
    Currently, there are **{high_risk_count} critical zones** flagged as high-risk regions (>70% probability).
    This represents a **{comparison_text}**.

    The highest concentrated rainfall recorded this month is **{max_rainfall:.1f}mm** at Grid **{max_rain_grid}**,
    which remains a primary factor driving localized predictive risk. Analysts should prioritize monitoring for the
    high-saturation grid clusters where hydraulic values correlate most strongly.
    """
    return dcc.Markdown(markdown_text)


# CSV Export
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("year", "value"),
    State("month", "value"),
    prevent_initial_call=True
)
def export_csv(n_clicks, year, month):
    year, month = int(year), int(month)
    dff = df[(df.year == year) & (df.month == month)].copy()
    if dff.empty:
        return None

    filename = f"Flood_Risk_Data_{MONTH_NAMES[month]}_{year}.csv"
    return dcc.send_data_frame(dff.to_csv, filename, index=False)


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)