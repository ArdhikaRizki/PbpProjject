"""
ReserView - Advanced Reservoir Volumetric Analysis Platform
Kelompok Villa Temanggung - IF-B 2025/2026

A modern application for subsurface mapping and reservoir volume calculations
with interactive 3D visualization and comprehensive reporting capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
from typing import Tuple, Dict, List
import io
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration management"""
    APP_NAME = "Kelompok Villa Temanggung"
    APP_ICON = "🏡"
    MIN_POINTS = 4
    INTERPOLATION_RESOLUTION = 100
    
    TEAM = [
        ("Ardhika Rizki Akbar Pratama", "123230057"),
        ("Saif Ali Addamawi", "123230164"),
        ("Ilham Cesario Putra Wippri", "123230106"),
        ("Sayyid Fakhri Nurjundi", "123230172"),
        ("Akmal Abrisam", "123230084"),
        ("Muhammad Rizal Wahyu Dharmawan", "123230200"),
        ("Elyuzar Fazlurrahman", "123230216"),
        ("Anak Agung Ngurah Sadewa Tedja", "123230050"),
        ("Akfina Ni'mawati", "123230076"),
        ("Rakha Taufiqurrahman Faisal Aziz", "123230071"),
        ("Adi Dwi Pambudi", "123230170"),
        ("Yediya Elshama Santosa", "123230174"),
        ("Muhammad Azmi Nasril", "123230190"),
        ("Laksana Atmaja Putra", "123230235"),
        ("Athallah Joyoningrat", "123230230"),
        ("Muhammad Azmi Nasril", "123230190"),
        ("Gorbi Ello Pasaribu", "123230083"),
        ("⁠Celsi Fransisca Sitompul", "123230015"),
        ("Priska Natalia Sembiring", "123230055"),
        ("Kurniasari Salasa Mubarokhati", "123230236"),
        ("Muhammad Irham Hadi Putra", "123230042"),
    ]
    
    DEFAULT_PETRO = {
        'porosity': 0.20,
        'sw': 0.30,
        'ntg': 0.80,
        'bo': 1.20,
        'bg': 0.005
    }


CFG = Config()


# ==================== THEME & STYLING ====================
class ThemeManager:
    """Manages application styling and theming"""
    
    @staticmethod
    def apply_theme():
        """Inject custom CSS for modern glassmorphic UI"""
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .main .block-container { padding: 2rem 1rem; }
            
            /* Animated gradient header */
            .app-title {
                font-size: 3.5rem;
                font-weight: 800;
                text-align: center;
                background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradient 3s ease infinite;
                letter-spacing: -1px;
                margin-bottom: 0.5rem;
            }
            
            @keyframes gradient {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            .app-subtitle {
                text-align: center;
                color: #94a3b8;
                font-size: 1.1rem;
                margin-bottom: 2rem;
                font-weight: 400;
            }
            
            /* Glassmorphic cards */
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(16px) saturate(180%);
                border: 1px solid rgba(255, 255, 255, 0.12);
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            div[data-testid="stMetric"]:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 48px rgba(102, 126, 234, 0.25);
            }
            
            div[data-testid="stMetric"] label {
                color: #94a3b8 !important;
                font-weight: 600 !important;
                font-size: 0.875rem !important;
            }
            
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: #f1f5f9 !important;
                font-size: 1.75rem !important;
                font-weight: 700 !important;
            }
            
            /* Enhanced buttons */
            .stButton>button, .stDownloadButton>button {
                border-radius: 12px;
                font-weight: 600;
                padding: 0.75rem 1.5rem;
                border: none;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                letter-spacing: 0.3px;
            }
            
            .stButton>button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }
            
            .stDownloadButton>button {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
            }
            
            .stDownloadButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(245, 87, 108, 0.5);
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
            }
            
            /* Modern tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: rgba(255, 255, 255, 0.04);
                padding: 8px;
                border-radius: 12px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px;
                color: #94a3b8;
                font-weight: 600;
                padding: 10px 20px;
                transition: all 0.2s;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
            
            /* Input styling */
            .stNumberInput>div>div>input, .stTextInput>div>div>input {
                background: rgba(255, 255, 255, 0.06) !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 8px !important;
                color: white !important;
            }
            
            .stNumberInput>div>div>input:focus, .stTextInput>div>div>input:focus {
                border-color: rgba(102, 126, 234, 0.5) !important;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
            }
            
            /* Slider enhancement */
            .stSlider>div>div>div>div {
                background: linear-gradient(90deg, #667eea, #764ba2) !important;
            }
            
            /* Alert boxes */
            .stAlert {
                background: rgba(255, 255, 255, 0.06) !important;
                backdrop-filter: blur(16px);
                border-radius: 12px !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
            }
            
            /* Team card */
            .team-info {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(16px);
                padding: 1.25rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            
            .team-info small {
                color: #cbd5e1;
                line-height: 1.75;
            }
            
            hr {
                border-color: rgba(255, 255, 255, 0.08) !important;
                margin: 1.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)


# ==================== REPORT GENERATION ====================
class PDFReportBuilder:
    """Generates professional PDF reports"""
    
    @staticmethod
    def generate(volumes: Dict, params: Dict) -> io.BytesIO:
        """Create comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title styling
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER,
            spaceAfter=24,
            fontName='Helvetica-Bold'
        )
        
        # Build document
        elements.append(Paragraph("Reservoir Volumetric Analysis", title_style))
        elements.append(Paragraph(
            f"<i>Report Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}</i>",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.4*inch))
        
        # Parameters table
        elements.append(Paragraph("Analysis Parameters", styles['Heading2']))
        elements.append(Spacer(1, 0.15*inch))
        
        param_data = [
            ['Parameter', 'Value'],
            ['Data Points', f"{params['points']}"],
            ['GOC Depth', f"{params['goc']:.2f} m"],
            ['WOC Depth', f"{params['woc']:.2f} m"],
            ['X Range', f"{params['x_min']:.1f} - {params['x_max']:.1f} m"],
            ['Y Range', f"{params['y_min']:.1f} - {params['y_max']:.1f} m"],
            ['Z Range', f"{params['z_min']:.1f} - {params['z_max']:.1f} m"],
        ]
        
        param_table = Table(param_data, colWidths=[3*inch, 3*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(param_table)
        elements.append(Spacer(1, 0.4*inch))
        
        # Volumetrics table
        elements.append(Paragraph("Volumetric Results", styles['Heading2']))
        elements.append(Spacer(1, 0.15*inch))
        
        vol_data = [
            ['Zone', 'Volume (m³)', 'Volume (MM m³)'],
            ['Gas Cap', f"{volumes['gas']:,.0f}", f"{volumes['gas']/1e6:.3f}"],
            ['Oil Zone', f"{volumes['oil']:,.0f}", f"{volumes['oil']/1e6:.3f}"],
            ['Total', f"{volumes['total']:,.0f}", f"{volumes['total']/1e6:.3f}"]
        ]
        
        vol_table = Table(vol_data, colWidths=[2*inch, 2*inch, 2*inch])
        vol_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1.5, colors.black)
        ]))
        elements.append(vol_table)
        
        doc.build(elements)
        buffer.seek(0)
        return buffer


class ExcelReportBuilder:
    """Generates detailed Excel reports"""
    
    @staticmethod
    def generate(volumes: Dict, params: Dict, raw_df: pd.DataFrame) -> io.BytesIO:
        """Create multi-sheet Excel workbook"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Parameters sheet
            params_df = pd.DataFrame({
                'Parameter': ['Points', 'GOC', 'WOC', 'X_min', 'X_max', 'Y_min', 'Y_max', 'Z_min', 'Z_max'],
                'Value': [
                    params['points'], params['goc'], params['woc'],
                    params['x_min'], params['x_max'],
                    params['y_min'], params['y_max'],
                    params['z_min'], params['z_max']
                ]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Volumes sheet
            vol_df = pd.DataFrame({
                'Zone': ['Gas Cap', 'Oil Zone', 'Total'],
                'Volume_m3': [volumes['gas'], volumes['oil'], volumes['total']],
                'Volume_MMm3': [volumes['gas']/1e6, volumes['oil']/1e6, volumes['total']/1e6]
            })
            vol_df.to_excel(writer, sheet_name='Volumes', index=False)
            
            # Raw data
            raw_df.to_excel(writer, sheet_name='Data', index=False)
        
        buffer.seek(0)
        return buffer


# ==================== COMPUTATIONAL ENGINE ====================
class ReservoirCalculator:
    """Core computation engine for volumetric analysis"""
    
    @staticmethod
    def interpolate_surface(df: pd.DataFrame, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate interpolated grid surface from sparse data points"""
        # Remove duplicate XY coordinates by averaging Z values
        df_clean = df.groupby(['X', 'Y'], as_index=False)['Z'].mean()
        
        # Create meshgrid
        x_range = np.linspace(df['X'].min(), df['X'].max(), resolution)
        y_range = np.linspace(df['Y'].min(), df['Y'].max(), resolution)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        
        # Interpolate with fallback
        try:
            grid_z = griddata(
                points=(df_clean['X'].values, df_clean['Y'].values),
                values=df_clean['Z'].values,
                xi=(grid_x, grid_y),
                method='cubic'
            )
        except:
            grid_z = griddata(
                points=(df_clean['X'].values, df_clean['Y'].values),
                values=df_clean['Z'].values,
                xi=(grid_x, grid_y),
                method='linear'
            )
        
        return grid_x, grid_y, grid_z
    
    @staticmethod
    def compute_volumes(grid_x: np.ndarray, grid_y: np.ndarray, 
                       grid_z: np.ndarray, goc: float, woc: float) -> Dict:
        """Calculate gross rock volumes for each zone"""
        # Cell dimensions
        dx = (grid_x[0, -1] - grid_x[0, 0]) / (grid_x.shape[1] - 1)
        dy = (grid_y[-1, 0] - grid_y[0, 0]) / (grid_y.shape[0] - 1)
        cell_area = dx * dy
        
        # Thickness calculations
        thickness_to_woc = np.maximum(0, woc - grid_z)
        vol_total = np.nansum(thickness_to_woc) * cell_area
        
        thickness_to_goc = np.maximum(0, goc - grid_z)
        vol_gas = np.nansum(thickness_to_goc) * cell_area
        
        vol_oil = max(0, vol_total - vol_gas)
        
        return {
            'gas': vol_gas,
            'oil': vol_oil,
            'total': vol_total
        }
    
    @staticmethod
    def compute_reserves(volumes: Dict, phi: float, sw: float, 
                        ntg: float, bo: float, bg: float) -> Dict:
        """Calculate hydrocarbon initially in place"""
        stoiip = (volumes['oil'] * ntg * phi * (1 - sw)) / bo
        giip = (volumes['gas'] * ntg * phi * (1 - sw)) / bg
        
        return {
            'stoiip': stoiip,
            'giip': giip
        }


# ==================== VISUALIZATION ENGINE ====================
class PlotlyVisualizer:
    """Advanced 2D/3D visualization using Plotly"""
    
    @staticmethod
    def contour_map(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                    df: pd.DataFrame, goc: float, woc: float) -> go.Figure:
        """Create structural contour map with fluid contacts"""
        fig = go.Figure()
        
        # Base contour surface
        z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
        fig.add_trace(go.Contour(
            z=grid_z,
            x=grid_x[0],
            y=grid_y[:, 0],
            colorscale='Turbo',
            contours=dict(
                start=z_min,
                end=z_max,
                size=(z_max - z_min) / 25,
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title="Depth (m)", thickness=20)
        ))
        
        # GOC contour line
        try:
            fig.add_trace(go.Contour(
                z=grid_z,
                x=grid_x[0],
                y=grid_y[:, 0],
                contours=dict(start=goc, end=goc, size=0.1, coloring='lines'),
                line=dict(color='gold', width=4, dash='dash'),
                showscale=False,
                name='GOC',
                hovertemplate='GOC: %{z:.1f}m<extra></extra>'
            ))
        except:
            pass
        
        # WOC contour line
        try:
            fig.add_trace(go.Contour(
                z=grid_z,
                x=grid_x[0],
                y=grid_y[:, 0],
                contours=dict(start=woc, end=woc, size=0.1, coloring='lines'),
                line=dict(color='cyan', width=4, dash='dot'),
                showscale=False,
                name='WOC',
                hovertemplate='WOC: %{z:.1f}m<extra></extra>'
            ))
        except:
            pass
        
        # Data points overlay
        fig.add_trace(go.Scatter(
            x=df['X'],
            y=df['Y'],
            mode='markers',
            marker=dict(color='white', size=7, line=dict(color='black', width=1.5)),
            name='Control Points',
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Structural Contour Map", font=dict(size=20)),
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            height=700,
            hovermode='closest',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
        )
        
        return fig
    
    @staticmethod
    def surface_3d(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                   goc: float, woc: float) -> go.Figure:
        """Create interactive 3D surface model"""
        fig = go.Figure()
        
        # Main geological surface
        fig.add_trace(go.Surface(
            z=grid_z,
            x=grid_x,
            y=grid_y,
            colorscale='Earth',
            opacity=0.9,
            name='Formation',
            colorbar=dict(title="Depth (m)", x=1.1),
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}m<extra></extra>'
        ))
        
        # GOC reference plane
        fig.add_trace(go.Surface(
            z=goc * np.ones_like(grid_z),
            x=grid_x,
            y=grid_y,
            colorscale=[[0, 'rgba(255,215,0,0.4)'], [1, 'rgba(255,215,0,0.4)']],
            showscale=False,
            name='GOC',
            hovertemplate='GOC: %{z:.1f}m<extra></extra>'
        ))
        
        # WOC reference plane
        fig.add_trace(go.Surface(
            z=woc * np.ones_like(grid_z),
            x=grid_x,
            y=grid_y,
            colorscale=[[0, 'rgba(0,255,255,0.4)'], [1, 'rgba(0,255,255,0.4)']],
            showscale=False,
            name='WOC',
            hovertemplate='WOC: %{z:.1f}m<extra></extra>'
        ))
        
        fig.update_layout(
            title="3D Reservoir Structure",
            height=750,
            scene=dict(
                xaxis_title="Easting (m)",
                yaxis_title="Northing (m)",
                zaxis=dict(title="Depth (m)", autorange="reversed"),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
                aspectmode='cube'
            )
        )
        
        return fig
    
    @staticmethod
    def cross_section(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                     goc: float, woc: float, y_slice: float) -> go.Figure:
        """Create 2D cross-sectional view"""
        idx = np.argmin(np.abs(grid_y[:, 0] - y_slice))
        
        fig = go.Figure()
        
        # Formation profile
        fig.add_trace(go.Scatter(
            x=grid_x[0, :],
            y=grid_z[idx, :],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2563eb', width=3),
            fillcolor='rgba(37, 99, 235, 0.25)',
            name='Formation',
            hovertemplate='X: %{x:.1f}m<br>Depth: %{y:.1f}m<extra></extra>'
        ))
        
        # Fluid contact lines
        fig.add_hline(
            y=goc,
            line_dash="dash",
            line_color="gold",
            line_width=3,
            annotation_text="GOC",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=woc,
            line_dash="dot",
            line_color="cyan",
            line_width=3,
            annotation_text="WOC",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f"Cross-Section at Y = {y_slice:.0f}m",
            xaxis_title="Easting (m)",
            yaxis_title="Depth (m)",
            height=600,
            yaxis=dict(autorange="reversed"),
            hovermode='x unified'
        )
        
        return fig


# ==================== DATA MANAGEMENT ====================
class DataManager:
    """Handles all data input/output operations"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state for data storage"""
        if 'points' not in st.session_state:
            st.session_state.points = []
    
    @staticmethod
    def add_point(x: float, y: float, z: float):
        """Add new data point to session"""
        st.session_state.points.append({'X': x, 'Y': y, 'Z': z})
    
    @staticmethod
    def load_from_file(file) -> bool:
        """Load data from uploaded CSV/Excel file"""
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            df.columns = [c.upper() for c in df.columns]
            
            if not {'X', 'Y', 'Z'}.issubset(df.columns):
                return False
            
            st.session_state.points.extend(df[['X', 'Y', 'Z']].to_dict('records'))
            return True
        except:
            return False
    
    @staticmethod
    def export_backup() -> str:
        """Export current session as JSON"""
        return json.dumps(st.session_state.points, indent=2)
    
    @staticmethod
    def restore_backup(file) -> bool:
        """Restore session from JSON backup"""
        try:
            st.session_state.points = json.load(file)
            return True
        except:
            return False
    
    @staticmethod
    def clear_all():
        """Clear all data points"""
        st.session_state.points = []
    
    @staticmethod
    def get_dataframe() -> pd.DataFrame:
        """Get current data as DataFrame"""
        return pd.DataFrame(st.session_state.points)


# ==================== UI COMPONENTS ====================
class UIBuilder:
    """Builds all user interface components"""
    
    @staticmethod
    def render_header():
        """Display application header"""
        st.markdown(f'<h1 class="app-title">{CFG.APP_ICON} {CFG.APP_NAME}</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-subtitle">Advanced Subsurface Mapping & Volumetric Analysis Platform</p>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_team_sidebar():
        """Display team information in sidebar"""
        st.markdown("### 👥 Team Members")
        html = '<div class="team-info">'
        for name, nim in CFG.TEAM:
            html += f'<small><b>{name}</b><br>{nim}</small><br><br>'
        html = html.rstrip('<br><br>') + '</div>'
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def render_data_input_panel() -> pd.DataFrame:
        """Render comprehensive data input panel"""
        DataManager.initialize_session()
        
        with st.expander("📊 Data Input & Management", expanded=False):
            # Tabs for different input methods
            tab1, tab2, tab3 = st.tabs(["✏️ Manual Entry", "📂 Batch Import", "💾 Session Management"])
            
            # Tab 1: Manual entry
            with tab1:
                with st.form("point_input", clear_on_submit=True):
                    st.markdown("**Enter coordinates:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x = st.number_input("Easting (X)", value=0.0, step=50.0, help="X coordinate in meters")
                    with col2:
                        y = st.number_input("Northing (Y)", value=0.0, step=50.0, help="Y coordinate in meters")
                    with col3:
                        z = st.number_input("Depth (Z)", value=1000.0, step=10.0, help="Depth in meters")
                    
                    submitted = st.form_submit_button("➕ Add Point", width="stretch", type="primary")
                    if submitted:
                        DataManager.add_point(x, y, z)
                        st.success(f"✅ Point added: X={x}, Y={y}, Z={z}")
                        st.rerun()
            
            # Tab 2: File upload
            with tab2:
                st.markdown("**Upload your data file:**")
                st.caption("Supported formats: CSV, Excel (.xlsx)")
                
                file = st.file_uploader(
                    "Choose file",
                    type=["csv", "xlsx"],
                    help="File must contain columns: X, Y, Z"
                )
                
                if file:
                    try:
                        # Preview data
                        preview_df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                        preview_df.columns = [c.upper() for c in preview_df.columns]
                        
                        st.markdown("**Preview:**")
                        st.dataframe(preview_df.head(), width="stretch")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.info(f"📊 Found {len(preview_df)} data points")
                        with col2:
                            if st.button("🚀 Import Data", width="stretch", type="primary"):
                                file.seek(0)  # Reset file pointer
                                if DataManager.load_from_file(file):
                                    st.success("✅ Data imported successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid file format!")
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
            
            # Tab 3: Session management
            with tab3:
                st.markdown("**Manage your session data:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 💾 Backup")
                    st.caption("Save current data to JSON file")
                    st.download_button(
                        "📥 Download Backup",
                        data=DataManager.export_backup(),
                        file_name=f"reservoir_backup_{datetime.now():%Y%m%d_%H%M%S}.json",
                        mime="application/json",
                        width="stretch"
                    )
                
                with col2:
                    st.markdown("##### 📥 Restore")
                    st.caption("Load data from backup file")
                    restore = st.file_uploader("Choose backup file", type=["json"], key="restore_file")
                    if restore:
                        if st.button("Load Backup", width="stretch", type="primary"):
                            if DataManager.restore_backup(restore):
                                st.success("✅ Data restored successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid backup file!")
                
                st.divider()
                
                # Clear data section
                st.markdown("##### 🗑️ Clear Data")
                st.caption("Remove all data points from current session")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("⚠️ Clear All Data", width="stretch", type="secondary"):
                        DataManager.clear_all()
                        st.rerun()
        
        return DataManager.get_dataframe()
    
    @staticmethod
    def render_control_panel(df: pd.DataFrame) -> Dict:
        """Render parameter control panel"""
        st.header("⚙️ Analysis Parameters")
        
        min_z, max_z = df['Z'].min(), df['Z'].max()
        
        # Fluid contacts
        st.subheader("💧 Fluid Contacts")
        
        if min_z == max_z:
            st.warning(f"⚠️ All depths are {min_z:.1f}m. Add varying depths to enable controls.")
            goc = woc = min_z
        else:
            goc = st.slider(
                "Gas-Oil Contact (GOC)",
                float(min_z), float(max_z),
                float(min_z + (max_z - min_z) * 0.3)
            )
            woc = st.slider(
                "Water-Oil Contact (WOC)",
                float(min_z), float(max_z),
                float(min_z + (max_z - min_z) * 0.7)
            )
        
        # Petrophysics
        st.subheader("🔬 Petrophysical Properties")
        c1, c2 = st.columns(2)
        
        with c1:
            phi = st.number_input("Porosity (φ)", 0.05, 0.40, CFG.DEFAULT_PETRO['porosity'], 0.01)
            sw = st.number_input("Water Sat (Sw)", 0.1, 1.0, CFG.DEFAULT_PETRO['sw'], 0.05)
            ntg = st.number_input("Net-to-Gross", 0.1, 1.0, CFG.DEFAULT_PETRO['ntg'], 0.05)
        
        with c2:
            bo = st.number_input("Oil FVF (Bo)", 1.0, 2.0, CFG.DEFAULT_PETRO['bo'], 0.1)
            bg = st.number_input("Gas FVF (Bg)", 0.001, 0.1, CFG.DEFAULT_PETRO['bg'], 0.001, format="%.4f")
        
        st.info(f"📍 {len(df)} points | Depth: {min_z:.1f} - {max_z:.1f}m")
        
        return {
            'goc': goc, 'woc': woc, 'phi': phi,
            'sw': sw, 'ntg': ntg, 'bo': bo, 'bg': bg
        }
    
    @staticmethod
    def render_results_metrics(volumes: Dict, reserves: Dict):
        """Display calculated results as metrics"""
        st.markdown("### 📊 Volumetric Results")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        c1.metric("Gas Cap", f"{volumes['gas']/1e6:.2f} MM m³")
        c2.metric("Oil Zone", f"{volumes['oil']/1e6:.2f} MM m³")
        c3.metric("Total GRV", f"{volumes['total']/1e6:.2f} MM m³")
        c4.metric("GIIP", f"{reserves['giip']/1e9:.2f} BCF")
        c5.metric("STOIIP", f"{reserves['stoiip']/1e6:.2f} MMbbl")
    
    @staticmethod
    def render_welcome_screen():
        """Display welcome screen when no data"""
        st.markdown("""
            <div style='text-align:center; padding:60px 20px;'>
                <h2 style='color:#94a3b8; font-size:2.5rem;'>🚀 Get Started</h2>
                <p style='color:#64748b; font-size:1.1rem; margin:20px 0;'>
                    No data loaded. Add points to begin analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.info(
                "**Quick Start:**\n\n"
                "1️⃣ Add data manually or upload CSV/Excel\n\n"
                "2️⃣ Set fluid contacts and properties\n\n"
                "3️⃣ View visualizations and export reports"
            )


# ==================== MAIN APPLICATION ====================
def main():
    """Application entry point"""
    # Page configuration
    st.set_page_config(
        page_title=CFG.APP_NAME,
        page_icon=CFG.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply theme
    ThemeManager.apply_theme()
    
    # Header
    UIBuilder.render_header()
    
    # Data input
    df = UIBuilder.render_data_input_panel()
    
    # Sidebar
    with st.sidebar:
        UIBuilder.render_team_sidebar()
        
        if not df.empty and len(df) >= CFG.MIN_POINTS:
            st.divider()
            params = UIBuilder.render_control_panel(df)
    
    # Main content area
    if df.empty:
        UIBuilder.render_welcome_screen()
        
    elif len(df) < CFG.MIN_POINTS:
        st.error(
            f"❌ Need at least **{CFG.MIN_POINTS}** points for analysis. "
            f"Current: **{len(df)}** points"
        )
        
    else:
        # Compute everything
        calc = ReservoirCalculator()
        grid_x, grid_y, grid_z = calc.interpolate_surface(df, CFG.INTERPOLATION_RESOLUTION)
        volumes = calc.compute_volumes(grid_x, grid_y, grid_z, params['goc'], params['woc'])
        reserves = calc.compute_reserves(
            volumes, params['phi'], params['sw'],
            params['ntg'], params['bo'], params['bg']
        )
        
        # Display results
        st.divider()
        UIBuilder.render_results_metrics(volumes, reserves)
        st.divider()
        
        # Visualization tabs
        viz = PlotlyVisualizer()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Contour Map",
            "🧊 3D Surface",
            "✂️ Cross-Section",
            "📥 Export"
        ])
        
        with tab1:
            fig_contour = viz.contour_map(grid_x, grid_y, grid_z, df, params['goc'], params['woc'])
            st.plotly_chart(fig_contour, use_container_width=True)
        
        with tab2:
            fig_3d = viz.surface_3d(grid_x, grid_y, grid_z, params['goc'], params['woc'])
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab3:
            y_values = np.unique(grid_y.flatten())
            selected_y = st.select_slider(
                "Select Y Position",
                options=y_values,
                value=y_values[len(y_values)//2]
            )
            fig_xs = viz.cross_section(grid_x, grid_y, grid_z, params['goc'], params['woc'], selected_y)
            st.plotly_chart(fig_xs, use_container_width=True)
        
        with tab4:
            st.markdown("#### 📄 Export Reports")
            
            # Prepare parameters for reports
            report_params = {
                'points': len(df),
                'goc': params['goc'],
                'woc': params['woc'],
                'x_min': df['X'].min(),
                'x_max': df['X'].max(),
                'y_min': df['Y'].min(),
                'y_max': df['Y'].max(),
                'z_min': df['Z'].min(),
                'z_max': df['Z'].max()
            }
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                pdf = PDFReportBuilder.generate(volumes, report_params)
                st.download_button(
                    "📄 PDF Report",
                    data=pdf,
                    file_name=f"reservoir_report_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
            
            with c2:
                excel = ExcelReportBuilder.generate(volumes, report_params, df)
                st.download_button(
                    "📊 Excel Report",
                    data=excel,
                    file_name=f"reservoir_report_{datetime.now():%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )
            
            with c3:
                grid_df = pd.DataFrame({
                    'X': grid_x.flatten(),
                    'Y': grid_y.flatten(),
                    'Z': grid_z.flatten()
                })
                st.download_button(
                    "💾 Grid CSV",
                    data=grid_df.to_csv(index=False),
                    file_name=f"grid_data_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    width="stretch"
                )
            
            st.divider()
            st.markdown("#### 📋 Raw Data")
            st.dataframe(
                df,
                width="stretch",
                height=350
            )


if __name__ == "__main__":
    main()
