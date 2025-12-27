import streamlit as st
import pandas as pd
import joblib
import io
import os
import datetime
import requests
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HopperAI | Professional Engineering Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #f8fafc; }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    /* RED BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background-color: #ef4444; 
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #dc2626;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }

    /* Section Headings */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 3px solid #ef4444;
        display: inline-block;
    }

    /* Professional Result Cards */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #ef4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #ef4444; 
        font-size: 2.25rem;
        font-weight: 800;
        margin-top: 5px;
    }
    .flow-hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_engine():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    clf_path = os.path.join(BASE_DIR, "model_classifier.pkl")
    regs_path = os.path.join(BASE_DIR, "model_regressors.pkl")
    
    try:
        clf = joblib.load(clf_path)
        regs = joblib.load(regs_path)
        return clf, regs
    except Exception as e:
        st.error(f"⚠️ Initialization Error: {e}")
        return None, None

clf, regs = load_engine()

# --- 4. GOOGLE SHEETS LOGGING BRIDGE ---
def log_to_sheets_bridge(data):
    try:
        url = st.secrets["connections"]["gsheets"]["script_url"]
        response = requests.post(url, data=json.dumps(data), timeout=10)
        return (True, "Success") if response.status_code == 200 else (False, f"HTTP {response.status_code}")
    except Exception as e:
        return False, str(e)

# --- 5. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1087/1087815.png", width=80)
    st.title("HopperAI v1.6")
    
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        st.caption("🔴 Database Connected")
    else:
        st.caption("⚪ Database Offline")
        
    st.markdown("---")
    st.subheader("📥 Material Inputs")
    bulk_rho = st.number_input("Bulk Density (kg/m³)", 200.0, 3000.0, 850.0)
    
    # REQUIRED: Tapped Density is needed to compute Hausner Ratio for the AI brain
    tapped_rho = st.number_input("Tapped Density (kg/m³)", bulk_rho, 4000.0, bulk_rho * 1.2)
    h_ratio = tapped_rho / bulk_rho
    st.info(f"Calculated Hausner Ratio: **{h_ratio:.3f}**")
    
    d50 = st.number_input("Particle Size (µm)", 1.0, 5000.0, 75.0)
    shape = st.selectbox("Particle Shape", ["Irregular", "Spherical", "Angular", "Granular"])
    
    st.markdown("---")
    generate_btn = st.button("🚀 CALCULATE & LOG DESIGN")

# --- 6. MAIN DASHBOARD ---
st.title("Industrial Hopper Design Suite")
st.markdown("Predictive Modeling for Mass and Funnel Flow Characteristics.")

if generate_btn:
    if clf and regs:
        # Columns must match the Trainer's 'input_cols' exactly
        input_df = pd.DataFrame([[bulk_rho, h_ratio, d50, shape]], 
                                columns=["Bulk Density - ρb (kg/m3)", "Hausner Ratio", "d50 (µm)", "Shape"])
        
        try:
            # Predict Flowability Class
            flow_status = clf.predict(input_df)[0]
            
            # Predict all Numerical Design Outputs
            # Standardize keys by stripping spaces
            results = {str(col).strip(): model.predict(input_df)[0] for col, model in regs.items()}

            # --- HERO SECTION: Flowability ---
            st.markdown(f"""
                <div class="flow-hero">
                    <p style="text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.9rem; opacity: 0.8;">Primary Prediction</p>
                    <h1 style="color: #ef4444; margin: 0; font-size: 3rem;">{flow_status}</h1>
                    <p style="margin-top: 10px; font-weight: 500;">Calculated Material Flowability Characterization</p>
                </div>
            """, unsafe_allow_html=True)

            # --- TECHNICAL DIMENSIONS GRID ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<p class="section-header">💠 Mass Flow (Conical)</p>', unsafe_allow_html=True)
                val_angle = results.get('Half Angle (°)', 0)
                val_outlet = results.get('Outlet Dimension NB', 0)
                
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Recommended Half Angle</div>
                        <div class="metric-value">{val_angle:.1f}°</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Outlet Dimension (NB)</div>
                        <div class="metric-value">{int(val_outlet)}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown('<p class="section-header">📐 Funnel Flow (Plane)</p>', unsafe_allow_html=True)
                val_plane_angle = results.get('Half Angle (°).1', 0)
                val_valley = results.get('Valley Angle - External (°)', 0)
                val_plane_outlet = results.get('Outlet Dimension NB.1', 0)
                
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Recommended Half Angle</div>
                        <div class="metric-value">{val_plane_angle:.1f}°</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Valley Angle (External)</div>
                        <div class="metric-value">{val_valley:.1f}°</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-label">Outlet Dimension (NB)</div>
                        <div class="metric-value">{int(val_plane_outlet)}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # --- LOGGING TO GOOGLE SHEETS ---
            log_data = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Density": bulk_rho, "Hausner": round(h_ratio, 3), "Size": d50, "Shape": shape, "Flow": flow_status,
                "Mass_Angle": round(float(val_angle), 2),
                "Mass_Outlet": int(val_outlet),
                "Funnel_Angle": round(float(val_plane_angle), 2),
                "Valley_Angle": round(float(val_valley), 2),
                "Funnel_Outlet": int(val_plane_outlet)
            }
            
            log_success, log_msg = log_to_sheets_bridge(log_data)
            if log_success: st.toast("✅ Design logged to Google Sheets!")

        except Exception as e:
            st.error(f"Computation Error: {e}")

        # --- REPORT GENERATION & DOWNLOAD ---
        st.write("---")
        report_df = pd.DataFrame({"Parameter": list(log_data.keys()), "Value": list(log_data.values())})
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Specification Report (CSV)",
            data=csv_data,
            file_name=f"hopper_specs_{datetime.date.today()}.csv",
            mime="text/csv"
        )
            
    else:
        st.error("System Error: AI Model Files Not Found.")
else:
    st.divider()
    st.info("💡 Adjust the material properties in the sidebar and click **CALCULATE & LOG DESIGN** to begin.")
