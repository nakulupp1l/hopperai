import streamlit as st
import pandas as pd
import joblib
import io

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
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Professional Result Cards */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e2e8f0;
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
        color: #0f172a;
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
    try:
        # These names must match your joblib.dump filenames
        clf = joblib.load("model_classifier.pkl")
        regs = joblib.load("model_regressors.pkl")
        return clf, regs
    except Exception as e:
        return None, None

clf, regs = load_engine()

# --- 4. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1087/1087815.png", width=80)
    st.title("HopperAI v1.1")
    st.markdown("---")
    st.subheader("📥 Material Inputs")
    density = st.number_input("Bulk Density (kg/m³)", 200.0, 3000.0, 850.0)
    d50 = st.number_input("Particle Size (µm)", 1.0, 5000.0, 75.0)
    shape = st.selectbox("Particle Shape", ["Spherical", "Angular", "Irregular", "Elongated"])
    st.markdown("---")
    generate_btn = st.button("🚀 CALCULATE DESIGN")

# --- 5. MAIN DASHBOARD ---
st.title("Industrial Hopper Design Suite")
st.markdown("Predictive Modeling for Mass and Funnel Flow Characteristics.")

if generate_btn:
    if clf and regs:
        # Prepare input data matching your training columns
        input_df = pd.DataFrame([[density, d50, shape]], 
                                columns=["Bulk Density - ρb (kg/m3)", "d50 (µm)", "Shape"])
        
        # Run AI Inference
        flow_status = clf.predict(input_df)[0]
        
        # FIXED: Correct dictionary unpacking using .items()
        results = {col: model.predict(input_df)[0] for col, model in regs.items()}

        # --- HERO SECTION: Flowability ---
        st.markdown(f"""
            <div class="flow-hero">
                <p style="text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.9rem; opacity: 0.8;">Primary Prediction</p>
                <h1 style="color: #3b82f6; margin: 0; font-size: 3rem;">{flow_status}</h1>
                <p style="margin-top: 10px; font-weight: 500;">Calculated Material Flowability Characterization</p>
            </div>
        """, unsafe_allow_html=True)

        # --- TECHNICAL DIMENSIONS GRID ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💠 Mass Flow (Conical)")
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Recommended Half Angle</div>
                    <div class="metric-value">{results.get('Half Angle (°)', 0):.1f}°</div>
                </div>
                <div class="metric-container">
                    <div class="metric-label">Outlet Dimension (NB)</div>
                    <div class="metric-value">{int(results.get('Outlet Dimension NB', 0))}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📐 Funnel Flow (Plane)")
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Recommended Half Angle</div>
                    <div class="metric-value">{results.get('Half Angle (°).1', 0):.1f}°</div>
                </div>
                <div class="metric-container">
                    <div class="metric-label">Valley Angle (External)</div>
                    <div class="metric-value">{results.get('Valley Angle - External (°)', 0):.1f}°</div>
                </div>
                <div class="metric-container">
                    <div class="metric-label">Outlet Dimension (NB)</div>
                    <div class="metric-value">{int(results.get('Outlet Dimension NB.1', 0))}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # --- REPORT GENERATION & DOWNLOAD ---
        st.write("---")
        
        report_data = {
            "Parameter": [
                "Bulk Density (kg/m³)", "Particle Size d50 (µm)", "Particle Shape",
                "Predicted Flowability",
                "Mass Flow: Half Angle (°)", "Mass Flow: Outlet Dimension (NB)",
                "Funnel Flow: Half Angle (°)", "Funnel Flow: Valley Angle (°)", "Funnel Flow: Outlet Dimension (NB)"
            ],
            "Value": [
                density, d50, shape,
                flow_status,
                f"{results.get('Half Angle (°)', 0):.1f}", int(results.get('Outlet Dimension NB', 0)),
                f"{results.get('Half Angle (°).1', 0):.1f}", f"{results.get('Valley Angle - External (°)', 0):.1f}", int(results.get('Outlet Dimension NB.1', 0))
            ]
        }
        report_df = pd.DataFrame(report_data)
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Specification Report (CSV)",
            data=csv_data,
            file_name="hopper_design_specs.csv",
            mime="text/csv"
        )
            
    else:
        st.error("System Error: AI Model Files Not Found. Ensure .pkl files are in the directory.")
else:
    st.divider()
    st.info("💡 Adjust the material properties in the sidebar and click **CALCULATE DESIGN** to begin.")