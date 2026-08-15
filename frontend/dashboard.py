import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# Page Setup
st.set_page_config(
    page_title="BotShield AI | Social Integrity Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ BotShield AI")
st.caption("Next-Gen Machine Learning Platform for Fake Social Media Account Detection")
st.divider()

# Sidebar - Quick Presets
st.sidebar.header("⚡ Quick Demo Presets")
preset = st.sidebar.selectbox(
    "Load sample account profile:",
    ["Custom Input", "Obvious Bot Network", "Celebrity / Authentic", "Suspicious Spammer"]
)

# Preset Values Setup
if preset == "Obvious Bot Network":
    p_u_len, p_u_dig, p_pic, p_bio, p_url = 14, 8, 0, 5, 1
    p_fol, p_fow, p_posts, p_age = 12, 3400, 1, 5
elif preset == "Celebrity / Authentic":
    p_u_len, p_u_dig, p_pic, p_bio, p_url = 8, 0, 1, 140, 1
    p_fol, p_fow, p_posts, p_age = 85000, 320, 650, 1200
elif preset == "Suspicious Spammer":
    p_u_len, p_u_dig, p_pic, p_bio, p_url = 12, 4, 1, 15, 1
    p_fol, p_fow, p_posts, p_age = 150, 4800, 12, 18
else:
    p_u_len, p_u_dig, p_pic, p_bio, p_url = 10, 2, 1, 45, 0
    p_fol, p_fow, p_posts, p_age = 350, 400, 45, 180

# Main Interactive Controls Layout
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("⚙️ Account Attributes Configuration")
    
    st.markdown("#### **Identity & Metadata**")
    username_len = st.slider("Username Length (Characters)", 1, 30, int(p_u_len))
    username_digits = st.slider("Digits in Username", 0, username_len, int(min(p_u_dig, username_len)))
    
    c1, c2 = st.columns(2)
    with c1:
        has_profile_pic = st.radio("Has Avatar / Profile Pic", [1, 0], index=0 if p_pic == 1 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
    with c2:
        has_external_url = st.radio("Has Bio Link / URL", [1, 0], index=0 if p_url == 1 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
        
    bio_length = st.number_input("Bio Character Count", min_value=0, max_value=500, value=int(p_bio))
    
    st.markdown("#### **Network & Activity Metrics**")
    followers = st.number_input("Exact Followers Count", min_value=0, max_value=1000000, value=int(p_fol), step=10)
    following = st.number_input("Exact Following Count", min_value=0, max_value=1000000, value=int(p_fow), step=10)
    posts_count = st.number_input("Total Posts Count", min_value=0, max_value=10000, value=int(p_posts), step=1)
    account_age_days = st.slider("Account Age (Days)", 1, 3650, int(p_age))

    analyze_btn = st.button("🔍 Run Machine Learning Risk Analysis", type="primary")

with col_right:
    st.subheader("📊 Real-Time Analytics Dashboard")
    
    if analyze_btn or preset != "Custom Input":
        payload = {
            "username_len": int(username_len),
            "username_digits": int(username_digits),
            "has_profile_pic": int(has_profile_pic),
            "bio_length": int(bio_length),
            "has_external_url": int(has_external_url),
            "followers": int(followers),
            "following": int(following),
            "posts_count": int(posts_count),
            "account_age_days": int(account_age_days)
        }
        
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                score = data["fake_probability"]
                risk_level = data["risk_level"]
                
                # Gauge Chart Visual
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Risk Rating: <b>{risk_level}</b>", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if score > 70 else ("#F59E0B" if score > 40 else "#10B981")},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Metric Summary Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Follower Ratio", f"{round(followers/(following+1), 2)}")
                m2.metric("Post Frequency", f"{round(posts_count/account_age_days, 2)} /day")
                m3.metric("Digit Density", f"{round((username_digits/username_len)*100, 1)}%")
                
                st.divider()
                
                # Explanations & Flags section
                st.markdown("#### 🚨 Risk Analysis & Explainable AI")
                if data["detected_flags"]:
                    for flag in data["detected_flags"]:
                        st.error(f"• {flag}")
                else:
                    st.success("• No anomalous bot behavioral patterns detected.")
                    
                # Additional Interactive Bar Graph for Network Ratio Comparison
                fig_net = px.bar(
                    x=["Followers", "Following"],
                    y=[followers, following],
                    color=["Followers", "Following"],
                    color_discrete_map={"Followers": "#3B82F6", "Following": "#8B5CF6"},
                    title="Network Imbalance Comparison"
                )
                fig_net.update_layout(height=230, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_net, use_container_width=True)
                
            else:
                st.error("Error from backend API.")
        except Exception as e:
            st.warning("⚡ Backend API not responding. Start the FastAPI server on port 8000.")
    else:
        st.info("👈 Adjust parameters on the left or click 'Run Machine Learning Risk Analysis' to trigger prediction.")