import os
import re
import pandas as pd
import streamlit as st
from utils.database import DatabaseManager
from utils.llm import LLMManager
from utils.visualization import VisualizationEngine
from utils.analytics import AnalyticsEngine

# Page configuration
st.set_page_config(
    page_title="Executive AI Business Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection (Dark Executive Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main layout colors */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Custom headers */
    .main-title {
        background: linear-gradient(135deg, #a5f3fc 0%, #6366f1 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 0 10px 30px rgba(99, 102, 241, 0.1);
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
        margin: 5px 0;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Styled code blocks and outputs */
    .stCodeBlock {
        background-color: #030712 !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    /* Interactive chat bubbles */
    .user-bubble {
        background-color: rgba(99, 102, 241, 0.15);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-style: italic;
    }
    
    .assistant-bubble {
        background-color: rgba(30, 41, 59, 0.3);
        border-left: 4px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        padding: 0 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: rgba(255, 255, 255, 0.03);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if "viz_engine" not in st.session_state:
    st.session_state.viz_engine = VisualizationEngine()
if "analytics_engine" not in st.session_state:
    st.session_state.analytics_engine = AnalyticsEngine()
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = ""
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "curr_file_name" not in st.session_state:
    st.session_state.curr_file_name = ""

# Sidebar - Settings & Configuration
st.sidebar.markdown("### ⚙️ Engine Configurations")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Ollama", "Gemini", "Groq", "OpenAI", "Hugging Face"],
    index=0
)

# API Keys and Models setup based on Provider
api_key = ""
host = "http://localhost:11434"
default_model = ""

if provider == "Ollama":
    host = st.sidebar.text_input("Ollama Host URL", value="http://localhost:11434")
    model_name = st.sidebar.text_input("Model Name", value="llama3.2:latest")
elif provider == "Gemini":
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Gemini API Key", value=env_key, type="password", help="Get a free key from Google AI Studio")
    model_name = st.sidebar.text_input("Model Name", value="gemini-2.5-flash")
elif provider == "Groq":
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.sidebar.text_input("Groq API Key", value=env_key, type="password")
    model_name = st.sidebar.text_input("Model Name", value="llama-3.3-70b-versatile")
elif provider == "OpenAI":
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("OpenAI API Key", value=env_key, type="password")
    model_name = st.sidebar.text_input("Model Name", value="gpt-4o-mini")
elif provider == "Hugging Face":
    env_key = os.getenv("HF_API_KEY", "")
    api_key = st.sidebar.text_input("Hugging Face API Token", value=env_key, type="password")
    model_name = st.sidebar.text_input("Model Name", value="Qwen/Qwen2.5-Coder-7B-Instruct")

# Instantiate LLM Manager in session
st.session_state.llm_manager = LLMManager(
    provider=provider,
    api_key=api_key,
    model_name=model_name,
    host=host
)

# Main Title Area
st.markdown('<div class="main-title">Executive Analyst AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turn data tables into structured SQL queries, forecast trends, detect anomalies, and uncover business growth areas using natural language.</div>', unsafe_allow_html=True)

# File Uploader Section
uploaded_file = st.file_uploader("Upload business dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Check if a new file is uploaded
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.name.split(".")[-1]}
    
    if st.session_state.curr_file_name != uploaded_file.name:
        with st.spinner("Parsing file and spinning up temporary SQL Database..."):
            # Reset manager
            st.session_state.db_manager = DatabaseManager()
            success, err_msg = st.session_state.db_manager.load_file(uploaded_file, file_details["FileType"])
            
            if success:
                st.session_state.curr_file_name = uploaded_file.name
                st.session_state.chat_history = [] # reset chat
                
                # Generate initial summary using LLM
                schema_sum = st.session_state.db_manager.get_schema_summary_text()
                
                # Check LLM key or status
                try:
                    init_summary = st.session_state.llm_manager.generate_dataset_summary(schema_sum)
                    if "ERROR_CALLING_LLM" in init_summary:
                        st.session_state.dataset_summary = "Dataset loaded successfully into SQLite database. Configure API credentials to unlock AI-generated executive summaries and natural language querying."
                        st.session_state.suggested_questions = [
                            "Show the highest value transactions",
                            "What is the row count?",
                            "List the unique categories in the dataset"
                        ]
                    else:
                        st.session_state.dataset_summary = init_summary
                        # Extract suggested questions from summary or write simple logic
                        lines = init_summary.split("\n")
                        questions = []
                        for line in lines:
                            if "?" in line:
                                # Clean up line numbers or bullet points
                                clean_q = re.sub(r'^\d+\.\s*|^\-\s*', '', line).strip()
                                # Double check it's not a header
                                if len(clean_q) > 10 and clean_q.endswith("?"):
                                    questions.append(clean_q)
                        st.session_state.suggested_questions = questions[:3] if questions else [
                            "Which categories have the highest totals?",
                            "Are there seasonal patterns in the dataset?",
                            "List the top 5 records sorted by key metrics"
                        ]
                except Exception as e:
                    st.session_state.dataset_summary = f"Dataset loaded. Error building AI overview: {str(e)}"
            else:
                st.error(f"Error loading file: {err_msg}")

# Display Dataset Overview if loaded
if st.session_state.db_manager.has_data:
    schema_info = st.session_state.db_manager.get_schema_info()
    
    # Render premium KPI metrics
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Database Status</div>
            <div class="kpi-val" style="color: #10b981;">ACTIVE (SQLITE)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Columns Loaded</div>
            <div class="kpi-val">{len(schema_info.get("columns", []))}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-val">{schema_info.get("row_count", 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # CDO / AI Summary Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Dataset Intelligence Overview")
    st.write(st.session_state.dataset_summary)
    st.markdown('</div>', unsafe_allow_html=True)

    # Core Tabs of App
    tab_chat, tab_forecast, tab_anomaly, tab_sandbox = st.tabs([
        "💬 Ask BI Assistant",
        "🔮 Time-Series Forecasting",
        "🚨 Anomaly Detector",
        "💻 SQL Sandbox & Schema"
    ])
    
    # ------------------ TAB 1: Chat Interface ------------------
    with tab_chat:
        st.write("Ask questions about your data in plain English. The AI will translate it into optimized SQL, fetch results, design visual graphs, and explain business findings.")
        
        # Suggested questions buttons
        q_selected = None
        if st.session_state.suggested_questions:
            st.write("**Suggested Questions:**")
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                if st.button(st.session_state.suggested_questions[0], key="btn_q1", use_container_width=True):
                    q_selected = st.session_state.suggested_questions[0]
            with col_q2:
                if len(st.session_state.suggested_questions) > 1 and st.button(st.session_state.suggested_questions[1], key="btn_q2", use_container_width=True):
                    q_selected = st.session_state.suggested_questions[1]
            with col_q3:
                if len(st.session_state.suggested_questions) > 2 and st.button(st.session_state.suggested_questions[2], key="btn_q3", use_container_width=True):
                    q_selected = st.session_state.suggested_questions[2]
        
        # Search input
        user_query = st.text_input("Ask a question about the dataset:", value=q_selected if q_selected else "", placeholder="e.g. Which product category generates the most sales and what is its profit margin?")
        
        if st.button("Submit Inquiry", type="primary"):
            if user_query:
                with st.spinner("Processing natural language inquiry..."):
                    # 1. Get schema details
                    schema_txt = st.session_state.db_manager.get_schema_summary_text()
                    
                    # 2. Call LLM for SQL query
                    sql_query, explanation, chart_spec = st.session_state.llm_manager.generate_sql(schema_txt, user_query)
                    
                    if not sql_query:
                        st.error(f"LLM failed to generate a query. Response:\n{explanation}")
                    else:
                        # 3. Execute SQL Query
                        df_res, err = st.session_state.db_manager.execute_query(sql_query)
                        
                        # 4. Self Correction Loop (1 retry)
                        if err is not None:
                            st.warning(f"Initial query failed. Triggering self-correction loops...\nError: {err}")
                            corrected_sql, correction_log = st.session_state.llm_manager.correct_sql(schema_txt, sql_query, err)
                            
                            df_res, err = st.session_state.db_manager.execute_query(corrected_sql)
                            if err is not None:
                                st.error(f"SQL execution failed even after self-correction.\nOriginal SQL:\n{sql_query}\nCorrected SQL:\n{corrected_sql}\nError: {err}")
                            else:
                                sql_query = corrected_sql # Success on corrected SQL
                                
                        if err is None and df_res is not None:
                            # Save to chat history
                            # Summarize the dataframe results for the LLM
                            summary_df_str = df_res.head(20).to_markdown(index=False)
                            if len(df_res) > 20:
                                summary_df_str += f"\n\n*(Truncated: showing first 20 of {len(df_res)} rows)*"
                                
                            # 5. Generate AI Insights
                            insights = st.session_state.llm_manager.generate_insights(user_query, sql_query, summary_df_str)
                            
                            # 6. Generate Plotly chart
                            fig, plot_status = st.session_state.viz_engine.create_plotly_chart(df_res, chart_spec)
                            
                            # Append history
                            st.session_state.chat_history.append({
                                "question": user_query,
                                "sql": sql_query,
                                "results": df_res,
                                "fig": fig,
                                "insights": insights,
                                "plot_status": plot_status
                            })
                            
        # Render Chat History (latest first)
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="user-bubble">🔍 Question: <strong>{chat["question"]}</strong></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="assistant-bubble">', unsafe_allow_html=True)
            
            # Show SQL inside expander
            with st.expander("🛠️ Executed SQL Code"):
                st.code(chat["sql"], language="sql")
                
            # Show chart if available
            if chat["fig"] is not None:
                st.plotly_chart(chat["fig"], use_container_width=True)
                if "Auto-inferred" in chat["plot_status"]:
                    st.caption(f"ℹ️ {chat['plot_status']}")
            else:
                st.info(f"Visualizations: {chat['plot_status']}")
                
            # Show Table Results
            st.markdown("##### Query Output Sample")
            st.dataframe(chat["results"], use_container_width=True, hide_index=True)
            
            # Show insights
            st.markdown("##### 💡 AI Business Insights")
            st.write(chat["insights"])
            st.markdown('</div>', unsafe_allow_html=True)
            
    # ------------------ TAB 2: Time Series Forecasting ------------------
    with tab_forecast:
        st.markdown("### 🔮 Advanced Time-Series Forecasting")
        st.write("Identify long-term forecasts and seasonal growth trajectories using automated machine learning (Prophet / Holt-Winters Exponential Smoothing).")
        
        # Filter available column options
        cols_info = schema_info.get("columns", [])
        date_cols = [c["name"] for c in cols_info if "date" in c["name"].lower() or "time" in c["name"].lower() or "year" in c["name"].lower() or c["type"] in ["DATE", "DATETIME"]]
        numeric_cols = [c["name"] for c in cols_info if c["type"] in ["REAL", "INTEGER", "NUMERIC", "FLOAT"]]
        
        # Fallbacks if metadata isn't perfectly classified
        all_cols_names = [c["name"] for c in cols_info]
        if not date_cols:
            date_cols = all_cols_names
        if not numeric_cols:
            numeric_cols = all_cols_names
            
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            sel_date_col = st.selectbox("Select Date Column", date_cols)
        with col_f2:
            sel_metric_col = st.selectbox("Select Metric to Forecast", numeric_cols)
        with col_f3:
            horizon_periods = st.slider("Forecast Horizon (Periods)", min_value=3, max_value=90, value=30)
        with col_f4:
            frequency_sel = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=0)
            
        if st.button("Generate Trend Forecast", type="primary"):
            # Load raw dataframe from SQL db to do analysis
            try:
                # Retrieve all rows
                df_raw, _ = st.session_state.db_manager.execute_query(f"SELECT {sel_date_col}, {sel_metric_col} FROM dataset")
                
                if df_raw is not None and not df_raw.empty:
                    with st.spinner("Fitting forecasting model and generating predictions..."):
                        fig, summary, status = st.session_state.analytics_engine.generate_forecast(
                            df_raw, sel_date_col, sel_metric_col, horizon_periods, frequency_sel
                        )
                        
                        if fig is not None:
                            st.success(status)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Render stats in a column format
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("Historical Avg", f"{summary['historical_average']:,.2f}")
                            with col_s2:
                                st.metric("Forecasted Avg", f"{summary['forecasted_average']:,.2f}")
                            with col_s3:
                                st.metric("Projected Trend", f"{summary['percentage_change']:.2f}%", 
                                          delta=f"{summary['percentage_change']:.2f}%")
                                
                            # Ask LLM to explain the forecast
                            with st.spinner("Generating AI explanation..."):
                                summ_str = (
                                    f"Engine: {summary['engine']}\n"
                                    f"Forecast Period: {summary['horizon']} {summary['frequency']} steps\n"
                                    f"Historical Mean: {summary['historical_average']:.2f}\n"
                                    f"Forecasted Mean: {summary['forecasted_average']:.2f}\n"
                                    f"Change percentage: {summary['percentage_change']:.2f}%\n"
                                    f"Confidence Interval: {summary['min_predicted']:.2f} to {summary['max_predicted']:.2f}"
                                )
                                insight_text = st.session_state.llm_manager.explain_trend_or_forecast(sel_metric_col, summ_str)
                                
                                st.markdown("##### 💡 AI Trend Interpretation")
                                st.write(insight_text)
                        else:
                            st.error(status)
                else:
                    st.error("Could not fetch data for forecasting.")
            except Exception as e:
                st.error(f"Error preparing forecasting parameters: {str(e)}")
                
    # ------------------ TAB 3: Anomaly Detection ------------------
    with tab_anomaly:
        st.markdown("### 🚨 Outlier & Anomaly Detection")
        st.write("Scan your dataset for operational risks, fraudulent data entry, or outstanding high/low outliers using the **Isolation Forest** machine learning algorithm.")
        
        numeric_cols_for_anomaly = [c["name"] for c in cols_info if c["type"] in ["REAL", "INTEGER", "NUMERIC", "FLOAT"]]
        if not numeric_cols_for_anomaly:
            numeric_cols_for_anomaly = all_cols_names
            
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            sel_anomaly_cols = st.multiselect("Select columns to analyze for anomalies (multi-column support)", numeric_cols_for_anomaly, default=numeric_cols_for_anomaly[:1])
        with col_a2:
            contamination_slider = st.slider("Expected Outlier Percentage (Contamination)", min_value=0.005, max_value=0.15, value=0.03, step=0.005, format="%.3f")
            
        if st.button("Scan for Anomalies", type="primary"):
            if not sel_anomaly_cols:
                st.warning("Please select at least one column for analysis.")
            else:
                try:
                    # Get complete dataframe from db
                    df_raw, _ = st.session_state.db_manager.execute_query("SELECT * FROM dataset")
                    
                    if df_raw is not None and not df_raw.empty:
                        with st.spinner("Building Isolation Forest forest and flagging records..."):
                            df_flagged, fig, status = st.session_state.analytics_engine.detect_anomalies(
                                df_raw, sel_anomaly_cols, contamination_slider
                            )
                            
                            st.success(status)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                
                            anomalous_rows = df_flagged[df_flagged['is_anomaly']]
                            
                            st.markdown("##### 📋 Flagged Anomaly Records")
                            if not anomalous_rows.empty:
                                # Show sample anomalies in interactive frame
                                st.dataframe(anomalous_rows.drop(columns=['is_anomaly', 'anomaly_score'], errors='ignore'), use_container_width=True, hide_index=True)
                                
                                # Explain anomalies via LLM
                                with st.spinner("Generating AI explanation of anomaly signatures..."):
                                    # Create summary string
                                    sum_str = anomalous_rows.head(8).to_markdown(index=False)
                                    if len(anomalous_rows) > 8:
                                        sum_str += f"\n\n*(Truncated: showing first 8 of {len(anomalous_rows)} anomaly records)*"
                                    anomaly_explanation = st.session_state.llm_manager.explain_anomalies(sum_str)
                                    
                                    st.markdown("##### 💡 AI Risk Analysis")
                                    st.write(anomaly_explanation)
                            else:
                                st.info("No anomalies detected based on current configuration settings.")
                    else:
                        st.error("Empty dataset, cannot perform anomaly detection.")
                except Exception as e:
                    st.error(f"Failed to execute anomaly routine: {str(e)}")
                    
    # ------------------ TAB 4: SQL Sandbox ------------------
    with tab_sandbox:
        st.markdown("### 💻 SQL Sandbox & Active Table Schema")
        st.write("Write standard SQLite queries directly to search the dataset or double check structural mappings.")
        
        # Render schema structures
        with st.expander("📁 View Database Columns and Schema Structures"):
            cols_tbl = []
            for col in schema_info.get("columns", []):
                cols_tbl.append({
                    "Sanitized SQL Column": col["name"],
                    "SQL Type": col["type"],
                    "Original CSV Column": col["original_name"]
                })
            st.table(pd.DataFrame(cols_tbl))
            
        custom_sql = st.text_area("Write SQL Query (e.g. SELECT * FROM dataset LIMIT 10)", value="SELECT * FROM dataset LIMIT 5")
        
        if st.button("Run Sandbox Query", type="primary"):
            if custom_sql:
                with st.spinner("Executing sandboxed query..."):
                    df_sandbox, err = st.session_state.db_manager.execute_query(custom_sql)
                    if err is not None:
                        st.error(f"SQL Error: {err}")
                    else:
                        st.success(f"Query returned {len(df_sandbox)} rows successfully.")
                        st.dataframe(df_sandbox, use_container_width=True, hide_index=True)
                        
                        # Generate basic visual for custom query
                        auto_spec, auto_x, auto_y = st.session_state.viz_engine.infer_chart_spec(df_sandbox)
                        if auto_spec != "none" and len(df_sandbox) > 1:
                            fig_sand, _ = st.session_state.viz_engine.create_plotly_chart(df_sandbox, {"type": auto_spec, "x": auto_x, "y": auto_y})
                            if fig_sand is not None:
                                st.plotly_chart(fig_sand, use_container_width=True)
                                
else:
    st.info("👋 Welcome! Please upload a CSV or Excel file in the field above to start the BI dashboard session.")
