"""Main Streamlit UI for invoice processing"""
# TODO: Build Streamlit dashboard
import os
import asyncio
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from enum import Enum
import fitz  # PyMuPDF
import re

from graph import get_workflow
from state import InvoiceProcessingState, ProcessingStatus, ValidationStatus, RiskLevel, PaymentStatus
from utils.logger import setup_logging, get_logger

import json
import google.generativeai as genai
from agents.smart_explainer_agent import SmartExplainerAgent
from agents.insights_agent import InsightAgent
from agents.forecast_agent import ForecastAgent


# Logging Setup
setup_logging()
logger = get_logger("InvoiceProcessingApp")

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any DataFrame to be Streamlit/Arrow compatible:
    - Converts Enums to string values
    - Replaces None/NaN with 'Not applicable'
    - Ensures all columns are strings (avoids Arrow conversion errors)
    - Capitalizes column headers
    """
    if df.empty:
        return df

    # Convert Enums to strings
    df = df.applymap(lambda x: x.value if isinstance(x, Enum) else x)

    # Replace None/NaN and make all values string
    df = df.fillna("Not applicable").astype(str)

    # Capitalize column names nicely
    df.columns = [col.capitalize() for col in df.columns]
    return df

import ast
import re
def parse_escalation_details(s):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}

    # Convert datetime.datetime(YYYY,MM,DD,HH,MM,SS) → "YYYY-MM-DD HH:MM:SS"
    def repl(match):
        parts = match.group(1).split(',')
        parts = [p.strip() for p in parts]
        # convert to ISO style
        return f"'{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}'"

    s_clean = re.sub(r"datetime\.datetime\((.*?)\)", repl, s)

    try:
        return ast.literal_eval(s_clean)
    except:
        return {}

def serialize_state(state):
    # Pydantic v2
    if hasattr(state, "model_dump"):
        return state.model_dump()

    # Pydantic v1 fallback
    if hasattr(state, "dict"):
        return state.dict()

    # Normal python object
    if hasattr(state, "__dict__"):
        return state.__dict__

    # Already a dict
    if isinstance(state, dict):
        return state

    # string, int, None, etc
    return {"value": state}


class InvoiceProcessingApp:
    """Main application class for AI Invoice Processing Dashboard"""

    def __init__(self):
        self.workflow = None
        self.initialize_session_state()
        self.initialize_workflow()
        self.smart_explainer = SmartExplainerAgent()
        self.insights = InsightAgent()
        self.forecast = ForecastAgent()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY_7")

    # INITIALIZATION
    def initialize_session_state(self):
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []
        if "results" not in st.session_state:
            st.session_state.results = []
        if "last_run" not in st.session_state:
            st.session_state.last_run = None
        if "workflow_type" not in st.session_state:
            st.session_state.workflow_type = "standard"
        if "max_concurrent" not in st.session_state:
            st.session_state.max_concurrent = 1
        if "annotated_pdfs" not in st.session_state:
            st.session_state.annotated_pdfs = {} 
        # if "priority_level" not in st.session_state:
        #     st.session_state.priority_level = 1

    def initialize_workflow(self):
        try:
            self.workflow = get_workflow()
            logger.info("Workflow initialized successfully.")
        except Exception as e:
            logger.exception("Workflow initialization failed: %s", e)
            st.error("Failed to initialize workflow. Check logs for details.")

    # SIDEBAR + HEADER
    def render_header(self):
        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                color: white;
                margin-bottom: 1rem;
            ">
                <h1 style="margin-bottom: 0;">🧾 Invoice AgenticAI - LangGraph</h1>
                <p style="font-size:1.1rem;">AI-Powered Invoice Processing with Intelligent Agent Workflows</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    def render_sidebar(self):
        st.sidebar.markdown("## ⚙️ Control Panel")
    
        with st.sidebar.expander("🧩 Workflow Configuration", expanded=True):
            st.session_state.workflow_type = st.selectbox(
                "Workflow Type", ["standard", "high_value", "expedited"], index=0
            )
            # st.session_state.priority_level = st.slider("Priority Level", 1, 3, 1)
            st.session_state.max_concurrent = st.slider("Max Concurrent Processing", 1, 10, 1)
    
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📁 Invoice Files")
    
        files = self.get_available_files()
        chosen = st.sidebar.multiselect("Select invoices to process", files)
    
        st.session_state.selected_files = chosen
    
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🚀 Processing Controls")
        # --- Processing Controls with Emojis ---
        if st.sidebar.button("🚀 Process Invoices"):
            if not chosen:
                st.sidebar.error("⚠️ Please select at least one invoice file.")
            else:
                asyncio.run(
                    self.process_invoices_async(
                        chosen,
                        st.session_state.workflow_type,
                        # st.session_state.priority_level,
                        st.session_state.max_concurrent,
                    )
                )
        
        if st.sidebar.button("🧹 Clear Results"):
            st.session_state.results = []
            st.sidebar.success("✅ Results cleared successfully!")



    # FILE HANDLING
    def get_available_files(self) -> List[str]:
        invoices_dir = "data/invoices"
        os.makedirs(invoices_dir, exist_ok=True)
        files = [
            os.path.join(invoices_dir, f)
            for f in os.listdir(invoices_dir)
            if f.lower().endswith(".pdf")
        ]
        return sorted(files)

    # WORKFLOW EXECUTION
    # def _get_stages_for_workflow(self, workflow_type: str) -> list[str]:
    #     """Return dynamic stage flow for given workflow type."""
    #     workflow_type = (workflow_type or "standard").lower()
    #     if workflow_type == "high_value":
    #         return ["Document", "Validation", "Risk", "Audit", "Escalation", "Human Review"]
    #     elif workflow_type == "expedited":
    #         return ["Document", "Validation", "Payment", "Audit"]
    #     else:  # standard
    #         return ["Document", "Validation", "Risk", "Payment", "Audit", "Escalation (if needed)", "Human Review (if needed)"]

    # async def process_invoices_async(self, selected_files, workflow_type, max_concurrent):
    #     if not self.workflow:
    #         st.error("Workflow not initialized.")
    #         return
    
    #     # Workflow stages dynamically chosen
    #     total_files = len(selected_files)
    #     stage_index = 0
    #     total_stages = 0

    #     progress_bar = st.progress(0)
    #     pipeline_placeholder = st.empty()
    #     status_placeholder = st.empty()
    #     # start_time = datetime.utcnow()
    #     duration=0
    #     results = []
    
    #     # Create a placeholder for dynamic banner updates
    #     banner_placeholder = st.empty()
    
    #     # Initial banner: "Processing..."
    #     banner_placeholder.markdown(
    #         f"""
    #         <div style="background: linear-gradient(90deg,#007cf0,#00dfd8);
    #                     padding:1rem;border-radius:10px;color:white;text-align:center;">
    #         🚀 <b>Processing {len(selected_files)} invoice(s)</b> via 
    #         <span style="text-transform:capitalize;">{workflow_type}</span> workflow...
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    #     for i, file in enumerate(selected_files):
    #         st.markdown(f"### 📄 `{os.path.basename(file)}` ({i+1}/{total_files})")

    #         try:
    #             # Process one file and get the agent flow
    #             with st.spinner("🤖 Processing Invoice(s) with AI agents..."):
    #                 start_time = datetime.utcnow()
    #                 state, worked_agents = await self.workflow.process_invoice(file, workflow_type=workflow_type)
    #                 duration += (datetime.utcnow() - start_time).total_seconds()
    #                 total_stages += len(worked_agents)
    #             results.append(state)
    #             # Loop through dynamic stages
    #             for j, stage in enumerate(worked_agents):
    #                 with pipeline_placeholder:
    #                     self.show_agent_pipeline(stage, workflow_type, worked_agents)
        
    #                 status_placeholder.markdown(
    #                     f"<div style='background-color:#f9f9f9;padding:1rem;border-radius:10px;'>🧠 Running <b>{stage}</b>...</div>",
    #                     unsafe_allow_html=True
    #                 )
        
    #                 # fake progress animation
    #                 for pct in range(0, 101, 25):
    #                     await asyncio.sleep(0.15)
                        
    #                 stage_index += 1
    #                 progress_bar.progress(int((stage_index/total_stages)*100))
    #                 # Step 2️: Update to "Completed" state
    #                 status_placeholder.markdown(
    #                     f"""
    #                     <div style='background-color:#e8f5e9;padding:1rem;border-radius:10px;
    #                                 border-left:6px solid #4caf50;'>
    #                     ✅ <b>All the above Agents</b> have been processed successfully!
    #                     </div>
    #                     """,
    #                     unsafe_allow_html=True
    #                 )
                
    #             await asyncio.sleep(0.3) # Small delay to let user see the completion
                
    #         except Exception as e:
    #             logger.exception(f"Error processing {file}: {e}")
    #             st.error(f"❌ Failed: {os.path.basename(file)}")
    #             continue
    #     progress_bar.progress(100)
    #     banner_placeholder.empty()
    #     # duration = (datetime.utcnow() - start_time).total_seconds()
    #     st.balloons()
    #     banner_placeholder.markdown(
    #         f"""
    #         <div style="background: linear-gradient(90deg,#00dfd8,#007cf0);
    #                     padding:1rem;border-radius:10px;color:white;text-align:center;">
    #         ✅ <b>Processed {len(results)} invoice(s)</b> successfully via 
    #         <span style="text-transform:capitalize;">{workflow_type}</span> workflow in {duration:.2f}s!
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    
    #     # Final summary
    #     st.success(f"🎉 All {len(results)} invoices processed in {duration:.2f} seconds")
    
    #     st.markdown(
    #         f"<div style='text-align:center;font-size:1.1rem;margin-top:1rem;'>"
    #         f"✅ <b>{workflow_type.capitalize()} Workflow Completed</b>"
    #         f"</div>", unsafe_allow_html=True
    #     )
    
    #     st.session_state.results = [r.model_dump() if hasattr(r, 'model_dump') else r for r in results]
    #     st.session_state.last_workflow_type = workflow_type
    async def process_invoices_async(self, selected_files, workflow_type, max_concurrent):
        if not self.workflow:
            st.error("Workflow not initialized.")
            return

        total_files = len(selected_files)
        progress_bar = st.progress(0)
        pipeline_placeholder = st.empty()
        status_placeholder = st.empty()
        banner_placeholder = st.empty()
        duration = 0
        results = []

        # Initial banner
        banner_placeholder.markdown(
            f"""
            <div style="background: linear-gradient(90deg,#007cf0,#00dfd8);
                        padding:1rem;border-radius:10px;color:white;text-align:center;">
            🚀 <b>Processing {len(selected_files)} invoice(s)</b> via 
            <span style="text-transform:capitalize;">{workflow_type}</span> workflow...
            </div>
            """,
            unsafe_allow_html=True,
        )

        # # ----------------------------------------------------------------------
        # # 🚀 PARALLEL / BATCH PROCESSING MODE (max_concurrent > 1)
        # # ----------------------------------------------------------------------
        # if max_concurrent > 1:
        #     st.info(f"⚡ Running in parallel mode with max_concurrent = {max_concurrent}")

        #     with st.spinner("🤖 Processing invoices in parallel..."):
        #         start_time = datetime.utcnow()

        #         # Run all invoices concurrently via process_batch()
        #         batch_states = await self.workflow.process_batch(
        #             selected_files,
        #             workflow_type=workflow_type,
        #             max_concurrent=max_concurrent
        #         )

        #         duration = (datetime.utcnow() - start_time).total_seconds()

        #     # Update progress bar as each file completes
        #     for idx, state in enumerate(batch_states):
        #         progress_bar.progress(int(((idx + 1) / total_files) * 100))
        #         await asyncio.sleep(0.1)

        #     results = batch_states

        #     st.success(f"🎉 Processed {len(results)} invoices in {duration:.2f} seconds (parallel mode)")

        #     banner_placeholder.markdown(
        #         f"""
        #         <div style="background: linear-gradient(90deg,#00dfd8,#007cf0);
        #                     padding:1rem;border-radius:10px;color:white;text-align:center;">
        #         ✅ <b>Processed {len(results)} invoice(s)</b> successfully via 
        #         <span style="text-transform:capitalize;">{workflow_type}</span> workflow in {duration:.2f}s!
        #         </div>
        #         """,
        #         unsafe_allow_html=True,
        #     )

        #     st.balloons()

        #     # Store in session
        #     clean_results = []
        #     for r in results:
        #         state = r["state"]
        #         worked = r["worked_agents"]
        #         clean_results.append({
        #             "state": state.model_dump(),
        #             "worked_agents": worked
        #         })
        #     st.session_state.results = clean_results
        #     st.session_state.last_workflow_type = workflow_type
        #     return
        # ----------------------------------------------------------------------
        # 🚀 PARALLEL / BATCH PROCESSING MODE (max_concurrent > 1)
        # ----------------------------------------------------------------------
        if max_concurrent > 1:

            st.info(f"⚡ Running in parallel mode with max_concurrent = {max_concurrent}")

            total_files = len(selected_files)

            # ----------------------- UI: QUEUED FILE LIST -----------------------
            st.markdown("### 📋 Invoice Queue")
            file_rows = []

            for f in selected_files:
                row = st.empty()
                row.markdown(
                    f"""
                    <div style="padding:0.8rem;border-radius:8px;background:#f2f2f2;">
                        📄 <b>{os.path.basename(f)}</b><br>
                        ⏳ <i>Queued...</i>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                file_rows.append(row)

            # Progress bar for file-level parallel completion
            progress_bar = st.progress(0)

            # ----------------------- RUN PARALLEL PROCESSING -----------------------
            with st.spinner("🤖 Processing invoices in parallel..."):
                start_time = datetime.utcnow()

                batch_results = await self.workflow.process_batch(
                    selected_files,
                    workflow_type=workflow_type,
                    max_concurrent=max_concurrent
                )
                duration = (datetime.utcnow() - start_time).total_seconds()

            # ----------------------- UPDATE UI AS FILES COMPLETE -----------------------
            st.markdown("### ✅ Processing Results")

            clean_results = []
            completed_count = 0
            print("batch_results from main", batch_results)
            for idx, item in enumerate(batch_results):
                state = item["state"]              # FIXED
                worked_agents = item["worked_agents"]  # FIXED

                print("state from main ---", state)
                print("worked_agents from main ---", worked_agents)

                file_rows[idx].markdown(
                    f"""
                    <div style="padding:0.8rem;border-radius:8px;
                                background:#e8f5e9;border-left:6px solid #4caf50;">
                        📄 <b>{os.path.basename(selected_files[idx])}</b><br>
                        ✅ Completed — {len(worked_agents)} agents executed
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                completed_count += 1
                progress_bar.progress(int((completed_count / total_files) * 100))
                await asyncio.sleep(0.15)

                with st.expander(f"🔍 Workflow Details — {os.path.basename(selected_files[idx])}"):
                    st.markdown("### 🧠 Agent Workflow Replay")
                    for agent in worked_agents:
                        self.show_agent_pipeline(agent, workflow_type, worked_agents)
                        await asyncio.sleep(0.25)

                # clean_results.append({
                #     "state": serialize_state(state),
                #     "worked_agents": worked_agents
                # })
                clean_results.append(state)
                print("cleaned res from main---", clean_results)

            # ----------------------- FINAL BANNER -----------------------
            st.balloons()
            st.success(
                f"🎉 Processed {completed_count} invoices in {duration:.2f} seconds (parallel mode)"
            )

            banner_placeholder.markdown(
                f"""
                <div style="background: linear-gradient(90deg,#00dfd8,#007cf0);
                            padding:1rem;border-radius:10px;color:white;text-align:center;">
                ✅ <b>Processed {completed_count} invoice(s)</b> successfully via 
                <span style="text-transform:capitalize;">{workflow_type}</span>
                workflow in {duration:.2f}s!
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----------------------- SAVE SESSION -----------------------
            st.session_state.results = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in clean_results
            ]
            # st.session_state.results = clean_results
            st.session_state.last_workflow_type = workflow_type
            return


        # SEQUENTIAL MODE (max_concurrent == 1)
        stage_index = 0
        total_stages = 0

        for i, file in enumerate(selected_files):
            st.markdown(f"### 📄 `{os.path.basename(file)}` ({i+1}/{total_files})")

            try:
                with st.spinner("🤖 Processing Invoice(s) with AI agents..."):
                    start_time = datetime.utcnow()

                    # sequential per-file detailed processing
                    state, worked_agents = await self.workflow.process_invoice(
                        file,
                        workflow_type=workflow_type
                    )

                    duration += (datetime.utcnow() - start_time).total_seconds()
                    total_stages += len(worked_agents)

                results.append(state)

                # UI pipeline per agent
                for j, stage in enumerate(worked_agents):

                    with pipeline_placeholder:
                        self.show_agent_pipeline(stage, workflow_type, worked_agents)

                    status_placeholder.markdown(
                        f"<div style='background-color:#f9f9f9;padding:1rem;border-radius:10px;'>"
                        f"🧠 Running <b>{stage}</b>..."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Fake animation to show progress nicely
                    for pct in range(0, 101, 25):
                        await asyncio.sleep(0.15)

                    stage_index += 1
                    progress_bar.progress(int((stage_index / total_stages) * 100))

                    # Mark completed
                    status_placeholder.markdown(
                        f"""
                        <div style='background-color:#e8f5e9;padding:1rem;border-radius:10px;
                                    border-left:6px solid #4caf50;'>
                        ✅ <b>All the above Agents</b> have been processed successfully!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                await asyncio.sleep(0.3)

            except Exception as e:
                logger.exception(f"Error processing {file}: {e}")
                st.error(f"❌ Failed: {os.path.basename(file)}")
                continue

        # Final UI completion
        progress_bar.progress(100)
        banner_placeholder.empty()
        st.balloons()

        banner_placeholder.markdown(
            f"""
            <div style="background: linear-gradient(90deg,#00dfd8,#007cf0);
                        padding:1rem;border-radius:10px;color:white;text-align:center;">
            ✅ <b>Processed {len(results)} invoice(s)</b> successfully via 
            <span style="text-transform:capitalize;">{workflow_type}</span> workflow in {duration:.2f}s!
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"🎉 All {len(results)} invoices processed in {duration:.2f} seconds")

        st.markdown(
            f"<div style='text-align:center;font-size:1.1rem;margin-top:1rem;'>"
            f"✅ <b>{workflow_type.capitalize()} Workflow Completed</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Store results
        st.session_state.results = [
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in results
        ]
        st.session_state.last_workflow_type = workflow_type


    def show_agent_pipeline(self, current_stage: str, workflow_type: str, stages: list[str]):
        """Display dynamic workflow pipeline based on selected workflow."""
        colors = {
            "document_agent": "#00bcd4",
            "validation_agent": "#fbc02d",
            "risk_agent": "#e64a19",
            "payment_agent": "#43a047",
            "audit_agent": "#1976d2",
            "escalation_agent": "#ff9800",
            # "Escalation (if needed)": "#f44336",
            "human_review_agent": "#8e24aa",
            # "human_review (if needed)": "#ba68c8",
        }
    
        html = "<div style='display:flex;justify-content:space-between;align-items:center;margin:1rem 0;'>"
        for stage in stages:
            glow = "box-shadow:0 0 12px rgba(0,255,0,0.7);" if stage == current_stage else ""
            html += f"""<div style="flex:1;text-align:center;padding:0.8rem;
                            border-radius:10px;background:{colors.get(stage,'#666')};
                            color:white;margin:0 4px;{glow}">
                    <b>{stage}</b>
                </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


    # SUMMARY OVERVIEW
    def show_processing_summary(self, results: List):
        if not results:
            st.info("No results yet.")
            return
        df_rows = []
        escalations = sum(1 for r in results if r.get("escalation_required"))
        completed = sum(1 for r in results if r.get("overall_status") == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.get("overall_status") == ProcessingStatus.FAILED)

        for r in results:
            df_rows.append(
                {
                    "File": os.path.basename(r.get("file_name", "")),
                    "Status": r.get("overall_status"),
                    "Risk Score": (r.get("risk_assessment") or {}).get("risk_score"),
                    "Amount": (r.get("invoice_data") or {}).get("total"),
                    "Escalation": r.get("escalation_required"),
                }
            )

        df = pd.DataFrame(df_rows)
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Invoices Processed", len(results))
        col2.metric("⚠️ Escalations", escalations)
        col3.metric("❌ Failures", failed)
        st.dataframe(df, width='stretch')

    # DASHBOARD TABS
    def render_main_dashboard(self):
        tabs = st.tabs(["Overview", "Invoice Details", "Agent Performance", "Escalations", "Analytics", "Smart Insights", "Health"])
        with tabs[0]:
            self.render_overview_tab()
        with tabs[1]:
            self.render_invoice_details_tab()
        with tabs[2]:
            self.render_agent_performance_tab()
        with tabs[3]:
            self.render_escalations_tab()
        with tabs[4]:
            self.render_analytics_tab()
        with tabs[5]:
            self.render_smart_insights_tab()
        with tabs[6]:
            self.show_health_check()

    def render_overview_tab(self):
        st.subheader("Processing Overview")
        st.markdown("""
        <div style="
            background-color: #f0f2f6;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
        📋 <b>Workflow Summary:</b> The AI system routes invoices automatically between extraction, validation, risk, and payment agents.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🤖 AI Agent Workflow")
        st.markdown("""
        1. 🧾 **Document Agent** – Extract invoice data using AI  
        2. ✅ **Validation Agent** – Validate against purchase orders  
        3. ⚠️ **Risk Agent** – Assess fraud risk and compliance  
        4. 💳 **Payment Agent** – Make payment decisions  
        5. 🧮 **Audit Agent** – Generate compliance records  
        6. 🚨 **Escalation Agent** – Handle exceptions  
        
        > 🧠 The workflow uses intelligent routing based on validations, risk scores, and business rules.
        """)

        self.show_processing_summary(st.session_state.results)

    # INVOICE DETAILS
    def render_invoice_details_tab(self):
        st.markdown("""
        <div style="
            background-color: #f0f2f6;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
        📋 <b>Workflow Summary:</b> The AI system routes invoices automatically between extraction, validation, risk, and payment agents.
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Detailed Invoice View")
        results = st.session_state.results
        if not results:
            st.info("No processed invoices yet.")
            return

        selected_file = st.selectbox("Select invoice for details:", [os.path.basename(r["file_name"]) for r in results])
        selected = next((r for r in results if os.path.basename(r["file_name"]) == selected_file), None)
        print("selected....", selected)
        if selected:
            self.show_detailed_invoice_view(selected)
            # --- Add Button for Highlighting Discrepancies ---
            st.markdown("### 🔍 Check PDF for Issues")
            if st.button("Highlight Discrepancies in Invoice PDF"):
                invoice_path = next(
                    (f for f in self.get_available_files() 
                    if os.path.basename(f) == os.path.basename(selected["file_name"])), None
                )
                if invoice_path:
                    with st.spinner("Analyzing invoice for discrepancies..."):
                        discrepancies, output_pdf = self.highlight_invoice_discrepancies(invoice_path)
                    st.session_state.annotated_pdfs[selected_file] = output_pdf
                    if discrepancies:
                        st.error("⚠️ Mismatches found:")
                        df_disc = pd.DataFrame(discrepancies)
                        st.dataframe(df_disc)
                        with open(output_pdf, "rb") as f:
                            st.download_button(
                                label=f"📥 Download Annotated Invoice PDF ({selected_file})",
                                data=f,
                                file_name=os.path.basename(output_pdf),
                                mime="application/pdf"
                            )
                    else:
                        st.success("✅ No discrepancies found. Invoice matches CSV perfectly!")

            # --- Show Dropdown of All Annotated PDFs ---
            if st.session_state.annotated_pdfs:
                st.markdown("### 📂 Annotated Invoices Library")
            
                selected_marked_pdf = st.selectbox(
                    "Select an annotated (highlighted) invoice:",
                    options=list(st.session_state.annotated_pdfs.keys()),
                    key="annotated_pdf_selector"
                )
            
                pdf_path = st.session_state.annotated_pdfs[selected_marked_pdf]
            
                st.info(f"Selected: {selected_marked_pdf}")
            
                # Download button
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {selected_marked_pdf}",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )

    def show_detailed_invoice_view(self, result):
        st.markdown(f"### 🧾 Invoice Summary")
        st.markdown(f"**File:** `{os.path.basename(result.get('file_name', 'N/A'))}`")
        st.markdown(f"**Status:** :{'green_circle:' if result.get('overall_status') == 'completed' else 'orange_circle:'} {result.get('overall_status', 'Unknown')}")
    
        # --- Invoice Data Section ---
        st.markdown("### 📋 Invoice Data")
        invoice_data = result.get("invoice_data", {})
    
        # Handle datetime formatting
        for k, v in invoice_data.items():
            if isinstance(v, datetime):
                invoice_data[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(v, str) and v.startswith("datetime.datetime"):
                invoice_data[k] = v.replace("datetime.datetime", "").strip("()")
    
        df_invoice = make_arrow_safe(pd.DataFrame(invoice_data.items(), columns=["Field", "Value"]))
        st.dataframe(df_invoice, width='stretch')
    
        # --- Line Items ---
        items = invoice_data.get("item_details") or invoice_data.get("line_items")
        if items:
            st.markdown("#### 📦 Line Items")
            df_items = make_arrow_safe(pd.DataFrame(items))
            st.dataframe(df_items, width='stretch')
    
        # --- Validation Result ---
        st.markdown("### ✅ Validation Result")
        validation = result.get("validation_result", {})
        if validation:
            po_data = validation.pop("po_data", {})
            df_val = make_arrow_safe(pd.DataFrame(validation.items(), columns=["Check", "Result"]))
            st.dataframe(df_val, width='stretch')
    
            if po_data:
                st.markdown("#### 📦 PO Data")
                df_po = make_arrow_safe(pd.DataFrame(po_data.items(), columns=["Field", "Value"]))
                st.dataframe(df_po, width='stretch')
    
        # --- Risk Assessment ---
        st.markdown("### ⚠️ Risk Assessment")
        risk = result.get("risk_assessment", {})
        if risk:
            df_risk = make_arrow_safe(pd.DataFrame(risk.items(), columns=["Factor", "Value"]))
            st.dataframe(df_risk, width='stretch')
    
        # --- Payment Decision ---
        st.markdown("### 💳 Payment Decision")
        payment = result.get("payment_decision", {})
        if payment:
            for k, v in payment.items():
                if isinstance(v, datetime):
                    payment[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(v, str) and v.startswith("datetime.datetime"):
                    payment[k] = v.replace("datetime.datetime", "").strip("()")
    
            df_payment = make_arrow_safe(pd.DataFrame(payment.items(), columns=["Attribute", "Value"]))
            st.dataframe(df_payment, width='stretch')
        else:
            st.error("There is something mismatch in PO_data and Invoice Details. Please look into the previous data.")
    
        # --- Audit Trail ---
        st.markdown("### 🧮 Audit Trail")
        audit = result.get("audit_trail", []) or []
        if audit:
            df_audit = make_arrow_safe(pd.DataFrame(audit).head(10))
            st.dataframe(df_audit, width='stretch')
        else:
            st.info("No audit trail entries found.")

    def highlight_invoice_discrepancies(self, invoice_path: str):
        """Compare invoice PDF with CSV and highlight mismatches visually."""
        DATA_DIR = os.path.join(os.getcwd(), "data")
        CSV_PATH = os.path.join(DATA_DIR, "purchase_orders.csv")
        OUTPUT_PATH = os.path.join(DATA_DIR, "annotated_invoice.pdf")
    
        FIELD_BOXES = {
            "invoice_number": (525, 55, 575, 75),
            "order_id": (45, 470, 230, 490),
            "customer_name": (40, 135, 100, 155),
            "quantity": (370, 235, 385, 250),
            "rate": (450, 235, 500, 250),
            "expected_amount": (520, 315, 570, 330),
        }
    
        pdf = fitz.open(invoice_path)
        page = pdf[0]
        pdf_text = page.get_text()
    
        # === extract fields ===
        def extract_field(pattern, text, group=1):
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(group).strip() if match else None
    
        invoice_number_pdf = extract_field(r"#\s*(\d+)", pdf_text)
        order_id_pdf = extract_field(r"Order ID\s*[:\-]?\s*(\S+)", pdf_text)
        customer_name_pdf = extract_field(r"Bill To:\s*(.*)", pdf_text)
    
        po_df = pd.read_csv(CSV_PATH)
        matched_row = po_df[
            (po_df['invoice_number'].astype(str) == str(invoice_number_pdf))
            | (po_df['order_id'] == order_id_pdf)
        ]
        if matched_row.empty:
            st.warning(f"No matching CSV row found for Invoice {invoice_number_pdf} / Order {order_id_pdf}")
            return None, None
    
        expected = matched_row.iloc[0].to_dict()
        expected = {k.lower(): str(v).strip() for k, v in expected.items()}
    
        invoice_data = {
            "invoice_number": invoice_number_pdf,
            "customer_name": customer_name_pdf,
            "order_id": order_id_pdf,
        }
    
        amounts = re.findall(r"\$?([\d,]+\.\d{2})", pdf_text)
        invoice_data["expected_amount"] = amounts[-3] if amounts else None
    
        item_lines = re.findall(
            r"([A-Za-z0-9 ,\-]+)\s+(\d+)\s+\$?([\d,]+\.\d{2})\s+\$?([\d,]+\.\d{2})",
            pdf_text,
        )
        if item_lines:
            invoice_data["quantity"] = item_lines[0][1]
            invoice_data["rate"] = item_lines[0][2]
    
        discrepancies = []
    
        def add_discrepancy(field, expected_val, found_val):
            discrepancies.append({"field": field, "expected": expected_val, "found": found_val})
    
        for field in ["invoice_number", "order_id", "customer_name"]:
            if str(invoice_data.get(field, "")).strip() != str(expected.get(field, "")).strip():
                add_discrepancy(field, expected.get(field, ""), invoice_data.get(field, ""))
    
        for field in ["quantity", "rate", "expected_amount"]:
            try:
                found_val = float(str(invoice_data.get(field, 0)).replace(",", "").replace("$", ""))
                expected_val = float(str(expected.get(field, 0)).replace(",", "").replace("$", ""))
                if round(found_val, 2) != round(expected_val, 2):
                    add_discrepancy(field, expected_val, found_val)
            except:
                if str(invoice_data.get(field, "")) != str(expected.get(field, "")):
                    add_discrepancy(field, expected.get(field, ""), invoice_data.get(field, ""))
    
        for d in discrepancies:
            field = d["field"]
            if field not in FIELD_BOXES:
                continue
            rect = fitz.Rect(FIELD_BOXES[field])
            expected_text = f"{d['expected']}"
            page.draw_rect(rect, color=(1, 0, 0), width=1.5)
            page.insert_text((rect.x0, rect.y1 + 10), expected_text, fontsize=9, color=(1, 0, 0))
    
        pdf.save(OUTPUT_PATH)
        pdf.close()
    
        return discrepancies, OUTPUT_PATH

    # AGENT PERFORMANCE TAB
    def render_agent_performance_tab(self):
        st.subheader("Agent Performance Metrics")
        try:
            metrics = asyncio.run(self.workflow.health_check())["agent"]
        except Exception as e:
            print("error in metrics from main.py............", e)
            st.warning("No live metrics found.")
            return

        print("metrics from main", metrics)
        rows = []
        for agent, health_check_history in metrics.items():
            rows.append({
                "Agent": health_check_history.get("Agent"),
                "Executions": health_check_history.get("Executions", 0),
                "Success Rate (%)": round(health_check_history.get("Success Rate (%)", 0), 2),
                "Avg Duration (ms)": health_check_history.get("Avg Duration (ms)", 0),
                "Total Failures": health_check_history.get("Total Failures", 0),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch')

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Agent"], y=df["Success Rate (%)"], name="Success Rate"))
        fig.add_trace(go.Bar(x=df["Agent"], y=df["Avg Duration (ms)"], name="Duration (ms)", yaxis="y2"))
        fig.update_layout(
            title="Agent Success vs Duration",
            yaxis_title="Success Rate (%)",
            yaxis2=dict(title="Duration (ms)", overlaying="y", side="right"),
            barmode="group",
        )
        st.plotly_chart(fig, width='stretch')

    # ESCALATIONS TAB
    def render_escalations_tab(self):
        st.subheader("Escalations Overview")
        results = st.session_state.results
        # rows = [
        #     {
        #         "File": os.path.basename(r.get("file_name", "")),
        #         "Reason": r.get("escalation_reason", "N/A"),
        #         "Assigned To": (r.get("escalation_details") or {}).get("assigned_to", "N/A"),
        #         "SLA Deadline": (r.get("escalation_details") or {}).get("sla_deadline", "N/A"),
        #     }
        #     for r in results if r.get("escalation_required")
        # ]
        # if not rows:
        #     st.info("No escalations detected.")
        #     return
        # st.dataframe(pd.DataFrame(rows), width='stretch')
        rows = []
        for r in results:
            if not r.get("escalation_required"):
                continue

            esc_dict = parse_escalation_details(r.get("escalation_details"))

            row = {
                "File": os.path.basename(r.get("file_name", "")),
                "Reason": r.get("escalation_reason", "N/A"),
                "Assigned To": esc_dict.get("assigned_to", "N/A"),
                "SLA Deadline": esc_dict.get("sla_deadline", "N/A"),
            }

            rows.append(row)
        if not rows:
            st.info("No escalations detected")
        else:
            st.dataframe(pd.DataFrame(rows))

    # ANALYTICS TAB
    def render_analytics_tab(self):
        st.subheader("Processing Analytics")
        results = st.session_state.results
        if not results:
            st.info("No analytics available.")
            return

        df = pd.DataFrame([
            {
                "File": os.path.basename(r["file_name"]),
                "Amount": (r.get("invoice_data") or {}).get("total", 0),
                "Risk Score": (r.get("risk_assessment") or {}).get("risk_score", 0),
                "Status": r.get("overall_status", "unknown"),
            }
            for r in results
        ])

        if df.empty:
            st.info("No numeric analytics.")
            return

        col1, col2 = st.columns(2)
        fig1 = px.bar(df, x="File", y="Amount", color="Status", title="Total Amount by Invoice")
        fig2 = px.scatter(df, x="Amount", y="Risk Score", color="Status", title="Risk vs Amount")
        col1.plotly_chart(fig1, width='stretch')
        col2.plotly_chart(fig2, width='stretch')


    def render_smart_insights_tab(self):
        st.subheader("Smart Insights Assistant")
    
        results = st.session_state.get("results", [])
        if not results:
            st.info("No processed invoices available for insights.")
            return
    
        # ------------------ SMART EXPLAINER ------------------
        st.markdown("### Smart Explainer")
    
        try:
            genai.configure(api_key=self.gemini_api_key)
        except Exception as e:
            st.warning(f"⚠️ Gemini API not configured properly: {e}")
    
        selected_file = st.selectbox(
            "Select an invoice to explain:",
            [os.path.basename(r.get("file_name", "Unknown")) for r in results],
        )
    
        selected = next(
            (r for r in results if os.path.basename(r.get("file_name", "")) == selected_file),
            None,
        )
    
        if selected:
            # ✅ Sanitize before parsing
            if "human_review_required" in selected and not isinstance(selected["human_review_required"], bool):
                selected["human_review_required"] = False
            if "escalation_details" in selected and not isinstance(selected["escalation_details"], (str, type(None))):
                selected["escalation_details"] = str(selected["escalation_details"])
    
            try:
                parsed_state = InvoiceProcessingState(**selected)
            except Exception:
                parsed_state = None
    
            # ------------------ Header Summary (Enhanced UI) ------------------
            st.markdown("### Invoice Summary")
    
            risk = ((selected.get("risk_assessment") or {}).get("risk_level", "Unknown")).capitalize()
            validation = ((selected.get("validation_result") or {}).get("validation_status", "Unknown")).capitalize()
            payment = (
                (selected.get("payment_decision") or {}).get("status", "Pending")
                if isinstance(selected.get("payment_decision"), dict)
                else "Pending"
            )
    
            # 🔹 Professional color palette
            risk_colors = {
                "Critical": "#F44336",  # Bright red
                "Medium": "#FF9800",    # Deep amber
                "Low": "#4CAF50",       # Balanced green
                "Unknown": "#9E9E9E",   # Neutral gray
            }
            validation_colors = {
                "Passed": "#4CAF50",
                "Failed": "#F44336",
                "Missing_po": "#FFC107",
                "Unknown": "#9E9E9E",
            }
            payment_colors = {
                "Paid": "#4CAF50",
                "Pending": "#FFC107",
                "Overdue": "#E91E63",
                "Unknown": "#9E9E9E",
            }
    
            def badge_html(label, value, color):
                return f"""
                <div style="
                    background-color:{color};
                    color:white;
                    padding:10px 16px;
                    border-radius:12px;
                    text-align:center;
                    font-weight:600;
                    box-shadow:0 2px 6px rgba(0,0,0,0.25);
                    font-size:0.95rem;
                ">
                    {label}: {value}
                </div>
                """
    
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(badge_html("Risk", risk, risk_colors.get(risk, "#9E9E9E")), unsafe_allow_html=True)
            with col2:
                st.markdown(badge_html("Validation", validation, validation_colors.get(validation, "#9E9E9E")), unsafe_allow_html=True)
            with col3:
                st.markdown(badge_html("Payment", payment, payment_colors.get(payment, "#9E9E9E")), unsafe_allow_html=True)
    
            # ------------------ Gemini Explanation ------------------
        
            if parsed_state:
                try:
                    # Safe fallback handling for missing invoice_data fields
                    inv = getattr(parsed_state, "invoice_data", None)
                    vendor = getattr(inv, "customer_name", None) or getattr(inv, "vendor_name", "Unknown Vendor")
                    amount = getattr(inv, "total", "Unknown Amount")
                    due_date = getattr(inv, "due_date", "Unknown Due Date")
            
                    prompt = (
                        f"Provide a clear, professional summary of the following invoice:\n\n"
                        f"Vendor: {vendor}\n"
                        f"Amount: {amount}\n"
                        f"Due Date: {due_date}\n"
                        f"Risk Level: {risk}\n"
                        f"Validation: {validation}\n"
                        f"Payment Status: {payment}\n\n"
                        f"Generate a concise, professional explanation with sections:\n"
                        f"1. Overview\n2. Risk and Validation Summary\n3. Recommended Actions"
                    )
            
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
            
                except Exception as e:
                    st.warning(f"Gemini explanation failed: {e}")
                    st.json(selected)
            else:
                st.warning("Could not parse invoice or Gemini unavailable.")
                st.json(selected)
        
        st.markdown("---")

        
        # ------------------ CONSOLIDATED SPEND INSIGHTS ------------------
        st.markdown("### Consolidated Spend Insights")
    
        try:
            df = pd.DataFrame([
                {
                    "File": os.path.basename(r.get("file_name", "Unknown")),
                    "Vendor": (r.get("invoice_data") or {}).get("customer_name", "Unknown"),
                    "Amount": float((r.get("invoice_data") or {}).get("total", 0)),
                    "Risk": ((r.get("risk_assessment") or {}).get("risk_level", "Unknown")).capitalize(),
                    "Validation": ((r.get("validation_result") or {}).get("validation_status", "Unknown")).capitalize(),
                }
                for r in results
            ])
    
            if df.empty:
                st.info("No invoice data available for visualization.")
            else:
                total_spend = df["Amount"].sum()
                st.metric("Total Spend (USD)", f"${total_spend:,.2f}")
                st.metric("Invoices Processed", len(df))
    
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.pie(
                        df,
                        names="Risk",
                        values="Amount",
                        title="Spend by Risk Level",
                        color="Risk",
                        color_discrete_map={
                            "Critical": "#F44336",
                            "Medium": "#FF9800",
                            "Low": "#4CAF50",
                            "Unknown": "#9E9E9E",
                        },
                    )
                    st.plotly_chart(fig1, use_container_width=True)
    
                with col2:
                    fig2 = px.bar(
                        df,
                        x="Vendor",
                        y="Amount",
                        color="Validation",
                        title="Spend by Vendor",
                        color_discrete_map={
                            "Passed": "#4CAF50",
                            "Failed": "#F44336",
                            "Missing_po": "#FFC107",
                            "Unknown": "#9E9E9E",
                        },
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating spend insights: {e}")
    
        st.markdown("---")
    
        # ------------------ FORECAST & ANOMALY SECTION ------------------
        st.markdown("### 📊 Forecast & Anomaly Insights")
    
        try:
            clean_states = []
            for r in results:
                if isinstance(r, dict):
                    try:
                        clean_states.append(InvoiceProcessingState(**r))
                    except Exception:
                        r_fixed = dict(r)
                        if isinstance(r_fixed.get("human_review_required"), str):
                            val = r_fixed["human_review_required"].strip().lower()
                            r_fixed["human_review_required"] = val in ("true", "yes", "1", "required")
                        if isinstance(r_fixed.get("escalation_details"), dict):
                            r_fixed["escalation_details"] = str(r_fixed["escalation_details"])
                        try:
                            clean_states.append(InvoiceProcessingState(**r_fixed))
                        except Exception:
                            continue
    
            forecast_data = self.forecast.predict_cashflow(clean_states, months=6)
            if forecast_data.get("chart"):
                st.plotly_chart(forecast_data["chart"], use_container_width=True)
                st.metric("Average Monthly Spend (USD)", f"${forecast_data['average_monthly_spend']:,.2f}")
                st.metric("Total Forecast (next 6 months)", f"${forecast_data['total_forecast']:,.2f}")

    
            if not forecast_data or not forecast_data.get("chart"):
                st.info("No sufficient data for forecast.")
            else:
                st.plotly_chart(forecast_data["chart"], use_container_width=True)
                st.success(
                    f"**Average Monthly Spend:** ${forecast_data['average_monthly_spend']:,}  \n"
                    f"**Total Forecast (Next {len(forecast_data['forecast_values'])} months):** ${forecast_data['total_forecast']:,}"
                )
    
            anomalies = self.forecast.detect_anomalies(clean_states)
            if anomalies is not None and not anomalies.empty:
                st.markdown("### ⚠️ Anomalies Detected")
                st.dataframe(
                    anomalies[["invoice_date", "vendor", "total", "risk_score", "anomaly_reason"]],
                    use_container_width=True,
                )
                st.warning(f"{len(anomalies)} anomalies found (high spend or high risk).")
            else:
                st.info("No anomalies detected. Spend trends look stable ✅")
    
        except Exception as e:
            st.error(f"Forecast section failed: {e}")

    def show_workflow_diagram(self):
        # pass
        st.subheader("Workflow Diagram (Conceptual)")
        st.image(os.path.join("assets", "workflow_diagram.png")) if os.path.exists(os.path.join("assets", "workflow_diagram.png")) else st.text("Diagram not provided.")


    # HEALTH CHECK TAB
    def show_health_check(self):
        st.subheader("System Health Check")
        try:
            health = asyncio.run(self.workflow.health_check())
            # st.json(health)
            agents_data = health.get("agent", {})
            if not agents_data:
                st.warning("No Agents Data Found")
                return
            df_health = pd.DataFrame(agents_data).T
            df_health = make_arrow_safe(df_health)
            orchestrator_status = health.get("orchestrator", "unknown")
            st.markdown(f"**Orchestrator Status:** `{orchestrator_status}`")
            st.dataframe(df_health, width='stretch')
        except Exception as e:
            st.error(f"Health check failed: {e}")

    # RUN APP
    def run(self):
        self.render_header()
        if self.workflow:
            st.success("✅ All agents and workflow initialized successfully!")
        else:
            st.error("⚠️ Workflow not initialized. Please check logs.")

        # if "last_pipeline_stage" in st.session_state and "last_workflow_type" in st.session_state:
        #     stages = self._get_stages_for_workflow(st.session_state["last_workflow_type"])
        #     self.show_agent_pipeline(st.session_state["last_pipeline_stage"], st.session_state["last_workflow_type"], stages)

        st.info("📁 Select invoice files from the sidebar and click **Process Invoices** to get started.")

        self.render_sidebar()
        self.render_main_dashboard()
        
        # if st.session_state.last_run:
        #     st.sidebar.markdown("---")
        #     st.sidebar.info(f"Last Run: {st.session_state.last_run}")


if __name__ == "__main__":
    app = InvoiceProcessingApp()
    app.run()