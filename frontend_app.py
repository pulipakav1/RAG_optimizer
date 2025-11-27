import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Pipeline Optimizer", layout="wide")
st.title("🧠 RAG Pipeline Optimizer")

st.markdown(
    "Upload your documents and benchmark multiple RAG configurations on accuracy, relevance, and cost-efficiency."
)

# --- Upload section ---
st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF or text files", type=["pdf", "txt"], accept_multiple_files=True
)

if st.button("Upload & Index") and uploaded_files:
    files = []
    for f in uploaded_files:
        files.append(
            ("files", (f.name, f.read(), f"type" if hasattr(f, "type") else "application/octet-stream"))
        )

    with st.spinner("Uploading and indexing documents..."):
        resp = requests.post(f"{BACKEND_URL}/upload", files=files)

    if resp.status_code == 200:
        st.success("Documents indexed successfully into all pipelines!")
    else:
        st.error(f"Upload failed: {resp.text}")


# --- Question / Evaluation section ---
st.header("2. Ask a Question & Compare Pipelines")

question = st.text_input("Enter your question:")

if st.button("Run Evaluation") and question:
    with st.spinner("Running all RAG pipelines and evaluating..."):
        resp = requests.post(f"{BACKEND_URL}/ask", json={"question": question})

    if resp.status_code != 200:
        st.error(f"Error from backend: {resp.text}")
    else:
        data = resp.json()
        evaluation = data.get("evaluation", {})
        pipelines = data.get("pipelines", [])
        
        # Extract scores for visualization
        if evaluation and "winner" in evaluation:
            winner = evaluation["winner"]
            st.success(f"🏆 **Winner: Pipeline {winner}**")
            
            # Prepare data for charts
            pipeline_ids = []
            accuracies = []
            relevances = []
            cost_efficiencies = []
            descriptions = []
            
            for pipeline_id in ["A", "B", "C", "D"]:
                if pipeline_id in evaluation:
                    scores = evaluation[pipeline_id]
                    pipeline_ids.append(f"Pipeline {pipeline_id}")
                    accuracies.append(scores.get("accuracy", 0))
                    relevances.append(scores.get("relevance", 0))
                    cost_efficiencies.append(scores.get("cost_efficiency", 0))
                    # Get description from pipeline data
                    desc = next((p["description"] for p in pipelines if p["pipeline_id"] == pipeline_id), "")
                    descriptions.append(desc)
            
            # Create DataFrame for easier manipulation
            df_scores = pd.DataFrame({
                "Pipeline": pipeline_ids,
                "Accuracy": accuracies,
                "Relevance": relevances,
                "Cost Efficiency": cost_efficiencies,
                "Description": descriptions,
                "Is Winner": [p.replace("Pipeline ", "") == winner for p in pipeline_ids]
            })
            
            # --- Bar Charts Section ---
            st.header("📊 Performance Charts")
            
            # Grouped bar chart for all metrics
            fig_bar = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            winner_color = '#FFD700'  # Gold for winner
            
            for i, pipeline_id in enumerate(pipeline_ids):
                is_winner = df_scores.iloc[i]["Is Winner"]
                color = winner_color if is_winner else colors[i % len(colors)]
                
                fig_bar.add_trace(go.Bar(
                    name=f"{pipeline_id}",
                    x=["Accuracy", "Relevance", "Cost Efficiency"],
                    y=[accuracies[i], relevances[i], cost_efficiencies[i]],
                    marker_color=color,
                    text=[accuracies[i], relevances[i], cost_efficiencies[i]],
                    textposition='auto',
                    hovertemplate=f"<b>{pipeline_id}</b><br>" +
                                 "Metric: %{x}<br>" +
                                 "Score: %{y}<br>" +
                                 "<extra></extra>"
                ))
            
            fig_bar.update_layout(
                title="Pipeline Performance Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score (1-10)",
                barmode='group',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Individual metric bar charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_acc = px.bar(
                    df_scores,
                    x="Pipeline",
                    y="Accuracy",
                    color="Is Winner",
                    color_discrete_map={True: '#FFD700', False: '#1f77b4'},
                    title="Accuracy Scores",
                    text="Accuracy"
                )
                fig_acc.update_traces(texttemplate='%{text}', textposition='outside')
                fig_acc.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                fig_rel = px.bar(
                    df_scores,
                    x="Pipeline",
                    y="Relevance",
                    color="Is Winner",
                    color_discrete_map={True: '#FFD700', False: '#ff7f0e'},
                    title="Relevance Scores",
                    text="Relevance"
                )
                fig_rel.update_traces(texttemplate='%{text}', textposition='outside')
                fig_rel.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_rel, use_container_width=True)
            
            with col3:
                fig_cost = px.bar(
                    df_scores,
                    x="Pipeline",
                    y="Cost Efficiency",
                    color="Is Winner",
                    color_discrete_map={True: '#FFD700', False: '#2ca02c'},
                    title="Cost Efficiency Scores",
                    text="Cost Efficiency"
                )
                fig_cost.update_traces(texttemplate='%{text}', textposition='outside')
                fig_cost.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_cost, use_container_width=True)
            
            # --- Comparison Table Section ---
            st.header("📋 Comparison Table")
            
            # Create styled DataFrame
            df_display = df_scores[["Pipeline", "Description", "Accuracy", "Relevance", "Cost Efficiency"]].copy()
            df_display["Total Score"] = df_display["Accuracy"] + df_display["Relevance"] + df_display["Cost Efficiency"]
            df_display = df_display.sort_values("Total Score", ascending=False)
            
            # Highlight winner row
            def highlight_winner(row):
                is_winner = row["Pipeline"].replace("Pipeline ", "") == winner
                return ['background-color: #FFD700' if is_winner else '' for _ in row]
            
            st.dataframe(
                df_display.style.apply(highlight_winner, axis=1).format({
                    "Accuracy": "{:.1f}",
                    "Relevance": "{:.1f}",
                    "Cost Efficiency": "{:.1f}",
                    "Total Score": "{:.1f}"
                }),
                use_container_width=True,
                height=300
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Accuracy", f"{df_display['Accuracy'].max():.1f}", 
                         f"Pipeline {df_display.loc[df_display['Accuracy'].idxmax(), 'Pipeline'].replace('Pipeline ', '')}")
            
            with col2:
                st.metric("Best Relevance", f"{df_display['Relevance'].max():.1f}",
                         f"Pipeline {df_display.loc[df_display['Relevance'].idxmax(), 'Pipeline'].replace('Pipeline ', '')}")
            
            with col3:
                st.metric("Best Cost Efficiency", f"{df_display['Cost Efficiency'].max():.1f}",
                         f"Pipeline {df_display.loc[df_display['Cost Efficiency'].idxmax(), 'Pipeline'].replace('Pipeline ', '')}")
            
            with col4:
                st.metric("Overall Winner", f"Pipeline {winner}", 
                         f"Total: {df_display[df_display['Pipeline'] == f'Pipeline {winner}']['Total Score'].values[0]:.1f}")
            
            # --- Pipeline Answers Section ---
            st.header("💬 Pipeline Answers")
            for p in pipelines:
                is_winner = p['pipeline_id'] == winner
                border_color = "#FFD700" if is_winner else "#e0e0e0"
                border_width = "3px" if is_winner else "1px"
                
                st.markdown(
                    f"""
                    <div style="border: {border_width} solid {border_color}; border-radius: 10px; padding: 15px; margin: 10px 0; 
                               {'background-color: #fff9e6;' if is_winner else 'background-color: #f9f9f9;'}">
                        <h3 style="margin-top: 0; {'color: #FFD700;' if is_winner else ''}">
                            {'🏆 ' if is_winner else ''}Pipeline {p['pipeline_id']} — {p['description']}
                        </h3>
                        <p><strong>Answer:</strong></p>
                        <p>{p['answer']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                with st.expander(f"Show retrieved context for Pipeline {p['pipeline_id']}"):
                    st.write(p.get("context", "No context available"))
        else:
            st.warning("Evaluation data format is not as expected. Showing raw data:")
            st.json(evaluation)
            
            st.subheader("Pipeline Answers")
            for p in pipelines:
                st.markdown(f"### Pipeline {p['pipeline_id']} — {p['description']}")
                st.markdown("**Answer:**")
                st.write(p["answer"])
                with st.expander("Show retrieved context"):
                    st.write(p.get("context", "No context available"))
