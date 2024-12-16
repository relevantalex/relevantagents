import streamlit as st
import pandas as pd
from database import DatabaseManager
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List

st.set_page_config(
    page_title="Historical Analysis",
    page_icon="📊",
    layout="wide"
)

def create_timeline_chart(analyses: List[Dict]):
    """Create a timeline visualization of analyses"""
    # Convert analyses to DataFrame for plotting
    df = pd.DataFrame(analyses)
    
    fig = px.timeline(
        df,
        x_start="created_at",
        x_end="created_at",
        y="analysis_type",
        color="status",
        hover_data=["insights"],
        title="Analysis Timeline"
    )
    
    return fig

def display_analysis_details(analysis: Dict):
    """Display detailed view of a single analysis"""
    with st.expander(f"📝 Analysis from {analysis['created_at']}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Key Insights")
            st.write(analysis['insights'])
            
            if 'recommendations' in analysis:
                st.subheader("Recommendations")
                for rec in analysis['recommendations']:
                    st.write(f"• {rec}")
        
        with col2:
            st.metric("Confidence Score", f"{analysis.get('confidence_score', 'N/A')}%")
            st.metric("Data Sources Used", analysis.get('source_count', 'N/A'))

def main():
    st.title("📊 Historical Analysis")
    
    db = DatabaseManager()
    
    # Startup selector
    with st.sidebar:
        st.header("Startup Selection")
        startups = db.get_startups()
        startup_names = [s['name'] for s in startups]
        
        if not startup_names:
            st.warning("No startups found. Please create one in the Document Manager.")
            return
            
        selected_startup_name = st.selectbox(
            "Select Startup",
            options=startup_names
        )
        
        selected_startup = next(s for s in startups if s['name'] == selected_startup_name)
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
        
        analysis_types = st.multiselect(
            "Analysis Types",
            options=["Competitor Analysis", "Niche Discovery", "Market Research"],
            default=["Competitor Analysis", "Niche Discovery", "Market Research"]
        )
    
    # Main content area
    st.header(f"Analysis History for {selected_startup_name}")
    
    # TODO: Fetch actual analyses from database
    # This is placeholder data
    example_analyses = [
        {
            "created_at": datetime.now() - timedelta(days=5),
            "analysis_type": "Competitor Analysis",
            "status": "completed",
            "insights": "Example competitor analysis insights",
            "confidence_score": 85,
            "source_count": 12,
            "recommendations": ["Focus on X market", "Develop Y feature"]
        }
    ]
    
    # Timeline visualization
    st.plotly_chart(create_timeline_chart(example_analyses), use_container_width=True)
    
    # Detailed analysis views
    st.subheader("Detailed Analyses")
    for analysis in example_analyses:
        display_analysis_details(analysis)
    
    # Export options
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            "Export Analysis History",
            data=pd.DataFrame(example_analyses).to_csv(index=False),
            file_name=f"{selected_startup_name}_analysis_history.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
