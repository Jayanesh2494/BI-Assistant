import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple

def clean_and_sort_df_for_plotting(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """
    Sorts and cleans the dataframe to ensure lines/bars are plotted in order.
    Especially useful for date sorting.
    """
    df_plot = df.copy()
    # Try converting x_col to datetime if it's a date-like object/string
    if df_plot[x_col].dtype == 'object':
        try:
            parsed_dates = pd.to_datetime(df_plot[x_col], errors='coerce')
            # If at least 70% of rows parse as valid dates, apply it
            if parsed_dates.notna().sum() / len(df_plot) > 0.7:
                df_plot[x_col] = parsed_dates
                df_plot = df_plot.sort_values(by=x_col)
        except Exception:
            pass
    elif pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
        df_plot = df_plot.sort_values(by=x_col)
        
    return df_plot

class VisualizationEngine:
    """
    Intelligently generates Plotly visualizations from query results.
    Refers to LLM-suggested specs and falls back to rule-based heuristics.
    """
    
    @staticmethod
    def infer_chart_spec(df: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Infers the best chart type, x-axis, and y-axis columns based on data types.
        Returns (chart_type, x_col, y_col).
        """
        cols = list(df.columns)
        if len(cols) < 2:
            return "none", "", ""
            
        # Identify column types
        date_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for col in cols:
            # Check for date characteristics
            col_lower = col.lower()
            if "date" in col_lower or "time" in col_lower or "year" in col_lower or "month" in col_lower:
                date_cols.append(col)
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                
        # Heuristics
        # 1. Date series -> Line Chart
        if date_cols and numeric_cols:
            return "line", date_cols[0], numeric_cols[0]
            
        # 2. Categorical vs Numeric -> Bar Chart
        if categorical_cols and numeric_cols:
            # If categories are small, pie chart is also nice, but bar is default
            return "bar", categorical_cols[0], numeric_cols[0]
            
        # 3. Numeric vs Numeric -> Scatter Plot
        if len(numeric_cols) >= 2:
            return "scatter", numeric_cols[0], numeric_cols[1]
            
        # 4. Fallback to first two columns
        return "bar", cols[0], cols[1]

    def create_plotly_chart(self, df: pd.DataFrame, llm_spec: Optional[Dict[str, Any]] = None) -> Tuple[Optional[go.Figure], str]:
        """
        Generates a Plotly figure based on the dataframe and LLM specifications.
        Returns a tuple of (Plotly Figure, status_message).
        """
        if df is None or df.empty:
            return None, "No data available to plot."
            
        if len(df.columns) < 2:
            return None, f"Query results have only {len(df.columns)} column. Visualizations require at least 2 columns."
            
        # Resolve x and y columns
        chart_type = "none"
        x_col = ""
        y_col = ""
        
        if llm_spec and llm_spec.get("type", "none") != "none":
            chart_type = llm_spec.get("type", "none")
            x_col = llm_spec.get("x", "")
            y_col = llm_spec.get("y", "")
            
        # Validate columns suggested by LLM
        cols = list(df.columns)
        if x_col not in cols or y_col not in cols:
            # Inference fallback
            chart_type, x_col, y_col = self.infer_chart_spec(df)
            status = f"LLM suggestions invalid or columns not found. Auto-inferred chart: {chart_type} (x='{x_col}', y='{y_col}')"
        else:
            status = f"Plotting LLM-suggested chart: {chart_type} (x='{x_col}', y='{y_col}')"
            
        if chart_type == "none" or not x_col or not y_col:
            return None, "Could not determine appropriate columns for visualization."
            
        try:
            df_plot = clean_and_sort_df_for_plotting(df, x_col)
            
            # Styling colors (Dark glass theme compatible)
            color_sequence = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
            theme_layout = dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                margin=dict(l=40, r=40, t=50, b=40),
                xaxis=dict(
                    gridcolor="rgba(148, 163, 184, 0.1)",
                    linecolor="rgba(148, 163, 184, 0.2)",
                    zerolinecolor="rgba(148, 163, 184, 0.1)",
                    tickfont=dict(color="#94A3B8")
                ),
                yaxis=dict(
                    gridcolor="rgba(148, 163, 184, 0.1)",
                    linecolor="rgba(148, 163, 184, 0.2)",
                    zerolinecolor="rgba(148, 163, 184, 0.1)",
                    tickfont=dict(color="#94A3B8")
                ),
                legend=dict(
                    bgcolor="rgba(30, 41, 59, 0.7)",
                    bordercolor="rgba(255, 255, 255, 0.1)",
                    borderwidth=1,
                    font=dict(color="#E2E8F0")
                )
            )
            
            title_text = f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
            
            if chart_type == "line":
                fig = px.line(
                    df_plot, 
                    x=x_col, 
                    y=y_col, 
                    title=title_text,
                    color_discrete_sequence=color_sequence,
                    markers=len(df_plot) < 50
                )
            elif chart_type == "bar":
                # If categories are many (e.g. > 15), draw horizontal bar chart
                unique_categories = df_plot[x_col].nunique()
                if unique_categories > 15:
                    fig = px.bar(
                        df_plot, 
                        y=x_col, 
                        x=y_col, 
                        orientation='h', 
                        title=title_text,
                        color_discrete_sequence=color_sequence
                    )
                else:
                    fig = px.bar(
                        df_plot, 
                        x=x_col, 
                        y=y_col, 
                        title=title_text,
                        color_discrete_sequence=color_sequence
                    )
            elif chart_type == "pie":
                # Ensure values are non-negative for pie charts
                if (df_plot[y_col] < 0).any():
                    # Fallback to bar if there are negative values
                    fig = px.bar(df_plot, x=x_col, y=y_col, title=title_text, color_discrete_sequence=color_sequence)
                    status += " (Note: Pie chart requested, but fell back to Bar chart due to negative values)."
                else:
                    fig = px.pie(
                        df_plot, 
                        names=x_col, 
                        values=y_col, 
                        title=title_text,
                        color_discrete_sequence=color_sequence,
                        hole=0.4
                    )
            elif chart_type == "scatter":
                fig = px.scatter(
                    df_plot, 
                    x=x_col, 
                    y=y_col, 
                    title=title_text,
                    color_discrete_sequence=color_sequence,
                    trendline="ols" if len(df_plot) > 5 and pd.api.types.is_numeric_dtype(df_plot[x_col]) else None
                )
            else:
                # Default fallback is Bar
                fig = px.bar(df_plot, x=x_col, y=y_col, title=title_text, color_discrete_sequence=color_sequence)
                
            fig.update_layout(**theme_layout)
            return fig, status
            
        except Exception as e:
            return None, f"Failed to generate Plotly chart: {str(e)}"
