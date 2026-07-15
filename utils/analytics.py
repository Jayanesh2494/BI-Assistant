import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Lazy import for prophet since it might have warning messages during initialization
def get_prophet_forecast(df: pd.DataFrame, date_col: str, metric_col: str, horizon: int, freq: str) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    """
    Fits a Prophet model and predicts future values.
    Returns (forecast_df, success, error_message).
    """
    try:
        from prophet import Prophet
        
        # Prepare dataframe for Prophet
        df_prophet = df[[date_col, metric_col]].copy()
        df_prophet.columns = ['ds', 'y']
        
        # Drop rows with null dates or metrics
        df_prophet = df_prophet.dropna(subset=['ds', 'y'])
        
        # Convert ds to datetime and remove timezone info if any
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        
        # Aggregate by date to ensure unique DS values (Prophet requirement)
        df_prophet = df_prophet.groupby('ds', as_index=False).mean()
        
        if len(df_prophet) < 5:
            return pd.DataFrame(), False, "Insufficient data points (at least 5 required) for Prophet."
            
        m = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            interval_width=0.80 # 80% confidence interval
        )
        m.fit(df_prophet)
        
        # Determine frequency parameter for Prophet
        # Map human-readable freq to Prophet codes
        freq_map = {
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "ME",
            "Quarterly": "QE",
            "Yearly": "YE"
        }
        p_freq = freq_map.get(freq, "D")
        
        future = m.make_future_dataframe(periods=horizon, freq=p_freq)
        forecast = m.predict(future)
        
        return forecast, True, None
    except Exception as e:
        return pd.DataFrame(), False, str(e)

def get_statsmodels_forecast(df: pd.DataFrame, date_col: str, metric_col: str, horizon: int, freq: str) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    """
    Fallback forecasting engine using Holt-Winters Exponential Smoothing.
    """
    try:
        df_ts = df[[date_col, metric_col]].copy()
        df_ts.columns = ['ds', 'y']
        df_ts['ds'] = pd.to_datetime(df_ts['ds'])
        df_ts = df_ts.dropna(subset=['ds', 'y'])
        df_ts = df_ts.groupby('ds').mean().sort_index()
        
        # Resample to match frequency
        freq_map = {
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "ME",
            "Quarterly": "QE",
            "Yearly": "YE"
        }
        r_freq = freq_map.get(freq, "D")
        df_resampled = df_ts.resample(r_freq).mean().interpolate(method='linear')
        
        if len(df_resampled) < 5:
            return pd.DataFrame(), False, "Insufficient data points for Holt-Winters."
            
        # Fit Holt-Winters
        model = ExponentialSmoothing(
            df_resampled['y'],
            trend='add',
            seasonal='add' if len(df_resampled) > 24 else None,
            seasonal_periods=12 if r_freq in ['M', 'ME'] else 7
        )
        fit = model.fit()
        
        # Forecast
        forecast_values = fit.forecast(horizon)
        
        # Construct output matching Prophet format for compatibility
        historical_len = len(df_resampled)
        all_dates = list(df_resampled.index)
        
        # Create future dates
        last_date = df_resampled.index[-1]
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=r_freq)[1:]
        all_dates.extend(future_dates)
        
        # Combine yhat
        yhat_hist = fit.fittedvalues
        yhat_fut = forecast_values
        yhat = np.concatenate([yhat_hist.values, yhat_fut.values])
        
        # Simple confidence interval calculation (1.28 * residual standard error for 80% CI)
        residuals = df_resampled['y'] - yhat_hist
        std_resid = residuals.std()
        ci_half = 1.28 * std_resid
        
        forecast_df = pd.DataFrame({
            'ds': all_dates,
            'yhat': yhat,
            'yhat_lower': yhat - ci_half,
            'yhat_upper': yhat + ci_half
        })
        
        # Add original y for historical comparison
        y_vals = list(df_resampled['y'].values)
        y_vals.extend([None] * horizon)
        forecast_df['actual'] = y_vals
        
        return forecast_df, True, None
    except Exception as e:
        return pd.DataFrame(), False, str(e)


class AnalyticsEngine:
    """
    Executes advanced statistical computations: Forecasting & Anomaly Detection.
    """
    
    def generate_forecast(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        metric_col: str, 
        horizon: int = 30, 
        freq: str = "Daily"
    ) -> Tuple[Optional[go.Figure], Dict[str, Any], str]:
        """
        Runs forecasting (Prophet or Statsmodels fallback) and returns a Plotly figure,
        a text summary for the LLM, and status info.
        """
        # Attempt Prophet first
        forecast_df, success, error_msg = get_prophet_forecast(df, date_col, metric_col, horizon, freq)
        engine_used = "Prophet"
        
        if not success:
            # Fallback to Statsmodels
            forecast_df, success, error_msg = get_statsmodels_forecast(df, date_col, metric_col, horizon, freq)
            engine_used = "Holt-Winters (Fallback)"
            if not success:
                return None, {}, f"Forecasting failed: {error_msg}"
                
        # Generate plot
        fig = go.Figure()
        
        # Determine split index between historical and forecast
        # For Prophet, we merge the original actual values back
        if 'actual' not in forecast_df.columns:
            # Join actuals back
            df_actuals = df[[date_col, metric_col]].copy()
            df_actuals.columns = ['ds', 'actual']
            df_actuals['ds'] = pd.to_datetime(df_actuals['ds']).dt.tz_localize(None)
            df_actuals = df_actuals.groupby('ds', as_index=False).mean()
            forecast_df = pd.merge(forecast_df, df_actuals, on='ds', how='left')
            
        historical = forecast_df[forecast_df['actual'].notna()]
        future = forecast_df[forecast_df['actual'].isna()]
        
        # Historical actuals
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['actual'],
            name="Historical Actuals",
            mode="markers+lines",
            marker=dict(size=4, color="#3B82F6"),
            line=dict(color="#3B82F6", width=1.5)
        ))
        
        # Predicted line
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            name="Forecasted Trend",
            mode="lines",
            line=dict(color="#10B981", width=2, dash="dash" if len(future) > 0 else "solid")
        ))
        
        # Confidence interval shading (80% CI)
        fig.add_trace(go.Scatter(
            x=list(forecast_df['ds']) + list(forecast_df['ds'])[::-1],
            y=list(forecast_df['yhat_upper']) + list(forecast_df['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="80% Confidence Interval"
        ))
        
        # Layout
        fig.update_layout(
            title=f"Forecast of {metric_col.replace('_', ' ').title()} using {engine_used}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", family="Inter, sans-serif"),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", tickfont=dict(color="#94A3B8")),
            legend=dict(
                bgcolor="rgba(30, 41, 59, 0.7)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                borderwidth=1
            )
        )
        
        # Compile statistics summary
        hist_avg = historical['actual'].mean()
        fut_avg = future['yhat'].mean() if not future.empty else forecast_df['yhat'].mean()
        change_pct = ((fut_avg - hist_avg) / hist_avg * 100) if hist_avg != 0 else 0
        
        summary = {
            "engine": engine_used,
            "horizon": horizon,
            "frequency": freq,
            "historical_average": float(hist_avg),
            "forecasted_average": float(fut_avg),
            "percentage_change": float(change_pct),
            "max_predicted": float(forecast_df['yhat_upper'].max()),
            "min_predicted": float(forecast_df['yhat_lower'].min())
        }
        
        status_msg = f"Forecasting completed successfully using {engine_used}."
        
        return fig, summary, status_msg

    def detect_anomalies(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        contamination: float = 0.03
    ) -> Tuple[pd.DataFrame, Optional[go.Figure], str]:
        """
        Runs IsolationForest anomaly detection on the selected numeric columns.
        Returns (dataframe_with_flags, plotly_figure, status_message).
        """
        if not feature_cols:
            return df, None, "No columns selected for anomaly detection."
            
        df_res = df.copy()
        
        # Extract features and handle missing values
        features = df_res[feature_cols].copy()
        # Fill missing values with median of each column
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(features[col].median())
            
        # If there are no rows left, error out
        if features.empty:
            return df, None, "Selected columns contain no valid numeric data."
            
        try:
            # Fit Isolation Forest
            # contamination is the proportion of outliers in the data set
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(features)
            scores = iso.decision_function(features)
            
            # Save results
            # Preds are 1 for inliers, -1 for outliers
            df_res['is_anomaly'] = [True if p == -1 else False for p in preds]
            df_res['anomaly_score'] = scores
            
            anomaly_count = int((df_res['is_anomaly'] == True).sum())
            
            # Generate plotly scatter plot
            # Let's find a date or index column to plot against
            x_axis = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    x_axis = col
                    break
                    
            if not x_axis:
                # Use index if no date column
                df_res['record_index'] = df_res.index
                x_axis = 'record_index'
                
            primary_metric = feature_cols[0]
            
            fig = go.Figure()
            
            # Split data
            normals = df_res[~df_res['is_anomaly']]
            anomalies = df_res[df_res['is_anomaly']]
            
            # Plot normals
            fig.add_trace(go.Scatter(
                x=normals[x_axis],
                y=normals[primary_metric],
                mode="markers",
                name="Normal Records",
                marker=dict(color="#3B82F6", size=6, opacity=0.7)
            ))
            
            # Plot anomalies
            fig.add_trace(go.Scatter(
                x=anomalies[x_axis],
                y=anomalies[primary_metric],
                mode="markers",
                name="Anomalies (Flagged)",
                marker=dict(color="#EF4444", size=10, symbol="x", line=dict(width=1.5, color="#FFFFFF"))
            ))
            
            fig.update_layout(
                title=f"Anomaly Detection (Highlighting {primary_metric.replace('_', ' ').title()})",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                margin=dict(l=40, r=40, t=50, b=40),
                xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", tickfont=dict(color="#94A3B8"), title=x_axis.replace('_', ' ').title()),
                yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", tickfont=dict(color="#94A3B8"), title=primary_metric.replace('_', ' ').title()),
                legend=dict(
                    bgcolor="rgba(30, 41, 59, 0.7)",
                    bordercolor="rgba(255, 255, 255, 0.1)",
                    borderwidth=1
                )
            )
            
            status = f"Detected {anomaly_count} anomalous records out of {len(df_res)} ({contamination*100:.1f}% contamination setting)."
            return df_res, fig, status
            
        except Exception as e:
            return df, None, f"Failed to detect anomalies: {str(e)}"
