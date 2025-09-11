import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

def display_data_preview(df: pd.DataFrame, num_rows: int = 10):
    """
    Display a preview of the DataFrame.
    
    Args:
        df: DataFrame to display
        num_rows: Number of rows to display
    """
    st.subheader("Data Preview")
    st.dataframe(df.head(num_rows))

def display_statistics(df: pd.DataFrame):
    """
    Display basic statistics of the DataFrame.
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

def display_missing_values(df: pd.DataFrame):
    """
    Display missing values information.
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("Missing Values")
    missing_data = pd.DataFrame({
        'Column': df.isnull().sum().index,
        'Missing Values': df.isnull().sum().values,
        'Percentage': (df.isnull().sum() / len(df) * 100).values
    })
    st.dataframe(missing_data)

def plot_missing_values_heatmap(df: pd.DataFrame):
    """
    Plot a heatmap of missing values.
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("Missing Values Heatmap")
    fig = px.imshow(
        df.isnull(),
        labels=dict(x="Column", y="Row", color="Missing"),
        title="Missing Values Heatmap"
    )
    st.plotly_chart(fig)

def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str, mode: str = 'lines'):
    """
    Plot a time series.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
        mode: Plot mode ('lines', 'markers', 'lines+markers')
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode=mode, name=y_col))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig)

def plot_histogram(df: pd.DataFrame, col: str, title: str, nbins: int = 50):
    """
    Plot a histogram.
    
    Args:
        df: DataFrame containing the data
        col: Column name to plot
        title: Plot title
        nbins: Number of bins
    """
    fig = px.histogram(df, x=col, nbins=nbins, title=title)
    st.plotly_chart(fig)

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Plot a bar chart.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig)

def plot_anomalies(df: pd.DataFrame, x_col: str, y_col: str, anomaly_col: str, title: str):
    """
    Plot a time series with anomalies highlighted.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        anomaly_col: Column name for anomaly flag
        title: Plot title
    """
    fig = go.Figure()
    
    # Plot normal points
    normal_df = df[df[anomaly_col] == 0]
    fig.add_trace(go.Scatter(
        x=normal_df[x_col],
        y=normal_df[y_col],
        mode='lines',
        name='Normal',
        line=dict(color='blue')
    ))
    
    # Plot anomalies
    anomaly_df = df[df[anomaly_col] == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_df[x_col],
        y=anomaly_df[y_col],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig)