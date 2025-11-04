"""
NBA Predictions Dashboard - Main Entry Point

This page automatically redirects to Today's Predictions.
"""
import streamlit as st

st.set_page_config(
    page_title="NBA Predictions Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-redirect to Today's Predictions
st.switch_page("pages/0_Todays_Predictions.py")
