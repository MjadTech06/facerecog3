# 📚 Step-by-Step Guide: Building a Touchless Satisfaction Survey App in your github

# -*- coding: utf-8 -*-
# ============================================================================
# TOUCHLESS SATISFACTION SURVEY - ENHANCED EDUCATIONAL VERSION
# Features:
# - Admin panel with password (no default message shown to respondents)
# - Multiple data cleaning strategies with educational notes
# - Multiple statistical methods with explanations
# - Multiple ML models with learning descriptions
# - Built-in SQLite database
# - Teachable Machine integration
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import json
import io
import base64

# Analysis libraries
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.impute import KNNImputer, SimpleImputer
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Survey System",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default admin password (NOT shown to users)
DEFAULT_ADMIN_PASSWORD = "admin123"

# Database file
DB_FILE = "survey_responses.db"

# Survey questions
SURVEY_QUESTIONS = [
    "How satisfied are you with the workshop content?",
    "How satisfied are you with the instructor's teaching?",
    "How satisfied are you with the workshop materials?",
    "How satisfied are you with the hands-on activities?",
    "How satisfied are you with the overall workshop experience?"
]

# Gesture mapping
GESTURE_MAP = {
    'thumbs_up': {'label': 'Satisfied', 'score': 4, 'emoji': '👍'},
    'heart_sign': {'label': 'Very Satisfied', 'score': 5, 'emoji': '❤️'},
    'thumbs_down': {'label': 'Unsatisfied', 'score': 2, 'emoji': '👎'},
    'waving_finger': {'label': 'Very Unsatisfied', 'score': 1, 'emoji': '☝️'},
    'closed_fist': {'label': 'No Answer', 'score': None, 'emoji': '✊'}
}
