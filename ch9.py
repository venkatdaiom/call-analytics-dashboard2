import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# --- Custom Color Palette ---
CUSTOM_COLORS = {
    'Biscay': '#183F5E',    # Darkest Blue
    'Edgewater': '#BADED7', # Lightest Blue-Green
    'Neptune': '#73B4B6',   # Mid Blue-Green
    'Mariner': '#267EBB'    # Medium Blue
}

# Mapping for Sentiment labels
SENTIMENT_COLOR_MAP = {
    'Positive': CUSTOM_COLORS['Edgewater'], # Lightest for positive
    'Neutral': CUSTOM_COLORS['Neptune'],    # Mid-tone for neutral
    'Negative': CUSTOM_COLORS['Biscay'],    # Darkest for negative
    'Unknown': '#CCCCCC' # A neutral grey for unknown/unspecified
}

# Custom sequential colorscale for heatmaps (using the provided palette)
HEATMAP_COLORSCALE = [
    [0.0, CUSTOM_COLORS['Edgewater']],   # Lightest blue-green
    [0.33, CUSTOM_COLORS['Neptune']],    # Mid blue-green
    [0.66, CUSTOM_COLORS['Mariner']],    # Medium blue
    [1.0, CUSTOM_COLORS['Biscay']]       # Darkest blue
]

# --- Page Configuration ---
st.set_page_config(
    page_title="Call Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown(f"""
<style>
/* Main Block Container */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}}

/* Metric Card Styling */
div[data-testid="metric-container"] {{
    background-color: #f0f2f6;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
}}

div[data-testid="stMetricLabel"] p {{
    font-size: 16px; /* Label font size, as requested */
    font-weight: bold;
    color: #555555; /* Darker label color */
}}

div[data-testid="stMetricValue"] {{
    font-size: 36px; /* Value font size */
    font-weight: bold;
    color: {CUSTOM_COLORS['Mariner']}; /* Use a color from the palette */
}}

/* Header Styling */
h1 {{
    color: {CUSTOM_COLORS['Biscay']}; /* Use a dark blue for main title */
    text-align: center;
    font-size: 36px; /* Increased for better visibility */
}}
h2 {{
    color: {CUSTOM_COLORS['Mariner']}; /* Use a prominent blue for subheaders */
    border-bottom: 2px solid {CUSTOM_COLORS['Neptune']}; /* Accent line */
    padding-bottom: 5px;
    margin-top: 20px;
    font-size: 28px; /* Increased for better visibility */
}}
h5 {{
    color: {CUSTOM_COLORS['Biscay']}; /* Darker tone for sub-sub headings */
    margin-top: 15px;
    margin-bottom: 5px;
    font-size: 20px; /* Increased for better visibility, larger than 16px */
}}

/* General paragraph text size */
p {{
    font-size: 16px; /* All paragraph text should be at least 16px */
}}

/* Adjust font size for markdown lists and other general text that might not be a direct <p> */
ul, ol, li, div.stMarkdown, div.stInfo, div.stWarning {{
    font-size: 16px !important; /* Force font size for info/warning boxes and list items */
}}

/* Specific adjustment for sidebar elements if needed, often p is enough */
.stMultiSelect p {{
    font-size: 16px !important;
}}
.stFileUploader p {{
    font-size: 16px !important;
}}

/* NEW: Business Question Strip Styling */
.question-strip {{
    background-color: #E6EEF6; /* Light blue-gray background */
    padding: 15px 20px;
    margin-top: 30px; /* Space above the strip */
    margin-bottom: 20px; /* Space below the strip before the chart */
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.question-strip h3 {{
    color: #333333; /* Darker text for the question */
    margin: 0;
    font-size: 20px; /* Slightly larger for the question */
    text-align: center;
}}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_data(file):
    """Loads and preprocesses data from an uploaded CSV file."""
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip() # Strip whitespace from column names
        
        nan_fill_defaults = {
            'UserIntentToBuy': 'Not Specified',
            'CallSentiment': 'Unknown',
            'CallObjective': 'Unknown',
            'NextAction': 'Unknown',
            'Customer_Language': 'Unknown',
            'CallType': 'Unknown'
        }
        
        for col, default_val in nan_fill_defaults.items():
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', default_val, regex=False)
                df[col] = df[col].fillna(default_val)
                df[col] = df[col].str.strip()
            else:
                df[col] = default_val
                
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}. Please ensure it's a valid CSV.")
        return pd.DataFrame()

def find_col(df_columns, possible_names):
    """Finds a column name, trying exact match, then case-insensitive."""
    for name in possible_names:
        if name in df_columns:
            return name
    for name in possible_names:
        for c in df_columns:
            if c.lower() == name.lower():
                return c
    return None

def ensure_datetime(df, col_name_candidates):
    """Tries to find and convert a column to datetime, returning series and column name."""
    for c in col_name_candidates:
        if c and c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors='coerce')
                if dt.notna().sum() > 0:
                    return dt, c
            except Exception:
                pass
    return None, None

def top_n_with_pct(series, n=5):
    """Calculates top N values and their percentages from a Series."""
    vc = series.value_counts(dropna=False) 
    total = vc.sum()
    
    if vc.empty or total == 0:
        return pd.DataFrame(), 0

    top = vc.head(n).reset_index()
    top.columns = ['label', 'count']
    top['pct'] = (top['count'] / total * 100).round(1)
    top['text_label'] = top.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)

    return top, total

def ordered_intent_counts(df_filtered, col, order=None):
    """Orders user intent counts by a predefined order and prepares data for plotting."""
    if order is None:
        order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Not Specified']
    
    if df_filtered.empty or col not in df_filtered.columns:
        return pd.DataFrame()

    counts = df_filtered[col].value_counts(dropna=False)
    counts_reindexed = counts.reindex(order, fill_value=0).astype(int)
    
    relevant_intents = [
        intent for intent in order 
        if counts_reindexed.get(intent, 0) > 0 or (intent == 'Not Specified' and intent in counts_reindexed.index)
    ]
    
    final_ordered_series = counts_reindexed.loc[relevant_intents].reindex(order).dropna()

    if final_ordered_series.empty or final_ordered_series.sum() == 0:
        return pd.DataFrame()

    out_df = pd.DataFrame({
        'intent': final_ordered_series.index.tolist(),
        'count': final_ordered_series.values
    })
    
    total_count = out_df['count'].sum()
    out_df['pct'] = (out_df['count'] / total_count * 100).round(1).fillna(0)
    out_df['text_label'] = out_df.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)

    out_df['intent'] = pd.Categorical(out_df['intent'], categories=order, ordered=True)
    out_df = out_df.sort_values('intent')
    
    return out_df

def display_placeholder(message, height_px):
    """Displays an info message followed by a spacer to match chart height."""
    st.info(message)
    spacer_height = max(0, height_px - 80) # Adjust 80px if st.info height varies
    st.markdown(f"<div style='height: {spacer_height}px;'></div>", unsafe_allow_html=True)

def display_business_question(question_text):
    """Displays a business question in a wide, styled horizontal strip."""
    st.markdown(
        """
        <style>
        .question-strip {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .question-strip h3 {
            font-size: 28px; /* Increase text size */
            font-weight: bold;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="question-strip"><h3>{question_text}</h3></div>',
        unsafe_allow_html=True
    )


@st.cache_data
def get_hourly_data_for_heatmap(df_subset, day_type_filter=None):
    """
    Prepares hourly call data for heatmaps, optionally filtering by day type.
    Includes percentages and returns only hours with a count > 0.
    """
    if df_subset.empty or '__hour' not in df_subset.columns or df_subset['__hour'].isna().all():
        # If no data or no valid hours, return an empty DataFrame
        return pd.DataFrame()

    current_df = df_subset.copy()
    if day_type_filter and day_type_filter != 'Overall' and 'DayType' in current_df.columns:
        current_df = current_df[current_df['DayType'] == day_type_filter]

    hourly_counts = current_df['__hour'].dropna().astype(int).value_counts().reindex(range(24), fill_value=0)
    
    df_result = pd.DataFrame({'Hour': hourly_counts.index, 'Count': hourly_counts.values})
    
    total_calls_in_subset = df_result['Count'].sum()
    
    if total_calls_in_subset > 0:
        df_result['Percentage'] = (df_result['Count'] / total_calls_in_subset * 100).round(1)
    else:
        df_result['Percentage'] = 0.0

    df_result['Hour_Label'] = df_result['Hour'].apply(lambda h: f"{h:02d}:00")

    # Only return rows where Count is greater than 0
    return df_result[df_result['Count'] > 0]


# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Config & Upload")
uploaded_file = st.sidebar.file_uploader("Upload call-analysis CSV", type=["csv"], accept_multiple_files=False)

st.sidebar.markdown("""
**Expected Columns (case-insensitive, whitespace stripped):**

- `CallType` (e.g., Sales / Service) - **Mandatory**
- `CallSentiment` (e.g., Positive/Negative/Neutral)
- `UserIntentToBuy` (e.g., Very Low / Low / Medium / High / Very High)
- `CallObjective` (text / theme)
- `NextAction` (text)
- `Time` or other datetime column (timestamp of call)
- `Customer_Language`
- `Region`, `State`, `City` (for geographical filters)
""")

df = pd.DataFrame()

if not uploaded_file:
    st.title("Call Analytics Dashboard — Mattress & Furniture India")
    st.info("Upload your CSV file from the sidebar to view the dashboard.")
    st.markdown("""
    **Getting Started:**
    1. Click 'Browse files' in the sidebar to upload your call analysis CSV.
    2. Ensure your CSV has columns like `CallType`, `Time`, `UserIntentToBuy`, etc. (See "Expected Columns" in sidebar).
    3. The dashboard will automatically update with your data.
    """)
    st.stop()

# ---------------------------
# Load & prepare
# ---------------------------
df = load_data(uploaded_file)

if df.empty:
    st.error("The uploaded file could not be processed or is empty after processing. Please check the file content and try again.")
    st.stop()

colnames = df.columns.tolist()

col_calltype = find_col(colnames, ['CallType', 'Call Type', 'calltype', 'call_type'])
col_sentiment = find_col(colnames, ['CallSentiment', 'Call Sentiment', 'Sentiment', 'call_sentiment'])
col_intent = find_col(colnames, ['UserIntentToBuy', 'User Intent to Buy', 'Intent', 'UserIntent', 'User_Intent'])
col_objective = find_col(colnames, ['CallObjective', 'Call Objective', 'Call_Objective', 'callobjective'])
col_nextaction = find_col(colnames, ['NextAction', 'Next Action', 'Next_Action', 'NextActionTaken'])
col_time = find_col(colnames, ['Time', 'CallTime', 'CreatedAt', 'CreatedDate', 'DateTime'])
col_language = find_col(colnames, ['Customer_Language', 'Customer Language', 'Language', 'customer_language'])
col_region = find_col(colnames, ['Region', 'region'])
col_state = find_col(colnames, ['State', 'state'])
col_city = find_col(colnames, ['City', 'city'])


dt_series, dt_col_used = ensure_datetime(df, [col_time] if col_time else [])
if dt_series is None:
    st.warning("No valid 'Time' or datetime column found. Hourly analysis will not be available.")
    df['__call_dt'] = pd.NaT
    df['__hour'] = np.nan
    df['DayOfWeek'] = 'Unknown' # Placeholder
    df['DayType'] = 'Unknown'   # Placeholder
else:
    df['__call_dt'] = dt_series
    df['__hour'] = df['__call_dt'].dt.hour
    df['DayOfWeek'] = df['__call_dt'].dt.day_name()
    # Monday=0, Tuesday=1, ..., Sunday=6. Weekends are Sat (5) and Sun (6)
    df['DayType'] = df['__call_dt'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')


if not col_calltype or not df[col_calltype].str.lower().isin(['sales', 'service', 'unknown']).any():
    st.error("Mandatory column 'CallType' not found or does not contain 'Sales'/'Service' values after cleaning. Please check your CSV.")
    st.stop()

# ---------------------------
# Filter Section (Sidebar)
# ---------------------------
st.sidebar.header("Filter Data")

filtered_df = df.copy()

all_regions_options = []
if col_region and not df[col_region].empty:
    all_regions_options = df[col_region].unique().tolist()
selected_region = st.sidebar.multiselect('Select Region', options=all_regions_options, default=all_regions_options)

if col_region and selected_region:
    filtered_df = filtered_df[filtered_df[col_region].isin(selected_region)]
else:
    if col_region is None: st.sidebar.info("No 'Region' column found for filtering.")
    elif not selected_region and col_region: st.sidebar.info("No regions selected; showing all data.")

all_states_options = []
if col_state:
    if not filtered_df.empty:
        all_states_options = filtered_df[col_state].unique().tolist()
    elif not df.empty:
        all_states_options = df[col_state].unique().tolist()

selected_state = st.sidebar.multiselect('Select State', options=all_states_options, default=all_states_options)

if col_state and selected_state:
    filtered_df = filtered_df[filtered_df[col_state].isin(selected_state)]
else:
    if col_state is None: st.sidebar.info("No 'State' column found for filtering.")
    elif not selected_state and col_state: st.sidebar.info("No states selected; showing all data.")

all_cities_options = []
if col_city:
    if not filtered_df.empty:
        all_cities_options = filtered_df[col_city].unique().tolist()
    elif not df.empty:
        all_cities_options = df[col_city].unique().tolist()

selected_city = st.sidebar.multiselect('Select City', options=all_cities_options, default=all_cities_options)

if col_city and selected_city:
    filtered_df = filtered_df[filtered_df[col_city].isin(selected_city)]
else:
    if col_city is None: st.sidebar.info("No 'City' column found for filtering.")
    elif not selected_city and col_city: st.sidebar.info("No cities selected; showing all data.")

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your filter selections or upload a different file.")
    st.stop()


# ---------------------------
# KPI calculations
# ---------------------------
total_calls = len(filtered_df)

sales_df = filtered_df[filtered_df[col_calltype].str.lower() == 'sales']
service_df = filtered_df[filtered_df[col_calltype].str.lower() == 'service']
num_sales = len(sales_df)
num_service = len(service_df)
pct_sales = (num_sales / total_calls * 100) if total_calls > 0 else 0
pct_service = (num_service / total_calls * 100) if total_calls > 0 else 0

# Sentiment distribution data
sentiment_order_for_plot = ['Positive', 'Negative', 'Neutral', 'Unknown']

# Customer Final Interest (Sales) - distribution for UserIntentToBuy
intent_df_sales = pd.DataFrame()
if col_intent and not sales_df.empty:
    intent_df_sales = ordered_intent_counts(sales_df, col_intent)

# Customer Final Interest (Service) - distribution for UserIntentToBuy (NEW)
intent_df_service = pd.DataFrame()
if col_intent and not service_df.empty:
    intent_df_service = ordered_intent_counts(service_df, col_intent)

# Top 5 Call Objective Themes - Sales & Service
top5_sales_obj = pd.DataFrame()
top5_service_obj = pd.DataFrame()
if col_objective:
    if not sales_df.empty:
        top5_sales_obj, _ = top_n_with_pct(sales_df[col_objective], n=5)
    if not service_df.empty:
        top5_service_obj, _ = top_n_with_pct(service_df[col_objective], n=5)

# Top 3 Next Actions - Sales & Service
top3_sales_next = pd.DataFrame()
top3_service_next = pd.DataFrame()
if col_nextaction:
    if not sales_df.empty:
        top3_sales_next, _ = top_n_with_pct(sales_df[col_nextaction], n=3)
    if not service_df.empty:
        top3_service_next, _ = top_n_with_pct(service_df[col_nextaction], n=3)

# Customer Language Volume & % for overall, sales, and service
lang_df_overall = pd.DataFrame()
lang_df_sales = pd.DataFrame()
lang_df_service = pd.DataFrame()

if col_language and not filtered_df.empty:
    vc_overall = filtered_df[col_language].value_counts(dropna=False)
    if not vc_overall.empty:
        lang_df_overall = pd.DataFrame({'language': vc_overall.index, 'count': vc_overall.values})
        lang_df_overall['pct'] = (lang_df_overall['count'] / lang_df_overall['count'].sum() * 100).round(1)
        lang_df_overall['text_label'] = lang_df_overall.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
        lang_df_overall = lang_df_overall.sort_values('count', ascending=False)

if col_language and not sales_df.empty:
    vc_sales = sales_df[col_language].value_counts(dropna=False)
    if not vc_sales.empty:
        lang_df_sales = pd.DataFrame({'language': vc_sales.index, 'count': vc_sales.values})
        lang_df_sales['pct'] = (lang_df_sales['count'] / lang_df_sales['count'].sum() * 100).round(1)
        lang_df_sales['text_label'] = lang_df_sales.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
        lang_df_sales = lang_df_sales.sort_values('count', ascending=False)

if col_language and not service_df.empty:
    vc_service = service_df[col_language].value_counts(dropna=False)
    if not vc_service.empty:
        lang_df_service = pd.DataFrame({'language': vc_service.index, 'count': vc_service.values})
        lang_df_service['pct'] = (lang_df_service['count'] / lang_df_service['count'].sum() * 100).round(1)
        lang_df_service['text_label'] = lang_df_service.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
        lang_df_service = lang_df_service.sort_values('count', ascending=False)


# ---------------------------
# Layout / Visuals
# ---------------------------
st.markdown("<h1 style='margin-bottom:6px'>Call Analytics Dashboard – Mattress & Furniture India</h1>", unsafe_allow_html=True)
st.markdown("All insights from store manager calls across India — sales and service performance at a glance.")

# --- Overall KPIs (Top Row) ---
col1, col2 = st.columns([1.2, 2]) # Adjusted columns for metric and single stacked bar

with col1:
    st.metric(label="Total Calls Analysed", value=f"{total_calls:,}", delta=f"Sales {num_sales:,} • Service {num_service:,}")

#with col2:
    # Q1. What type of Calls do we get? - Single Horizontal Stacked Bar Chart
display_business_question("Q1. What type of Calls do we get?")
st.markdown("<h5>Call Type Distribution</h5>", unsafe_allow_html=True)

#st.markdown("<h5>Call Type Distribution</h5>", unsafe_allow_html=True)

if total_calls > 0:
    # Prepare data
    call_type_data = pd.DataFrame({
        'Call Type': ['Sales', 'Service', 'Overall'],
        'Volume': [num_sales, num_service, num_sales + num_service]
    })

    # Calculate percentage and add label
    call_type_data['Percentage'] = (call_type_data['Volume'] / total_calls * 100).round(1)
    call_type_data['Label'] = call_type_data.apply(
        lambda row: f"{row['Volume']} ({row['Percentage']}%)",
        axis=1
    )

    # Custom color mapping
    color_map = {
        'Sales': CUSTOM_COLORS['Biscay'],
        'Service': CUSTOM_COLORS['Neptune'],
        'Overall': CUSTOM_COLORS['Mariner']
    }

    # Plot
    fig_call_type = px.bar(
        call_type_data,
        x='Volume',
        y='Call Type',
        orientation='h',
        text='Label',
        color='Call Type',
        color_discrete_map=color_map
    )

    # Trace formatting
    fig_call_type.update_traces(
        textposition='outside',
        cliponaxis=False,
        width=0.6
    )

    # Layout
    fig_call_type.update_layout(
        bargap=0.4,
        barmode='group',
        height=220,  # Slightly taller for text clarity
        margin=dict(t=40, b=40, l=80, r=40),
        showlegend=False,
        xaxis_title="Number of Calls",
        yaxis_title="",
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        xaxis_range=[0, call_type_data['Volume'].max() * 1.15]
    )

    st.plotly_chart(fig_call_type, use_container_width=True)
else:
    display_placeholder("No calls to analyze Call Type distribution.", height_px=200)


st.markdown("---")

# --- SALES INSIGHTS ---
st.header("Sales Insights")

# Q2. What are the Top Reasons Customers Call Us? (Sales)
display_business_question("Q2. What are the Top Reasons Customers Call Us? (Sales)")
st.markdown("<h5>Top 5 Call Objective Themes — Sales Calls</h5>", unsafe_allow_html=True)
if not top5_sales_obj.empty:
    max_count_sales_obj = top5_sales_obj['count'].max()
    fig = px.bar(top5_sales_obj[::-1], x='count', y='label', orientation='h',
                 labels={'count':'Volume','label':'Call Objective'}, text='text_label',
                 color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Rely on automargin for proper spacing
        height=350, # Keep generous height
        xaxis_range=[0, max_count_sales_obj * 1.3]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    display_placeholder("No 'CallObjective' data for Sales calls.", height_px=350)

# Q3. How Serious are the Callers? (Sales)
display_business_question("Q3. How Serious are the Callers? (Sales)")
st.markdown("<h5>Customer Final Interest (Sales) — Volume & %</h5>", unsafe_allow_html=True)
if not intent_df_sales.empty: # Changed from intent_df to intent_df_sales
    max_count_intent = intent_df_sales['count'].max()
    fig = px.bar(intent_df_sales, x='intent', y='count', text='text_label', labels={'intent':'Intent Level','count':'Volume'},
                 title="", color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig.update_traces(
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=40, r=40),
        yaxis_title='Volume', 
        xaxis_title='',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        height=250,
        yaxis_range=[0, max_count_intent * 1.3]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    display_placeholder("No 'UserIntentToBuy' data for sales calls, or the column is missing/empty after filtering.", height_px=250)

# Q4. What is the Customer Sentiment for the Sales Calls?
display_business_question("Q4. What is the Customer Sentiment for the Sales Calls?")
st.markdown("<h5>Sales Sentiment Distribution</h5>", unsafe_allow_html=True)
if col_sentiment:
    if not sales_df.empty and sales_df[col_sentiment].notna().any():
        sentiment_dist_sales = sales_df[col_sentiment].value_counts(dropna=False).reindex(sentiment_order_for_plot, fill_value=0)
        sentiment_dist_sales = sentiment_dist_sales[sentiment_dist_sales > 0] # Filter out categories with zero count
        
        if not sentiment_dist_sales.empty:
            sentiment_df_sales = sentiment_dist_sales.reset_index()
            sentiment_df_sales.columns = ['Sentiment', 'Count']
            sentiment_df_sales['Percentage'] = (sentiment_df_sales['Count'] / sentiment_df_sales['Count'].sum() * 100).round(1)
            sentiment_df_sales['Text'] = sentiment_df_sales.apply(lambda r: f"{r['Count']} ({r['Percentage']:.1f}%)", axis=1)
            max_count_sales_sent = sentiment_df_sales['Count'].max()

            fig_sales_sent = px.bar(
                sentiment_df_sales,
                x='Sentiment',
                y='Count',
                text='Text',
                labels={'Sentiment': 'Sentiment', 'Count': 'Number of Calls'},
                color='Sentiment',
                color_discrete_map=SENTIMENT_COLOR_MAP,
                category_orders={"Sentiment": sentiment_order_for_plot}
            )

            fig_sales_sent.update_traces(
                textposition='outside',
                textfont_size=16,
                cliponaxis=False,
                width=0.6
            )

            fig_sales_sent.update_layout(
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                margin=dict(t=40, b=80, l=40, r=40, autoexpand=True),
                height=400,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                xaxis_tickfont_size=16,
                yaxis_tickfont_size=16,
                yaxis_range=[0, max_count_sales_sent * 1.4],  # more padding for text
                bargap=1.0,
                xaxis_type='category'
            )

            
            st.plotly_chart(fig_sales_sent, use_container_width=True)
        else:
            display_placeholder("No sales sentiment data available for plotting.", height_px=250)
    else:
        display_placeholder("No sales calls or sentiment data available.", height_px=250)
else:
    display_placeholder("Column 'CallSentiment' not found. Detailed sentiment analysis is not available.", height_px=250)

# Q5. What are the Languages in which Customers Call? (Sales)
display_business_question("Q5. What are the Languages in which Customers Call? (Sales)")
st.markdown("<h5>Sales Customer Language — Volume & %</h5>", unsafe_allow_html=True)
if not lang_df_sales.empty:
    max_count_lang_sales = lang_df_sales['count'].max()
    fig_lang_sales = px.bar(lang_df_sales.head(10), x='count', y='language', orientation='h', text='text_label',
                            labels={'count':'Volume','language':'Language'}, title="",
                            color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig_lang_sales.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0, # Keep text horizontal on the bar
        cliponaxis=False
    )
    fig_lang_sales.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Enable auto margin to ensure labels fit
        height=350, # Keep generous height
        xaxis_range=[0, max_count_lang_sales * 1.3]
    ) 
    st.plotly_chart(fig_lang_sales, use_container_width=True)
else:
    display_placeholder("No 'Customer_Language' data for sales calls found or is empty after filtering.", height_px=350)

# Q6. What are the Next Action post Calls? (Sales)
display_business_question("Q6. What are the Next Action post Calls? (Sales)")
st.markdown("<h5>Top 3 Next Actions — Sales Calls</h5>", unsafe_allow_html=True)
if not top3_sales_next.empty:
    max_count_sales_next = top3_sales_next['count'].max()
    fig3 = px.bar(top3_sales_next[::-1], x='count', y='label', orientation='h', text='text_label',
                  labels={'label':'Next Action','count':'Volume'},
                  color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig3.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig3.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Rely on automargin for proper spacing
        height=350, # Keep generous height
        xaxis_range=[0, max_count_sales_next * 1.3]
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    display_placeholder("No 'NextAction' data for Sales calls.", height_px=350)

# Q7. What are the Peak Call Hours? (Sales)
display_business_question("Q7. What are the Peak Call Hours? (Sales)")
st.markdown("<h5>Sales Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
sales_day_type = st.selectbox(
    "Select Day Type for Sales Calls:",
    options=['Overall', 'Weekday', 'Weekend'],
    index=0,
    key='sales_day_type_selector'
)
current_sales_df_for_heatmap = sales_df
if sales_day_type != 'Overall':
    if 'DayType' not in sales_df.columns or sales_df['DayType'].isna().all():
        st.warning(f"Cannot filter sales calls by '{sales_day_type}': 'DayType' column not available or empty.")
        sales_day_type = 'Overall' 
    else:
        current_sales_df_for_heatmap = sales_df[sales_df['DayType'] == sales_day_type]

sales_hourly_data = get_hourly_data_for_heatmap(current_sales_df_for_heatmap, sales_day_type)

if not sales_hourly_data.empty and sales_hourly_data['Count'].sum() > 0:
    heatmap_fig_sales = go.Figure(data=go.Heatmap(
        z=sales_hourly_data['Count'].values.reshape(1, -1),
        x=sales_hourly_data['Hour_Label'].tolist(),
        y=['Sales Calls'],
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>",
        customdata=[sales_hourly_data['Percentage'].values.tolist()]
    ))
    heatmap_fig_sales.update_layout(
        height=150,
        margin=dict(t=40,b=80,l=40,r=40),
        xaxis_nticks=len(sales_hourly_data['Hour_Label']), # Display ticks only for hours with data
        xaxis_title="Hour of Day",
        yaxis_title="",
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=16,
    )
    st.plotly_chart(heatmap_fig_sales, use_container_width=True)
else:
    display_placeholder(f"No sales calls data to display hourly heatmap on {sales_day_type}.", height_px=150)

st.markdown("---")

# --- SERVICE INSIGHTS ---
st.header("Service Insights")

# Q2. What are the Top Reasons Customers Call Us? (Service)
display_business_question("Q2. What are the Top Reasons Customers Call Us? (Service)")
st.markdown("<h5>Top 5 Call Objective Themes — Service Calls</h5>", unsafe_allow_html=True)
if not top5_service_obj.empty:
    max_count_service_obj = top5_service_obj['count'].max()
    fig2 = px.bar(top5_service_obj[::-1], x='count', y='label', orientation='h', text='text_label',
                  labels={'count':'Volume','label':'Call Objective'},
                  color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig2.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig2.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Rely on automargin for proper spacing
        height=350, # Keep generous height
        xaxis_range=[0, max_count_service_obj * 1.3]
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    display_placeholder("No 'CallObjective' data for Service calls.", height_px=350)

# Q3. How Serious are the Callers? (Service) - NEW CHART FOR SERVICE
display_business_question("Q3. How Serious are the Callers? (Service)")
st.markdown("<h5>Customer Final Interest (Service) — Volume & %</h5>", unsafe_allow_html=True)
if not intent_df_service.empty:
    max_count_intent_service = intent_df_service['count'].max()
    fig_service_intent = px.bar(intent_df_service, x='intent', y='count', text='text_label', labels={'intent':'Intent Level','count':'Volume'},
                 title="", color_discrete_sequence=[CUSTOM_COLORS['Neptune']]) # Use a service-specific color
    fig_service_intent.update_traces(
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig_service_intent.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=40, r=40),
        yaxis_title='Volume', 
        xaxis_title='',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        height=250,
        yaxis_range=[0, max_count_intent_service * 1.3]
    )
    st.plotly_chart(fig_service_intent, use_container_width=True)
else:
    display_placeholder("No 'UserIntentToBuy' data for service calls, or the column is missing/empty after filtering.", height_px=250)


# Q4. What is the Customer Sentiment for the Service Calls?
display_business_question("Q4. What is the Customer Sentiment for the Service Calls?")
st.markdown("<h5>Service Sentiment Distribution</h5>", unsafe_allow_html=True)
if col_sentiment:
    if not service_df.empty and service_df[col_sentiment].notna().any():
        sentiment_dist_service = service_df[col_sentiment].value_counts(dropna=False).reindex(sentiment_order_for_plot, fill_value=0)
        sentiment_dist_service = sentiment_dist_service[sentiment_dist_service > 0]
        
        if not sentiment_dist_service.empty:
            sentiment_df_service = sentiment_dist_service.reset_index()
            sentiment_df_service.columns = ['Sentiment', 'Count']
            sentiment_df_service['Percentage'] = (sentiment_df_service['Count'] / sentiment_df_service['Count'].sum() * 100).round(1)
            sentiment_df_service['Text'] = sentiment_df_service.apply(lambda r: f"{r['Count']} ({r['Percentage']:.1f}%)", axis=1)
            max_count_service_sent = sentiment_df_service['Count'].max()

            fig_service_sent = px.bar(
                sentiment_df_service,
                x='Sentiment',
                y='Count',
                text='Text',
                labels={'Sentiment': 'Sentiment', 'Count': 'Number of Calls'},
                color='Sentiment',
                color_discrete_map=SENTIMENT_COLOR_MAP,
                category_orders={"Sentiment": sentiment_order_for_plot}
            )

            fig_service_sent.update_traces(
                textposition='outside',
                textfont_size=16,
                cliponaxis=False,
                width=0.6
            )

            fig_service_sent.update_layout(
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                margin=dict(t=40, b=80, l=40, r=40),
                height=500,
                #width=100,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                xaxis_tickfont_size=16,
                yaxis_tickfont_size=16,
                yaxis_range=[0, 100],  # more padding for text
                bargap=1.0,
                xaxis_type='category'
            )

            st.plotly_chart(fig_service_sent, use_container_width=True)
        else:
            display_placeholder("No service sentiment data available for plotting.", height_px=250)
    else:
        display_placeholder("No service calls or sentiment data available.", height_px=250)
else:
    display_placeholder("Column 'CallSentiment' not found. Detailed sentiment analysis is not available.", height_px=250)

# Q5. What are the Languages in which Customers Call? (Service)
display_business_question("Q5. What are the Languages in which Customers Call? (Service)")
st.markdown("<h5>Service Customer Language — Volume & %</h5>", unsafe_allow_html=True)
if not lang_df_service.empty:
    max_count_lang_service = lang_df_service['count'].max()
    fig_lang_service = px.bar(lang_df_service.head(10), x='count', y='language', orientation='h', text='text_label',
                              labels={'count':'Volume','language':'Language'}, title="",
                              color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig_lang_service.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0, # Keep text horizontal on the bar
        cliponaxis=False
    )
    fig_lang_service.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Enable auto margin to ensure labels fit
        height=350, # Keep generous height
        xaxis_range=[0, max_count_lang_service * 1.3]
    ) 
    st.plotly_chart(fig_lang_service, use_container_width=True)
else:
    display_placeholder("No 'Customer_Language' data for service calls found or is empty after filtering.", height_px=350)

# Q6. What are the Next Action post Calls? (Service)
display_business_question("Q6. What are the Next Action post Calls? (Service)")
st.markdown("<h5>Top 3 Next Actions — Service Calls</h5>", unsafe_allow_html=True)
if not top3_service_next.empty:
    max_count_service_next = top3_service_next['count'].max()
    fig4 = px.bar(top3_service_next[::-1], x='count', y='label', orientation='h', text='text_label',
                  labels={'label':'Next Action','count':'Volume'},
                  color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig4.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0,
        cliponaxis=False
    )
    fig4.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=40, b=80, l=150, r=40), # Adjusted left margin for horizontal text
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Rely on automargin for proper spacing
        height=350, # Keep generous height
        xaxis_range=[0, max_count_service_next * 1.3]
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    display_placeholder("No 'NextAction' data for Service calls.", height_px=350)

# Q7. What are the Peak Call Hours? (Service)
display_business_question("Q7. What are the Peak Call Hours? (Service)")
st.markdown("<h5>Service Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
service_day_type = st.selectbox(
    "Select Day Type for Service Calls:",
    options=['Overall', 'Weekday', 'Weekend'],
    index=0,
    key='service_day_type_selector'
)
current_service_df_for_heatmap = service_df
if service_day_type != 'Overall':
    if 'DayType' not in service_df.columns or service_df['DayType'].isna().all():
        st.warning(f"Cannot filter service calls by '{service_day_type}': 'DayType' column not available or empty.")
        service_day_type = 'Overall' 
    else:
        current_service_df_for_heatmap = service_df[service_df['DayType'] == service_day_type]

service_hourly_data = get_hourly_data_for_heatmap(current_service_df_for_heatmap, service_day_type)

if not service_hourly_data.empty and service_hourly_data['Count'].sum() > 0:
    heatmap_fig_service = go.Figure(data=go.Heatmap(
        z=service_hourly_data['Count'].values.reshape(1, -1),
        x=service_hourly_data['Hour_Label'].tolist(),
        y=['Service Calls'],
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>", # Corrected hovertemplate
        customdata=[service_hourly_data['Percentage'].values.tolist()]
    ))
    heatmap_fig_service.update_layout(
        height=150,
        margin=dict(t=40,b=80,l=40,r=40),
        xaxis_nticks=len(service_hourly_data['Hour_Label']), # Display ticks only for hours with data
        xaxis_title="Hour of Day",
        yaxis_title="",
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=16,
    )
    st.plotly_chart(heatmap_fig_service, use_container_width=True)
else:
    display_placeholder(f"No service calls data to display hourly heatmap on {service_day_type}.", height_px=150)

st.markdown("---")

# --- GENERAL INSIGHTS ---
st.header("General Insights")

# Q5. What are the Languages in which Customers Call? (Overall)
display_business_question("Q5. What are the Languages in which Customers Call? (Overall)")
st.markdown("<h5>Overall Customer Language — Volume & %</h5>", unsafe_allow_html=True)
if not lang_df_overall.empty:
    max_count_lang_overall = lang_df_overall['count'].max()
    fig_lang_overall = px.bar(lang_df_overall.head(10), x='count', y='language', orientation='h', text='text_label',
                              labels={'count':'Volume','language':'Language'}, title="",
                              color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
    fig_lang_overall.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        textfont_size=16, 
        textangle=0, # Keep text horizontal on the bar
        cliponaxis=False
    )
    fig_lang_overall.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(l=150, t=40, b=80, r=40), # Adjusted left margin
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Volume',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        yaxis_tickangle=0, # Set Y-axis labels horizontal
        yaxis_automargin=True, # Enable auto margin to ensure labels fit
        height=350, # Keep generous height
        xaxis_range=[0, max_count_lang_overall * 1.3]
    ) 
    st.plotly_chart(fig_lang_overall, use_container_width=True)
else:
    display_placeholder("No 'Customer_Language' data found or is empty after filtering.", height_px=350)

# Q7. What are the Peak Call Hours? (Overall)
display_business_question("Q7. What are the Peak Call Hours? (Overall)")
st.markdown("<h5>Overall Calls Hourly Volume Heatmap</h5>", unsafe_allow_html=True)
overall_day_type = st.selectbox(
    "Select Day Type for Overall Calls:",
    options=['Overall', 'Weekday', 'Weekend'],
    index=0,
    key='overall_day_type_selector'
)
current_total_df_for_heatmap = filtered_df
if overall_day_type != 'Overall':
    if 'DayType' not in filtered_df.columns or filtered_df['DayType'].isna().all():
        st.warning(f"Cannot filter overall calls by '{overall_day_type}': 'DayType' column not available or empty.")
        overall_day_type = 'Overall' 
    else:
        current_total_df_for_heatmap = filtered_df[filtered_df['DayType'] == overall_day_type]

overall_hourly_data = get_hourly_data_for_heatmap(current_total_df_for_heatmap, overall_day_type)

if not overall_hourly_data.empty and overall_hourly_data['Count'].sum() > 0:
    heatmap_fig_total = go.Figure(data=go.Heatmap(
        z=overall_hourly_data['Count'].values.reshape(1, -1),
        x=overall_hourly_data['Hour_Label'].tolist(),
        y=['Total Calls'],
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Hour: %{x}<br>Count: %{z}<br>Percentage: %{customdata:.1f}%<extra></extra>", # Corrected hovertemplate
        customdata=[overall_hourly_data['Percentage'].values.tolist()]
    ))
    heatmap_fig_total.update_layout(
        height=150,
        margin=dict(t=40,b=80,l=40,r=40),
        xaxis_nticks=len(overall_hourly_data['Hour_Label']), # Display ticks only for hours with data
        xaxis_title="Hour of Day",
        yaxis_title="",
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=16,
    )
    st.plotly_chart(heatmap_fig_total, use_container_width=True)
else:
    display_placeholder(f"No valid 'Time' or hour information found for overall calls on {overall_day_type}.", height_px=150)


# Option to show raw data table
with st.expander("Show raw data (first 200 rows)"):
    st.dataframe(df.head(200))

# Footer / notes
st.markdown("---")
st.markdown("**Notes:**")
st.markdown("""
- The dashboard adapts to your uploaded CSV. Column names are detected case-insensitively and whitespace is stripped.
- Missing values in key categorical columns are filled with 'Not Specified' or 'Unknown'.
- Intent order used: Very Low → Low → Medium → High → Very High → Not Specified.
- Sentiment detection expects `CallSentiment` to contain 'Positive', 'Negative', 'Neutral', or 'Unknown'. Missing sentiment is treated as 'Unknown'.
- Hours for the heatmap are extracted from the detected datetime column (e.g., `Time`, `CreatedDate`).
""")