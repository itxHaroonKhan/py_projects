import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="📊 Data Analysis Pro", layout="wide")

# Add title
st.title("📊 Advanced Data Analysis Pro")
st.subheader("A Comprehensive Data Analysis Tool - Developed by @Haroon")
st.markdown("---")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Dataset selection
data_option = st.radio("📌 Choose Data Source:", ["Upload Your Own File", "Select a Preloaded Dataset"])

if data_option == "Upload Your Own File":
    uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            st.session_state.df = df
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

elif data_option == "Select a Preloaded Dataset":
    dataset_list = ["tips", "iris", "flights", "mpg", "diamonds", "planets", "taxis", "titanic"]
    selected_dataset = st.selectbox("📌 Select a Dataset", dataset_list)
    st.session_state.df = sns.load_dataset(selected_dataset)
    st.success(f"✅ Loaded dataset: {selected_dataset}")

# Main analysis section
if not st.session_state.df.empty:
    df = st.session_state.df

    # Data Cleaning and Preprocessing
    with st.expander("🧹 Data Cleaning & Preprocessing", expanded=False):
        st.subheader("Data Cleaning Tools")

        col1, col2 = st.columns(2)

        with col1:
            # Missing values handling
            st.markdown("**Handle Missing Values**")
            if st.checkbox("Show Missing Values Summary"):
                missing = df.isnull().sum()
                st.write(missing[missing > 0])

            fill_option = st.selectbox("Fill Missing Values:",
                ["None", "Mean", "Median", "Mode", "Specific Value"])

            if fill_option != "None":
                selected_columns = st.multiselect("Select columns:", df.columns)
                if selected_columns:
                    for col in selected_columns:
                        if fill_option == "Mean" and df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif fill_option == "Median" and df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].median(), inplace=True)
                        elif fill_option == "Mode":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        elif fill_option == "Specific Value":
                            value = st.text_input(f"Enter value for {col}:")
                            if value:
                                try:
                                    df[col] = df[col].fillna(value)
                                except:
                                    st.error("Invalid value type for this column")
            st.session_state.df = df

        with col2:
            # Column operations
            st.markdown("**Column Operations**")
            drop_option = st.selectbox("Select Operation:", ["None", "Drop Columns", "Rename Columns"])

            if drop_option == "Drop Columns":
                columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    st.session_state.df = df

            elif drop_option == "Rename Columns":
                col_to_rename = st.selectbox("Select column:", df.columns)
                new_name = st.text_input("New column name:")
                if st.button("Rename"):
                    df = df.rename(columns={col_to_rename: new_name})
                    st.session_state.df = df

    # Dataset Preview Section
    st.header("📜 Dataset Preview")
    with st.expander("Dataset Summary"):
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", list(df.columns))

    # Search & Filter
    if st.checkbox("🔍 Enable Advanced Filtering"):
        filter_query = st.text_input("Enter filter query (e.g., age > 30 and gender == 'male'):")
        if filter_query:
            try:
                df = df.query(filter_query)
                st.session_state.df = df
            except Exception as e:
                st.error(f"Invalid filter query: {str(e)}")

    # Data Visualization Section
    st.header("📊 Interactive Visualizations")
    plot_type = st.selectbox("📌 Choose Visualization Type:", [
        "Scatter Plot", "Line Chart", "Bar Chart", "Histogram",
        "Box Plot", "Violin Plot", "Correlation Heatmap", "3D Scatter Plot"
    ])

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")
        color_palette = st.selectbox("🎨 Color Palette:", ["viridis", "plasma", "magma", "inferno", "rainbow"])
        height = st.slider("Chart Height", 400, 1000, 600)

    with col2:
        st.subheader("Visualization")

        if plot_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", df.columns)
            color_by = st.selectbox("Color by", [None] + list(df.columns))
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                            height=height, color_continuous_scale=color_palette)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "3D Scatter Plot":
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", df.columns)
            z_axis = st.selectbox("Z-axis", df.columns)
            color_by = st.selectbox("Color by", [None] + list(df.columns))
            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_by,
                               color_continuous_scale=color_palette)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Histogram":
            column = st.selectbox("Select Column:", df.columns)
            bins = st.slider("Number of Bins", 5, 100, 20)
            fig = px.histogram(df, x=column, nbins=bins, color_discrete_sequence=[color_palette])
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Correlation Heatmap":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale=color_palette))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")

    # Advanced Analysis Section
    with st.expander("🔬 Advanced Analysis Tools", expanded=False):
        st.subheader("Statistical Analysis")
        if st.checkbox("Run Statistical Summary"):
            st.write(df.describe(include='all'))

        if st.checkbox("Perform Group Analysis"):
            group_col = st.selectbox("Group by:", df.columns)
            agg_col = st.selectbox("Analyze column:", df.select_dtypes(include=np.number).columns)
            agg_type = st.selectbox("Aggregation:", ["mean", "sum", "count", "min", "max"])
            grouped = df.groupby(group_col)[agg_col].agg(agg_type)
            st.write(grouped)

    # Data Export Section
    st.header("💾 Export Options")
    if st.button("Download Processed Data as CSV"):
        csv = df.to_csv(index=False).encode()
        st.download_button(label="Download CSV", data=csv,
                          file_name="processed_data.csv", mime="text/csv")

    # Sidebar Information
    st.sidebar.header("🔍 Dataset Insights")
    st.sidebar.subheader("Quick Statistics")
    st.sidebar.write(f"Total Records: {df.shape[0]}")
    st.sidebar.write(f"Total Features: {df.shape[1]}")

    st.sidebar.subheader("Data Types Overview")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['Data Type', 'Count']
    st.sidebar.dataframe(dtype_counts)

else:
    st.info("ℹ️ Please upload a file or select a dataset to begin analysis")

# Footer
st.markdown("---")
st.markdown("🚀 Developed with ❤️ by Haroon | 📧 Contact: itxharoonkhan@gmail.com")
