import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO

# Page configuration
st.set_page_config(page_title="CloudMart Cost Governance Dashboard", layout="wide", page_icon="☁️")

# Title and description
st.title("☁️ CloudMart Resource Tagging & Cost Governance Dashboard")
st.markdown("""
This dashboard analyzes cloud resource tagging compliance and cost visibility for CloudMart Inc.
Upload your CSV file to begin the analysis.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CloudMart CSV File", type=['csv'])

# Initialize session state for edited data
if 'df_edited' not in st.session_state:
    st.session_state.df_edited = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# Load data function with RIGHT-ALIGNED parsing (last 3 columns always have values)
@st.cache_data
def load_data(file):
    try:
        # Read the file content
        content = file.read()
        
        # Decode if bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Split into lines
        lines = content.strip().split('\n')
        
        # Get header
        header_line = lines[0].strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[1:-1]
        
        headers = [h.strip() for h in header_line.split(',')]
        expected_cols = len(headers)
        
        # Process data rows with RIGHT-ALIGNMENT
        # Last 3 columns (CreatedBy, MonthlyCostUSD, Tagged) ALWAYS have values
        data_rows = []
        for i, line in enumerate(lines[1:], start=1):
            line = line.strip()
            
            # Remove surrounding quotes
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Split by comma
            fields = line.split(',')
            
            # If row has fewer fields than expected, RIGHT-ALIGN the last 3 columns
            if len(fields) < expected_cols:
                missing_count = expected_cols - len(fields)
                
                # Last 3 fields are always: CreatedBy, MonthlyCostUSD, Tagged
                first_part = fields[:-3]  # Everything except last 3
                last_three = fields[-3:]   # Last 3 fields that always have values
                
                # Insert empty fields in the middle (between first_part and last_three)
                padding = [''] * missing_count
                fields = first_part + padding + last_three
            
            # If row has more fields, truncate (shouldn't happen but just in case)
            elif len(fields) > expected_cols:
                fields = fields[:expected_cols]
            
            data_rows.append(fields)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Replace empty strings with NaN for better handling
        df.replace('', pd.NA, inplace=True)
        
        # Convert MonthlyCostUSD to numeric
        if 'MonthlyCostUSD' in df.columns:
            df['MonthlyCostUSD'] = pd.to_numeric(df['MonthlyCostUSD'], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.info("Debug info: Please check if your CSV file is properly formatted.")
        import traceback
        st.code(traceback.format_exc())
        return None

# Main application logic
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        # Store original dataframe
        if st.session_state.original_df is None:
            st.session_state.original_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['ResourceID', 'Service', 'MonthlyCostUSD', 'Tagged']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("**Available columns in your file:**")
            st.write(df.columns.tolist())
            st.info("**Required columns:**")
            st.write(required_columns)
            st.stop()
        
        # Sidebar navigation
        st.sidebar.title("📊 Navigation")
        task_set = st.sidebar.radio(
            "Select Task Set:",
            ["Overview", "Task 1: Data Exploration", "Task 2: Cost Visibility", 
             "Task 3: Tagging Compliance", "Task 4: Visualization Dashboard", 
             "Task 5: Tag Remediation"]
        )
        
        # ============================================
        # OVERVIEW PAGE
        # ============================================
        if task_set == "Overview":
            st.header("📋 Lab Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Resources", len(df))
            with col2:
                total_cost = df['MonthlyCostUSD'].sum()
                st.metric("Total Monthly Cost", f"${total_cost:,.2f}")
            with col3:
                tagged_pct = (df['Tagged'].value_counts().get('Yes', 0) / len(df)) * 100
                st.metric("Tagged Resources", f"{tagged_pct:.1f}%")
            with col4:
                departments = df['Department'].nunique() if 'Department' in df.columns else 0
                st.metric("Departments", departments)
            
            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display column information
            st.markdown("---")
            st.subheader("📊 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Columns:**", len(df.columns))
                st.write("**Rows:**", len(df))
            
            with col2:
                st.write("**Column Names:**")
                st.write(list(df.columns))
        
        # ============================================
        # TASK SET 1: DATA EXPLORATION
        # ============================================
        elif task_set == "Task 1: Data Exploration":
            st.header("🔍 Task Set 1: Data Exploration")
            
            # Task 1.1: Display first 5 rows
            st.subheader("Task 1.1: Display First 5 Rows")
            st.info("💡 Hint: Use pd.read_csv() or upload via Streamlit")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("---")
            
            # Task 1.2: Count missing values
            st.subheader("Task 1.2: Count Missing Values")
            st.info("💡 Hint: Use df.isnull().sum()")
            
            missing_values = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Missing %': (missing_values.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("---")
            
            # Task 1.3: Identify columns with most missing values
            st.subheader("Task 1.3: Columns with Most Missing Values")
            st.info("💡 Hint: Look for Department, Project, or Owner")
            
            most_missing = missing_df[missing_df['Missing Count'] > 0].nlargest(3, 'Missing Count')
            if not most_missing.empty:
                st.dataframe(most_missing, use_container_width=True)
                st.warning(f"⚠️ Top columns with missing values: {', '.join(most_missing['Column'].tolist())}")
                
                # Visualization
                fig = px.bar(missing_df[missing_df['Missing Count'] > 0], 
                             x='Column', y='Missing Count',
                             title='Missing Values by Column',
                             color='Missing %',
                             color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            st.markdown("---")
            
            # Task 1.4: Count tagged vs untagged
            st.subheader("Task 1.4: Count Tagged vs Untagged Resources")
            st.info("💡 Hint: Use df['Tagged'].value_counts()")
            
            tagged_counts = df['Tagged'].value_counts()
            total_resources = len(df)
            untagged_count = tagged_counts.get('No', 0)
            tagged_count = tagged_counts.get('Yes', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Resources", total_resources)
                st.metric("Tagged Resources", tagged_count)
                st.metric("Untagged Resources", untagged_count)
            
            with col2:
                # Pie chart
                fig = px.pie(tagged_counts.reset_index(), values='count', names='Tagged',
                            title='Tagged vs Untagged Distribution',
                            color='Tagged',
                            color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Task 1.5: Calculate percentage of untagged resources
            st.subheader("Task 1.5: Percentage of Untagged Resources")
            st.info("💡 Hint: Compute (untagged/total)*100")
            
            untagged_pct = (untagged_count / total_resources) * 100
            
            st.metric("Untagged Percentage", f"{untagged_pct:.2f}%")
            st.info(f"📊 Out of {total_resources} total resources, {untagged_count} are untagged ({untagged_pct:.2f}%)")
        
        # ============================================
        # TASK SET 2: COST VISIBILITY
        # ============================================
        elif task_set == "Task 2: Cost Visibility":
            st.header("💰 Task Set 2: Cost Visibility")
            
            # Task 2.1: Calculate total cost by tagging status
            st.subheader("Task 2.1: Total Cost by Tagging Status")
            st.info("💡 Hint: Group by 'Tagged' and sum 'MonthlyCostUSD'")
            
            cost_by_tagged = df.groupby('Tagged')['MonthlyCostUSD'].sum()
            total_cost = df['MonthlyCostUSD'].sum()
            untagged_cost = cost_by_tagged.get('No', 0)
            tagged_cost = cost_by_tagged.get('Yes', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col2:
                st.metric("Tagged Cost", f"${tagged_cost:,.2f}")
            with col3:
                st.metric("Untagged Cost", f"${untagged_cost:,.2f}")
            
            # Bar chart
            cost_df = pd.DataFrame({
                'Status': ['Tagged', 'Untagged'],
                'Cost': [tagged_cost, untagged_cost]
            })
            fig = px.bar(cost_df, x='Status', y='Cost', 
                        title='Cost by Tagging Status',
                        color='Status',
                        color_discrete_map={'Tagged': '#28a745', 'Untagged': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Task 2.2: Calculate percentage of untagged cost
            st.subheader("Task 2.2: Percentage of Untagged Cost")
            st.info("💡 Hint: Compute (untagged_cost/total_cost)*100")
            
            untagged_cost_pct = (untagged_cost / total_cost) * 100 if total_cost > 0 else 0
            
            st.metric("Untagged Cost Percentage", f"{untagged_cost_pct:.2f}%")
            st.info(f"📊 ${untagged_cost:,.2f} out of ${total_cost:,.2f} is untagged ({untagged_cost_pct:.2f}%)")
            
            st.markdown("---")
            
            # Task 2.3: Department with most untagged cost
            st.subheader("Task 2.3: Department with Most Untagged Cost")
            st.info("💡 Hint: Filter by Tagged=='No' and group by 'Department'")
            
            if 'Department' in df.columns:
                dept_cost = df[df['Tagged'] == 'No'].groupby('Department')['MonthlyCostUSD'].sum().sort_values(ascending=False)
                
                if not dept_cost.empty:
                    st.dataframe(dept_cost.reset_index().rename(columns={'MonthlyCostUSD': 'Untagged Cost (USD)'}), 
                                 use_container_width=True)
                    st.success(f"🏆 Department with most untagged cost: **{dept_cost.index[0]}** (${dept_cost.iloc[0]:,.2f})")
                    
                    # Bar chart
                    fig = px.bar(dept_cost.reset_index(), x='Department', y='MonthlyCostUSD',
                                 title='Untagged Cost by Department',
                                 color='MonthlyCostUSD',
                                 color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No untagged costs found!")
            else:
                st.warning("⚠️ 'Department' column not found in dataset")
            
            st.markdown("---")
            
            # Task 2.4: Project with most cost
            st.subheader("Task 2.4: Project Consuming Most Cost")
            st.info("💡 Hint: Use .groupby('Project')['MonthlyCostUSD'].sum()")
            
            if 'Project' in df.columns:
                project_cost = df.groupby('Project')['MonthlyCostUSD'].sum().sort_values(ascending=False)
                
                if not project_cost.empty:
                    st.dataframe(project_cost.reset_index().rename(columns={'MonthlyCostUSD': 'Total Cost (USD)'}),
                                 use_container_width=True)
                    st.success(f"🏆 Project consuming most cost: **{project_cost.index[0]}** (${project_cost.iloc[0]:,.2f})")
                    
                    # Bar chart
                    fig = px.bar(project_cost.reset_index(), x='Project', y='MonthlyCostUSD',
                                 title='Total Cost by Project',
                                 color='MonthlyCostUSD',
                                 color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 'Project' column not found in dataset")
            
            st.markdown("---")
            
            # Task 2.5: Compare Prod vs Dev vs Test
            st.subheader("Task 2.5: Compare Prod vs Dev vs Test Environments")
            st.info("💡 Hint: Group by 'Environment' and 'Tagged'")
            
            if 'Environment' in df.columns:
                # Grouped analysis
                env_analysis = df.groupby(['Environment', 'Tagged'])['MonthlyCostUSD'].sum().reset_index()
                
                fig = px.bar(env_analysis, x='Environment', y='MonthlyCostUSD',
                             color='Tagged', barmode='group',
                             title='Cost and Tagging Quality by Environment',
                             labels={'MonthlyCostUSD': 'Monthly Cost (USD)'},
                             color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                env_summary = df.groupby('Environment').agg({
                    'MonthlyCostUSD': 'sum',
                    'ResourceID': 'count'
                }).round(2)
                env_summary.columns = ['Total Cost (USD)', 'Resource Count']
                st.dataframe(env_summary, use_container_width=True)
            else:
                st.warning("⚠️ 'Environment' column not found in dataset")
        
        # ============================================
        # TASK SET 3: TAGGING COMPLIANCE
        # ============================================
        elif task_set == "Task 3: Tagging Compliance":
            st.header("✅ Task Set 3: Tagging Compliance")
            
            tag_fields = ['Department', 'Project', 'Environment', 'Owner', 'CostCenter']
            existing_tag_fields = [field for field in tag_fields if field in df.columns]
            
            # Task 3.1: Tag Completeness Score
            st.subheader("Task 3.1: Create Tag Completeness Score")
            st.info("💡 Hint: Count how many of the tag fields are non-empty")
            
            df_copy = df.copy()
            df_copy['TagCompletenessScore'] = df_copy[existing_tag_fields].notna().sum(axis=1)
            df_copy['CompletenessPercentage'] = (df_copy['TagCompletenessScore'] / len(existing_tag_fields)) * 100
            
            st.dataframe(df_copy[['ResourceID', 'Service', 'TagCompletenessScore', 
                                  'CompletenessPercentage', 'MonthlyCostUSD']].head(10),
                        use_container_width=True)
            
            st.markdown("---")
            
            # Task 3.2: Resources with lowest completeness
            st.subheader("Task 3.2: Identify Top 5 Resources with Lowest Completeness")
            st.info("💡 Hint: Sort by TagCompletenessScore ascending")
            
            display_columns = ['ResourceID', 'Service'] + existing_tag_fields + ['TagCompletenessScore', 'CompletenessPercentage', 'MonthlyCostUSD']
            
            lowest_completeness = df_copy.nsmallest(5, 'TagCompletenessScore')[
                [col for col in display_columns if col in df_copy.columns]
            ]
            st.dataframe(lowest_completeness, use_container_width=True)
            
            st.markdown("---")
            
            # Task 3.3: Most frequently missing tag fields
            st.subheader("Task 3.3: Most Frequently Missing Tag Fields")
            st.info("💡 Hint: Count missing entries per tag column")
            
            missing_tags = df[existing_tag_fields].isnull().sum().sort_values(ascending=False)
            missing_tags_df = pd.DataFrame({
                'Tag Field': missing_tags.index,
                'Missing Count': missing_tags.values,
                'Missing %': (missing_tags.values / len(df) * 100).round(2)
            })
            
            st.dataframe(missing_tags_df, use_container_width=True)
            
            if missing_tags.sum() > 0:
                fig = px.bar(missing_tags_df, x='Tag Field', y='Missing Count',
                             title='Missing Tag Fields',
                             color='Missing %',
                             color_continuous_scale='Oranges')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing tag fields!")
            
            st.markdown("---")
            
            # Task 3.4: List untagged resources
            st.subheader("Task 3.4: List Untagged Resources and Their Costs")
            st.info("💡 Hint: Filter where Tagged == 'No'")
            
            display_cols = ['ResourceID', 'Service', 'Region'] + existing_tag_fields + ['MonthlyCostUSD']
            
            untagged_resources = df[df['Tagged'] == 'No'][
                [col for col in display_cols if col in df.columns]
            ]
            st.dataframe(untagged_resources, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Untagged Resources", len(untagged_resources))
            with col2:
                st.metric("Total Untagged Cost", f"${untagged_resources['MonthlyCostUSD'].sum():,.2f}")
            
            st.markdown("---")
            
            # Task 3.5: Export untagged resources
            st.subheader("Task 3.5: Export Untagged Resources to CSV")
            st.info("💡 Hint: Use df[df['Tagged']=='No'].to_csv('untagged.csv')")
            
            csv_buffer = BytesIO()
            untagged_resources.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Untagged Resources CSV",
                data=csv_buffer,
                file_name="untagged_resources.csv",
                mime="text/csv"
            )
        
        # ============================================
        # TASK SET 4: VISUALIZATION DASHBOARD
        # ============================================
        elif task_set == "Task 4: Visualization Dashboard":
            st.header("📊 Task Set 4: Visualization Dashboard")
            
            # Task 4.5 at top: Interactive filters
            st.subheader("Task 4.5: Add Interactive Filters")
            st.info("💡 Hint: Use st.selectbox or st.multiselect")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                services = ['All'] + sorted(df['Service'].unique().tolist())
                selected_service = st.selectbox("Select Service", services)
            
            with col2:
                if 'Region' in df.columns:
                    regions = ['All'] + sorted(df['Region'].unique().tolist())
                    selected_region = st.selectbox("Select Region", regions)
                else:
                    selected_region = 'All'
            
            with col3:
                if 'Department' in df.columns:
                    departments = ['All'] + sorted(df['Department'].dropna().unique().tolist())
                    selected_department = st.selectbox("Select Department", departments)
                else:
                    selected_department = 'All'
            
            # Filter data
            filtered_df = df.copy()
            if selected_service != 'All':
                filtered_df = filtered_df[filtered_df['Service'] == selected_service]
            if selected_region != 'All' and 'Region' in df.columns:
                filtered_df = filtered_df[filtered_df['Region'] == selected_region]
            if selected_department != 'All' and 'Department' in df.columns:
                filtered_df = filtered_df[filtered_df['Department'] == selected_department]
            
            st.info(f"Showing {len(filtered_df)} resources (Total Cost: ${filtered_df['MonthlyCostUSD'].sum():,.2f})")
            
            st.markdown("---")
            
            # Task 4.1: Pie chart
            st.subheader("Task 4.1: Pie Chart - Tagged vs Untagged")
            st.info("💡 Hint: Use plotly.express.pie")
            
            tagged_counts = filtered_df['Tagged'].value_counts().reset_index()
            tagged_counts.columns = ['Tagged', 'Count']
            
            fig = px.pie(tagged_counts, values='Count', names='Tagged',
                         title='Resource Tagging Status Distribution',
                         color='Tagged',
                         color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Task 4.2: Bar chart by department
            st.subheader("Task 4.2: Bar Chart - Cost per Department by Tagging Status")
            st.info("💡 Hint: Use barmode='group'")
            
            if 'Department' in df.columns:
                dept_cost_tagged = filtered_df.groupby(['Department', 'Tagged'])['MonthlyCostUSD'].sum().reset_index()
                
                fig = px.bar(dept_cost_tagged, x='Department', y='MonthlyCostUSD',
                             color='Tagged', barmode='group',
                             title='Cost by Department and Tagging Status',
                             labels={'MonthlyCostUSD': 'Monthly Cost (USD)'},
                             color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 'Department' column not found")
            
            st.markdown("---")
            
            # Task 4.3: Horizontal bar chart
            st.subheader("Task 4.3: Horizontal Bar Chart - Total Cost per Service")
            st.info("💡 Hint: Group by 'Service' and use orientation='h'")
            
            service_cost = filtered_df.groupby('Service')['MonthlyCostUSD'].sum().sort_values().reset_index()
            
            fig = px.bar(service_cost, y='Service', x='MonthlyCostUSD',
                         orientation='h',
                         title='Total Cost by Service Type',
                         labels={'MonthlyCostUSD': 'Monthly Cost (USD)'},
                         color='MonthlyCostUSD',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Task 4.4: Environment cost distribution
            st.subheader("Task 4.4: Cost by Environment (Prod/Dev/Test)")
            st.info("💡 Hint: Pie or bar chart works")
            
            if 'Environment' in df.columns:
                env_cost = filtered_df.groupby('Environment')['MonthlyCostUSD'].sum().reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(env_cost, values='MonthlyCostUSD', names='Environment',
                                 title='Cost Distribution by Environment',
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(env_cost, x='Environment', y='MonthlyCostUSD',
                                 title='Cost by Environment',
                                 color='MonthlyCostUSD',
                                 color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 'Environment' column not found")
        
        # ============================================
        # TASK SET 5: TAG REMEDIATION WORKFLOW
        # ============================================
        elif task_set == "Task 5: Tag Remediation":
            st.header("🔧 Task Set 5: Tag Remediation Workflow")
            
            # Initialize edited dataframe
            if st.session_state.df_edited is None:
                st.session_state.df_edited = df.copy()
            
            # Task 5.1: Display editable table
            st.subheader("Task 5.1: Display Editable Table for Untagged Resources")
            st.info("💡 Hint: Use st.data_editor")
            
            st.markdown("💡 **Tip:** Double-click on any cell to edit. Fill in missing Department, Project, and Owner fields.")
            
            # Get untagged resources
            untagged_mask = st.session_state.df_edited['Tagged'] == 'No'
            untagged_df = st.session_state.df_edited[untagged_mask].copy()
            
            if len(untagged_df) > 0:
                st.info(f"Found {len(untagged_df)} untagged resources to remediate")
                
                # Determine which columns to make editable
                disabled_columns = ['AccountID', 'ResourceID', 'Service', 'Region', 'MonthlyCostUSD', 'CreatedBy', 'Tagged']
                
                # Display editable table
                edited_data = st.data_editor(
                    untagged_df,
                    use_container_width=True,
                    num_rows="fixed",
                    disabled=[col for col in disabled_columns if col in untagged_df.columns],
                    key='data_editor'
                )
            else:
                st.success("🎉 Great! There are no untagged resources to remediate.")
                edited_data = None
            
            st.markdown("---")
            
            # Task 5.2: Simulate remediation
            st.subheader("Task 5.2: Simulate Remediation by Filling Missing Tags")
            st.info("💡 Hint: Apply changes and update Tagged status")
            
            if edited_data is not None and len(edited_data) > 0:
                if st.button("✅ Apply Changes and Update Tagged Status"):
                    # Update the main dataframe with edited data
                    for idx, row in edited_data.iterrows():
                        # Check if all important fields are filled
                        has_dept = pd.notna(row.get('Department', None)) if 'Department' in row else True
                        has_project = pd.notna(row.get('Project', None)) if 'Project' in row else True
                        has_owner = pd.notna(row.get('Owner', None)) if 'Owner' in row else True
                        
                        if has_dept and has_project and has_owner:
                            st.session_state.df_edited.at[idx, 'Tagged'] = 'Yes'
                        
                        # Update all fields
                        for col in edited_data.columns:
                            st.session_state.df_edited.at[idx, col] = row[col]
                    
                    st.success("✅ Changes applied successfully! Resources with complete tags have been marked as 'Tagged'.")
                    st.rerun()
            
            st.markdown("---")
            
            # Task 5.3: Download updated dataset
            st.subheader("Task 5.3: Download the Remediated Dataset")
            st.info("💡 Hint: Use st.download_button")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_buffer = BytesIO()
                st.session_state.df_edited.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Remediated Dataset",
                    data=csv_buffer,
                    file_name="cloudmart_remediated.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Original dataset download
                csv_buffer_original = BytesIO()
                st.session_state.original_df.to_csv(csv_buffer_original, index=False)
                csv_buffer_original.seek(0)
                
                st.download_button(
                    label="📥 Download Original Dataset",
                    data=csv_buffer_original,
                    file_name="cloudmart_original.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            
            # Task 5.4: Before and after comparison
            st.subheader("Task 5.4: Compare Before and After Remediation")
            st.info("💡 Hint: Recalculate tagging metrics after updates")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Before Remediation")
                before_tagged = st.session_state.original_df['Tagged'].value_counts()
                before_untagged = before_tagged.get('No', 0)
                before_total = len(st.session_state.original_df)
                before_pct = (before_untagged / before_total) * 100
                before_cost = st.session_state.original_df[st.session_state.original_df['Tagged'] == 'No']['MonthlyCostUSD'].sum()
                
                st.metric("Untagged Resources", f"{before_untagged} ({before_pct:.2f}%)")
                st.metric("Untagged Cost", f"${before_cost:,.2f}")
            
            with col2:
                st.markdown("#### ✅ After Remediation")
                after_tagged = st.session_state.df_edited['Tagged'].value_counts()
                after_untagged = after_tagged.get('No', 0)
                after_total = len(st.session_state.df_edited)
                after_pct = (after_untagged / after_total) * 100
                after_cost = st.session_state.df_edited[st.session_state.df_edited['Tagged'] == 'No']['MonthlyCostUSD'].sum()
                
                improvement = before_untagged - after_untagged
                cost_improvement = before_cost - after_cost
                
                st.metric("Untagged Resources", f"{after_untagged} ({after_pct:.2f}%)", 
                         delta=f"-{improvement}", delta_color="inverse")
                st.metric("Untagged Cost", f"${after_cost:,.2f}",
                         delta=f"-${cost_improvement:,.2f}", delta_color="inverse")
            
            # Visualization comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Before', x=['Untagged Resources'], 
                                     y=[before_untagged], marker_color='#dc3545'))
                fig.add_trace(go.Bar(name='After', x=['Untagged Resources'], 
                                     y=[after_untagged], marker_color='#28a745'))
                fig.update_layout(title='Remediation Impact on Untagged Resources', 
                                 yaxis_title='Count', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Before', x=['Untagged Cost'], 
                                     y=[before_cost], marker_color='#dc3545'))
                fig.add_trace(go.Bar(name='After', x=['Untagged Cost'], 
                                     y=[after_cost], marker_color='#28a745'))
                fig.update_layout(title='Remediation Impact on Untagged Cost', 
                                 yaxis_title='Cost (USD)', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Task 5.5: Reflection
            st.subheader("Task 5.5: Reflection on Tagging Impact")
            st.info("💡 Hint: Write a short reflection on accountability")
            
            st.markdown("""
            ### 🎯 Key Insights on Tag Remediation Impact:
            
            **Cost Visibility & Accountability:**
            - Proper tagging enables accurate cost allocation to specific departments and projects
            - Financial teams can track budget consumption more effectively
            - Departments become accountable for their cloud spending
            
            **Governance Improvements:**
            - Clear ownership identification helps in resource lifecycle management
            - Better compliance with organizational policies
            - Easier identification of orphaned or unused resources
            
            **Operational Benefits:**
            - Simplified cost optimization initiatives
            - Enhanced ability to forecast and budget cloud expenses
            - Improved resource tracking and audit capabilities
            
            **Best Practices Recommendations:**
            1. Implement automated tagging policies during resource provisioning
            2. Enforce mandatory tags through cloud governance tools (AWS Organizations, Azure Policy)
            3. Regular audits to maintain tagging compliance
            4. Establish clear tagging standards across the organization
            5. Integrate tagging requirements into CI/CD pipelines (Terraform, CloudFormation)
            
            **Financial Impact:**
            - Reduction in untagged resources directly improves cost allocation accuracy
            - Better showback/chargeback reporting to departments
            - Identification of cost optimization opportunities
            """)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **CloudMart Cost Governance Dashboard**  
        Week 10 Activity - Cloud Economics  
        Fall 2025
        """)
        
else:
    # Landing page when no file is uploaded
    st.info("👆 Please upload your CloudMart CSV file to begin the analysis")
    
    st.markdown("""
    ### 📋 Lab Objectives
    By the end of this lab, you will be able to:
    - ✅ Understand the structure and importance of resource tagging in cloud environments
    - ✅ Measure tagging compliance and cost visibility
    - ✅ Identify untagged resources and quantify their hidden costs
    - ✅ Visualize cloud costs across departments, services, and environments
    - ✅ Simulate tag remediation and observe its effect on cost reporting
    
    ### 📂 Expected CSV Format
    Your CSV file should contain the following columns:
    - AccountID
    - ResourceID
    - Service
    - Region
    - Department
    - Project
    - Environment
    - Owner
    - CostCenter
    - CreatedBy
    - MonthlyCostUSD
    - Tagged
    """)
