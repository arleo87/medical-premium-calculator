import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Medical Premium Calculator",
    page_icon="🏥",
    layout="wide"
)

# Load data function
@st.cache_data
def load_data(gender):
    try:
        df = pd.read_excel('premiumtable.xlsx', sheet_name=gender.lower())
        # Round all premium values in the dataframe
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].round(0)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Calculate premiums with inflation
def calculate_inflated_premiums(premiums, inflation_rate, years):
    inflated_premiums = []
    premiums = [round(p) for p in premiums]  # Round input premiums
    for i in range(len(premiums)):
        inflated_premium = round(premiums[i] * (1 + inflation_rate/100) ** i)
        inflated_premiums.append(inflated_premium)
    return inflated_premiums

# Calculate 5-year averages
def calculate_5year_averages(premiums):
    averages = []
    premiums = [round(p) for p in premiums]  # Round input premiums
    for i in range(0, len(premiums)-4, 5):
        five_year_slice = premiums[i:i+5]
        if len(five_year_slice) == 5:
            avg = round(sum(five_year_slice) / 5)
            averages.append(avg)
    return averages

# Get premiums for the selected plan and age range
def get_premiums(df, plan, current_age, max_age):
    try:
        premiums = df[plan].values[current_age:max_age]
        return [round(float(p)) for p in premiums]  # Convert to float and round
    except Exception as e:
        st.error(f"Error getting premiums: {str(e)}")
        return []

# Main application
def main():
    st.title("Medical Premium Calculator 醫療保費計算器")
    
    # Load data
    try:
        female_df = load_data("female")
        male_df = load_data("male")
        if female_df is None or male_df is None:
            st.error("Error loading data")
            return
    except Exception as e:
        st.error("Error loading data. Please check if the Excel file exists and is accessible.")
        return

    # Move all inputs to sidebar
    with st.sidebar:
        st.header("Input Parameters 輸入參數")
        
        # User inputs
        gender = st.selectbox("性別 Gender", ["Male 男", "Female 女"])
        current_age = st.number_input("年齡 Age", min_value=0, max_value=99, value=30)
        
        # Select dataframe based on gender
        df = female_df if "Female" in gender else male_df
        
        # Get available plans
        plan_columns = [col for col in df.columns if col not in ['Age']]
        
        # Plan selection
        plan1 = st.selectbox("計劃 1 Plan 1", plan_columns)
        plan2 = st.selectbox("計劃 2 Plan 2 (Optional)", ["None"] + plan_columns)
        
        # Inflation rate
        inflation_rate = st.slider("通脹率 Inflation Rate (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        
        # Currency selection
        currency = st.selectbox("貨幣 Currency", ["HKD 港幣", "USD 美元"])
        exchange_rate = 7.85
        if currency == "USD 美元":
            exchange_rate = st.number_input("匯率 Exchange Rate (HKD to USD)", min_value=1.0, value=7.85, step=0.01)

    # Get maximum age from data
    max_age = len(df)
    
    # Calculate years
    years = max_age - current_age
    
    # Get premiums for selected plans
    plan1_premiums = get_premiums(df, plan1, current_age, max_age)
    if plan2 != "None":
        plan2_premiums = get_premiums(df, plan2, current_age, max_age)
    
    # Apply currency conversion
    if currency == "USD 美元":
        plan1_premiums = [round(p / exchange_rate) for p in plan1_premiums]
        if plan2 != "None":
            plan2_premiums = [round(p / exchange_rate) for p in plan2_premiums]
    
    # Calculate inflated premiums
    plan1_inflated = calculate_inflated_premiums(plan1_premiums, inflation_rate, len(plan1_premiums))
    if plan2 != "None":
        plan2_inflated = calculate_inflated_premiums(plan2_premiums, inflation_rate, len(plan2_premiums))

    # Create three columns for results
    col1, col2, col3 = st.columns(3)

    # Column 1: Plan 1
    with col1:
        st.header(f"Plan 1: {plan1}")
        
        # Premium Growth Projection
        st.subheader("醫療保費預計增長情況")
        fig1 = go.Figure()
        ages = list(range(current_age, max_age))
        
        fig1.add_trace(go.Scatter(x=ages, y=plan1_premiums, 
                                name="Original", line=dict(color='blue')))
        if inflation_rate > 0:
            fig1.add_trace(go.Scatter(x=ages, y=plan1_inflated, 
                                    name="Inflated", line=dict(color='red')))
        
        fig1.update_layout(
            xaxis_title="Age",
            yaxis_title=f"Premium ({currency.split()[0]})",
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Total Premiums
        st.subheader("總保費")
        total_original = sum(plan1_premiums)
        st.write(f"Original Total: {currency.split()[0]} {total_original:,.0f}")
        if inflation_rate > 0:
            total_inflated = sum(plan1_inflated)
            st.write(f"Inflated Total: {currency.split()[0]} {total_inflated:,.0f}")

        # Premium Table
        with st.expander("保費表", expanded=True):
            premium_table = pd.DataFrame({
                'Age': ages,
                'Original': plan1_premiums
            })
            if inflation_rate > 0:
                premium_table['Inflated'] = plan1_inflated
            st.dataframe(premium_table, use_container_width=True)

        # 5-Year Average Premiums
        with st.expander("每5年平均保費", expanded=True):
            avg_premiums = calculate_5year_averages(plan1_premiums)
            age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
            avg_table = pd.DataFrame({
                'Age Range': age_ranges,
                'Original': avg_premiums
            })
            if inflation_rate > 0:
                avg_inflated = calculate_5year_averages(plan1_inflated)
                if len(avg_inflated) == len(avg_premiums):
                    avg_table['Inflated'] = avg_inflated
            st.dataframe(avg_table, use_container_width=True)

    # Column 2: Plan 2 (if selected)
    with col2:
        if plan2 != "None":
            st.header(f"Plan 2: {plan2}")
            
            # Premium Growth Projection
            st.subheader("醫療保費預計增長情況")
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(x=ages, y=plan2_premiums, 
                                    name="Original", line=dict(color='green')))
            if inflation_rate > 0:
                fig2.add_trace(go.Scatter(x=ages, y=plan2_inflated, 
                                        name="Inflated", line=dict(color='orange')))
            
            fig2.update_layout(
                xaxis_title="Age",
                yaxis_title=f"Premium ({currency.split()[0]})",
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Total Premiums
            st.subheader("總保費")
            total_original = sum(plan2_premiums)
            st.write(f"Original Total: {currency.split()[0]} {total_original:,.0f}")
            if inflation_rate > 0:
                total_inflated = sum(plan2_inflated)
                st.write(f"Inflated Total: {currency.split()[0]} {total_inflated:,.0f}")

            # Premium Table
            with st.expander("保費表", expanded=True):
                premium_table = pd.DataFrame({
                    'Age': ages,
                    'Original': plan2_premiums
                })
                if inflation_rate > 0:
                    premium_table['Inflated'] = plan2_inflated
                st.dataframe(premium_table, use_container_width=True)

            # 5-Year Average Premiums
            with st.expander("每5年平均保費", expanded=True):
                avg_premiums = calculate_5year_averages(plan2_premiums)
                age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
                avg_table = pd.DataFrame({
                    'Age Range': age_ranges,
                    'Original': avg_premiums
                })
                if inflation_rate > 0:
                    avg_inflated = calculate_5year_averages(plan2_inflated)
                    if len(avg_inflated) == len(avg_premiums):
                        avg_table['Inflated'] = avg_inflated
                st.dataframe(avg_table, use_container_width=True)

    # Column 3: Combined (if Plan 2 is selected)
    with col3:
        if plan2 != "None":
            st.header("Combined Total")
            
            # Combined Premium Growth Projection
            st.subheader("醫療保費預計增長情況")
            fig_combined = go.Figure()
            
            # Calculate combined premiums
            combined_premiums = []
            combined_inflated = []
            
            # Get the minimum length to ensure arrays match
            min_length = min(len(plan1_premiums), len(plan2_premiums))
            plan1_premiums_adj = [round(p) for p in plan1_premiums[:min_length]]
            plan2_premiums_adj = [round(p) for p in plan2_premiums[:min_length]]
            
            # Calculate combined premiums
            for p1, p2 in zip(plan1_premiums_adj, plan2_premiums_adj):
                combined_premiums.append(round(p1 + p2))
            
            # Update ages array to match the length
            ages_combined = list(range(current_age, current_age + min_length))
            
            fig_combined.add_trace(go.Scatter(x=ages_combined, y=combined_premiums, 
                                            name="Original", line=dict(color='purple')))
            
            if inflation_rate > 0:
                # Calculate combined inflated premiums
                plan1_inflated_adj = plan1_inflated[:min_length]
                plan2_inflated_adj = plan2_inflated[:min_length]
                
                for p1, p2 in zip(plan1_inflated_adj, plan2_inflated_adj):
                    combined_inflated.append(round(p1 + p2))
                
                fig_combined.add_trace(go.Scatter(x=ages_combined, y=combined_inflated, 
                                                name="Inflated", line=dict(color='red')))
            
            fig_combined.update_layout(
                xaxis_title="Age",
                yaxis_title=f"Premium ({currency.split()[0]})",
                showlegend=True
            )
            st.plotly_chart(fig_combined, use_container_width=True)

            # Combined Total Premiums
            st.subheader("總保費 (Combined)")
            combined_original = sum(combined_premiums)
            st.write(f"Original Total: {currency.split()[0]} {combined_original:,.0f}")
            
            if inflation_rate > 0:
                combined_inflated_total = sum(combined_inflated)
                st.write(f"Inflated Total: {currency.split()[0]} {combined_inflated_total:,.0f}")

            # Combined Premium Table
            with st.expander("保費表 (Combined)", expanded=True):
                combined_table = pd.DataFrame({
                    'Age': ages_combined,
                    'Original': combined_premiums
                })
                if inflation_rate > 0:
                    combined_table['Inflated'] = combined_inflated
                st.dataframe(combined_table, use_container_width=True)

            # Combined 5-Year Average Premiums
            with st.expander("每5年平均保費 (Combined)", expanded=True):
                avg_premiums = calculate_5year_averages(combined_premiums)
                age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
                avg_table = pd.DataFrame({
                    'Age Range': age_ranges,
                    'Original': avg_premiums
                })
                if inflation_rate > 0:
                    avg_inflated = calculate_5year_averages(combined_inflated)
                    if len(avg_inflated) == len(avg_premiums):
                        avg_table['Inflated'] = avg_inflated
                st.dataframe(avg_table, use_container_width=True)

if __name__ == "__main__":
    main()
