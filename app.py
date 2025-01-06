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

# Create an empty premium table template from age 0 to 99
def create_empty_premium_table():
    """Create an empty premium table template from age 0 to 99"""
    return pd.DataFrame({
        'Age': range(100),
        'Other': [0] * 100
    }).set_index('Age')

# Get premiums for the selected plan and age range
def get_premiums(df, plan_name, current_age, max_age):
    """Get premiums for a plan from current age to max age"""
    if plan_name == 'Other':
        # Use custom plan data
        premiums = st.session_state.custom_plan['Other'].values[current_age:max_age]
    else:
        premiums = df[plan_name].values[current_age:max_age]
    return [round(p) for p in premiums]

# Calculate savings plan
def calculate_savings_plan(annual_saving, current_age, premiums, inflated_premiums=None, interest_rate=0.05):
    """
    Calculate the savings plan with compound interest and premium withdrawals
    
    Parameters:
    annual_saving: Annual saving amount for first 5 years
    current_age: Current age of the person
    premiums: List of annual premiums starting from current_age
    inflated_premiums: List of inflated premiums (if inflation rate > 0)
    interest_rate: Annual interest rate (default 5%)
    
    Returns:
    Tuple of (results list, boolean indicating if savings are insufficient)
    """
    results = []
    balance = 0
    saving_years = 5
    insufficient = False
    
    # First 5 years: Saving period
    for year in range(saving_years):
        age = current_age + year
        balance += annual_saving
        
        results.append({
            'Age': age,
            'Year': year + 1,
            'Savings': annual_saving,
            'Interest': 0,
            'Premium': 0,
            'Balance': balance
        })
    
    # After 5 years: Interest accrual and premium withdrawals
    for year in range(saving_years, len(premiums)):
        age = current_age + year
        # Calculate interest first (only if balance is positive)
        interest = max(0, balance * interest_rate)
        balance += interest
        
        # Withdraw premium
        premium = inflated_premiums[year] if inflated_premiums else premiums[year]
        
        # Check if balance will be insufficient
        if balance < premium:
            insufficient = True
            # Set withdrawal to remaining balance
            actual_premium = balance
            balance = 0
        else:
            actual_premium = premium
            balance -= premium
        
        results.append({
            'Age': age,
            'Year': year + 1,
            'Savings': 0,
            'Interest': round(interest),
            'Premium': actual_premium,
            'Balance': round(balance)
        })
    
    return results, insufficient

# Calculate minimum savings
def calculate_minimum_savings(current_age, premiums, inflated_premiums=None, interest_rate=0.05):
    """
    Calculate the minimum annual savings needed to cover all future premiums
    
    Parameters:
    current_age: Current age of the person
    premiums: List of annual premiums starting from current_age
    inflated_premiums: List of inflated premiums (if inflation rate > 0)
    interest_rate: Annual interest rate (default 5%)
    
    Returns:
    Minimum annual savings needed
    """
    # Binary search to find minimum savings
    min_savings = 0
    max_savings = max(premiums) * 2  # Start with a reasonable upper bound
    target_savings = None
    
    while min_savings <= max_savings:
        test_savings = (min_savings + max_savings) // 2
        results, insufficient = calculate_savings_plan(test_savings, current_age, premiums, inflated_premiums, interest_rate)
        
        if insufficient:
            min_savings = test_savings + 1
        else:
            target_savings = test_savings
            max_savings = test_savings - 1
    
    return target_savings if target_savings is not None else max_savings + 1

# Main application
def main():
    st.title("Medical Premium Calculator 醫療保費計算器")
    
    # Add CSS for fixed height headers and tabs
    st.markdown("""
        <style>
        .fixed-height-header {
            min-height: 4rem;
            display: flex;
            align-items: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px;
            padding: 10px 20px;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
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

    # Initialize session state for custom plan
    if 'custom_plan' not in st.session_state:
        st.session_state.custom_plan = create_empty_premium_table()

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Premium Calculator 保費計算器", "Medical Financing 醫療儲蓄"])
    
    with tab1:
        # Move all inputs to sidebar
        with st.sidebar:
            st.header("Input Parameters 輸入參數")
            
            # Custom Plan Editor
            with st.expander("Custom Plan Editor"):
                st.write("Enter premium values for custom plan (age 0-99)")
                edited_df = st.data_editor(
                    st.session_state.custom_plan,
                    use_container_width=True,
                    num_rows="dynamic",
                    disabled=["Age"],
                    hide_index=False
                )
                st.session_state.custom_plan = edited_df
            
            # User inputs
            gender = st.selectbox("性別 Gender", ["Male 男", "Female 女"])
            current_age = st.number_input("年齡 Age", min_value=0, max_value=99, value=30)
            
            # Select dataframe based on gender
            df = female_df if "Female" in gender else male_df
            
            # Get available plans and add custom plan
            plan_columns = [col for col in df.columns if col not in ['Age']] + ['Other']
            
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
            st.markdown(f'<div class="fixed-height-header"><h2>Plan 1: {plan1}</h2></div>', unsafe_allow_html=True)
            
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
                st.markdown(f'<div class="fixed-height-header"><h2>Plan 2: {plan2}</h2></div>', unsafe_allow_html=True)
                
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
                st.markdown('<div class="fixed-height-header"><h2>Combined Total</h2></div>', unsafe_allow_html=True)
                
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

    with tab2:
        st.header("Medical Financing Plan 醫療儲蓄計劃")
        st.write("""
        This calculator demonstrates how early savings can help offset future medical premiums through compound interest.
        The plan involves saving for 5 years, followed by interest accrual and premium withdrawals.
        """)
        
        # Saving inputs
        saving_col1, saving_col2 = st.columns(2)
        
        with saving_col1:
            # Set default saving amount based on currency
            default_saving = 10000 if currency == "USD 美元" else 80000
            annual_saving = st.number_input(
                "Annual Saving Amount 每年儲蓄金額",
                min_value=0,
                value=default_saving,
                step=1000,
                help="Amount to save each year for the first 5 years"
            )
            
            st.write(f"Total Savings after 5 years: {currency.split()[0]} {annual_saving * 5:,.0f}")
            
        with saving_col2:
            st.info("""
            **Plan Structure:**
            - Save for 5 years
            - 5% annual compound interest from Year 6
            - Annual withdrawal for premium payment from Year 6
            """)
        
        # Calculate savings plan
        if plan1 != "None":
            # Determine which premiums to use
            if plan2 != "None":
                # Combine premiums from both plans
                min_length = min(len(plan1_premiums), len(plan2_premiums))
                combined_premiums = [plan1_premiums[i] + plan2_premiums[i] for i in range(min_length)]
                if inflation_rate > 0:
                    combined_inflated = [plan1_inflated[i] + plan2_inflated[i] for i in range(min_length)]
                premiums_to_use = combined_premiums
                inflated_premiums = combined_inflated if inflation_rate > 0 else None
            else:
                # Use only plan 1
                premiums_to_use = plan1_premiums
                inflated_premiums = plan1_inflated if inflation_rate > 0 else None
            
            savings_results, insufficient = calculate_savings_plan(
                annual_saving,
                current_age,
                premiums_to_use,
                inflated_premiums
            )
            
            # Display warning if savings are insufficient
            if insufficient:
                # Calculate minimum required savings
                min_required_savings = calculate_minimum_savings(
                    current_age,
                    premiums_to_use,
                    inflated_premiums
                )
                
                st.error(f"""
                ⚠️ Warning: Insufficient Savings
                
                The proposed saving amount of {currency.split()[0]} {annual_saving:,.0f} per year is not sufficient 
                to cover the entire medical plan.
                
                Minimum required annual savings: {currency.split()[0]} {min_required_savings:,.0f}
                Additional savings needed: {currency.split()[0]} {(min_required_savings - annual_saving):,.0f} per year
                """)
            
            # Create detailed breakdown table
            df_savings = pd.DataFrame(savings_results)
            
            # Create savings visualization
            st.subheader("Savings and Premium Comparison")
            fig_savings = go.Figure()
            
            # Extract data for visualization
            ages = [r['Age'] for r in savings_results]
            balances = [r['Balance'] for r in savings_results]
            interests = [r['Interest'] for r in savings_results]
            premiums = [r['Premium'] for r in savings_results]
            savings = [r['Savings'] for r in savings_results]
            
            # Add traces
            fig_savings.add_trace(go.Bar(
                name='Annual Savings',
                x=ages,
                y=savings,
                marker_color='green'
            ))
            
            fig_savings.add_trace(go.Bar(
                name='Interest Earned',
                x=ages,
                y=interests,
                marker_color='blue'
            ))
            
            fig_savings.add_trace(go.Bar(
                name='Premium Paid',
                x=ages,
                y=premiums,
                marker_color='red'
            ))
            
            fig_savings.add_trace(go.Scatter(
                name='Balance',
                x=ages,
                y=balances,
                line=dict(color='purple', width=2),
                yaxis='y2'
            ))
            
            # Update layout for dual y-axis
            fig_savings.update_layout(
                barmode='group',
                yaxis=dict(
                    title=f"Annual Amount ({currency.split()[0]})",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title=f"Balance ({currency.split()[0]})",
                    titlefont=dict(color="purple"),
                    tickfont=dict(color="purple"),
                    overlaying="y",
                    side="right"
                ),
                xaxis=dict(title="Age"),
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig_savings, use_container_width=True)
            
            # Format currency values for display
            for col in ['Savings', 'Interest', 'Premium', 'Balance']:
                df_savings[col] = df_savings[col].apply(lambda x: f"{currency.split()[0]} {x:,.0f}")
            
            # Display results
            st.subheader("Annual Breakdown")
            st.dataframe(
                df_savings,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Age': 'Age 年齡',
                    'Year': 'Year 年份',
                    'Savings': 'Savings 儲蓄',
                    'Interest': 'Interest 利息',
                    'Premium': 'Premium 保費',
                    'Balance': 'Balance 結餘'
                }
            )
            
            # Calculate key metrics
            total_savings = annual_saving * 5
            total_interest = sum([float(result['Interest']) for result in savings_results])
            total_premiums = sum([float(result['Premium']) for result in savings_results])
            final_balance = float(savings_results[-1]['Balance'])
            
            # Display summary metrics
            st.subheader("Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Savings 總儲蓄", 
                         f"{currency.split()[0]} {total_savings:,.0f}")
            
            with metric_col2:
                st.metric("Total Interest 總利息",
                         f"{currency.split()[0]} {total_interest:,.0f}")
            
            with metric_col3:
                st.metric("Total Premiums 總保費",
                         f"{currency.split()[0]} {total_premiums:,.0f}")
            
            with metric_col4:
                st.metric("Final Balance 最終結餘",
                         f"{currency.split()[0]} {final_balance:,.0f}")

if __name__ == "__main__":
    main()
