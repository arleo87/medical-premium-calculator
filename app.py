import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import io
from saving_plan import display_saving_plan

# Constants for age milestones
AGE_MILESTONES = [65, 75, 85, 99]

# Set page config
st.set_page_config(
    page_title="Medical Premium Calculator",
    page_icon="",
    layout="wide"
)

# Load data function
@st.cache_data(ttl=1)  # Cache will expire after 1 second
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
    """Calculate 5-year averages starting from the 6th year"""
    if len(premiums) < 6:  # Need at least 6 years of data
        return []
    
    # Start from 6th year (index 5)
    remaining_premiums = premiums[5:]
    
    # Calculate averages for each 5-year period
    averages = []
    for i in range(0, len(remaining_premiums), 5):
        five_year_slice = remaining_premiums[i:i+5]
        if len(five_year_slice) == 5:  # Only include complete 5-year periods
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
def calculate_savings_plan(annual_saving, current_age, premiums, inflated_premiums=None, interest_rate=0.05, withdrawal_at_65=0):
    """
    Calculate the savings plan with compound interest and premium withdrawals
    
    Parameters:
    annual_saving: Annual saving amount for first 5 years
    current_age: Current age of the person
    premiums: List of annual premiums starting from current_age
    inflated_premiums: List of inflated premiums (if inflation rate > 0)
    interest_rate: Annual interest rate (default 5%)
    withdrawal_at_65: One-time withdrawal amount at age 65
    
    Returns:
    Tuple of (results list, boolean indicating if savings are insufficient, age when insufficient, withdrawal amount that couldn't be fully made)
    """
    results = []
    savings = 0
    insufficient = False
    insufficient_age = None
    insufficient_amount = None
    premiums_to_use = inflated_premiums if inflated_premiums is not None else premiums
    
    for year in range(len(premiums_to_use)):
        age = current_age + year
        
        # Apply interest starting from year 6
        interest = savings * interest_rate if year >= 5 else 0
        
        # Add annual saving for first 5 years
        saving_this_year = annual_saving if year < 5 else 0
        
        # Calculate available balance before withdrawals
        available_balance = savings + interest + saving_this_year
        
        # Calculate withdrawals (only after year 5)
        premium_withdrawal = premiums_to_use[year] if year >= 5 else 0
        age_65_withdrawal = withdrawal_at_65 if age == 65 else 0
        total_withdrawal = premium_withdrawal + age_65_withdrawal
        
        # Check if withdrawal would make balance negative
        if total_withdrawal > available_balance:
            if not insufficient:
                insufficient = True
                insufficient_age = age
                insufficient_amount = total_withdrawal - available_balance
            # Adjust withdrawal to available balance
            if premium_withdrawal > available_balance:
                premium_withdrawal = available_balance
                age_65_withdrawal = 0
            else:
                age_65_withdrawal = min(age_65_withdrawal, available_balance - premium_withdrawal)
        
        # Calculate new savings
        new_savings = available_balance - premium_withdrawal - age_65_withdrawal
        
        # Store results
        results.append({
            'Age': age,
            'Start': round(savings),
            'Interest': round(interest),
            'Saving': round(saving_this_year),
            'Premium': round(premium_withdrawal),
            'Age65': round(age_65_withdrawal),
            'End': round(new_savings)
        })
        
        savings = new_savings
    
    return results, insufficient, insufficient_age, insufficient_amount

# Calculate minimum savings
def calculate_minimum_savings(current_age, premiums, inflated_premiums=None, interest_rate=0.05, withdrawal_at_65=0):
    """
    Calculate the minimum annual savings needed to cover all future premiums
    
    Parameters:
    current_age: Current age of the person
    premiums: List of annual premiums starting from current_age
    inflated_premiums: List of inflated premiums (if inflation rate > 0)
    interest_rate: Annual interest rate (default 5%)
    withdrawal_at_65: One-time withdrawal amount at age 65
    
    Returns:
    Minimum annual savings needed
    """
    min_saving = 0
    max_saving = max(premiums) * 2  # Start with a reasonable upper bound
    
    while max_saving - min_saving > 1:  # Binary search with precision of 1
        mid = (min_saving + max_saving) / 2
        results, insufficient, _, _ = calculate_savings_plan(mid, current_age, premiums, inflated_premiums, interest_rate, withdrawal_at_65)
        
        if insufficient:
            min_saving = mid
        else:
            max_saving = mid
    
    return round(max_saving)

# Calculate total premium to age
def calculate_total_premium_to_age(premiums, current_age, target_age):
    """Calculate total premium from current age to target age"""
    if target_age <= current_age:
        return 0
    end_index = min(target_age - current_age, len(premiums))
    return sum(premiums[:end_index])

# Main application
def main():
    st.title("Medical Insurance Planning Tool")
    
    # Initialize session state for premium if not exists
    if 'medical_premium' not in st.session_state:
        st.session_state.medical_premium = 0
    
    # Add CSS for fixed height headers and tabs
    st.markdown("""
        <style>
        .fixed-height-header {
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background-color: #f0f2f6;
        }
        .fixed-height-header.combined {
            background-color: #e6f3e6;
        }
        .fixed-height-header h2 {
            margin: 0;
            text-align: center;
            font-size: 1.3em;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] div[data-testid="stTable"] {
            text-align: left !important;
        }
        div[data-testid="stDataFrame"] td, 
        div[data-testid="stDataFrame"] th {
            text-align: left !important;
            white-space: nowrap !important;
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
    tab1, tab2, tab3 = st.tabs(["Premium Calculator ", "Medical Financing ", "Our Saving Plan"])
    
    with tab1:
        # Move all inputs to sidebar
        with st.sidebar:
            st.header("Input Parameters ")
            
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
            gender = st.selectbox("", ["Male ", "Female "])
            current_age = st.number_input("", min_value=0, max_value=99, value=30)
            
            # Select dataframe based on gender
            df = female_df if "Female" in gender else male_df
            
            # Get available plans and add custom plan
            plan_columns = [col for col in df.columns if col not in ['Age']] + ['Other']
            
            # Plan selection
            plan1 = st.selectbox(" 1", plan_columns)
            plan2 = st.selectbox(" 2 (Optional)", ["None"] + plan_columns)
            
            # Inflation rate
            inflation_rate = st.slider(" (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
            
            # Currency selection
            currency = st.selectbox("", ["HKD ", "USD "])
            exchange_rate = 7.85
            if currency == "USD ":
                exchange_rate = st.number_input(" (HKD to USD)", min_value=1.0, value=7.85, step=0.01)
            
            # Store currency and exchange rate in session state
            st.session_state['currency'] = currency
            st.session_state['exchange_rate'] = exchange_rate
        
        # Get maximum age from data
        max_age = len(df)
        
        # Calculate years
        years = max_age - current_age
        
        # Get premiums for selected plans
        plan1_premiums = get_premiums(df, plan1, current_age, max_age)
        if plan2 != "None":
            plan2_premiums = get_premiums(df, plan2, current_age, max_age)
        
        # Apply currency conversion
        if currency == "USD ":
            plan1_premiums = [round(p / exchange_rate) for p in plan1_premiums]
            if plan2 != "None":
                plan2_premiums = [round(p / exchange_rate) for p in plan2_premiums]
        
        # Calculate inflated premiums
        plan1_inflated = calculate_inflated_premiums(plan1_premiums, inflation_rate, len(plan1_premiums))
        if plan2 != "None":
            plan2_inflated = calculate_inflated_premiums(plan2_premiums, inflation_rate, len(plan2_premiums))

        # Store premium data in session state for both medical financing and saving plan
        if plan2 != "None":
            # Combine premiums from both plans
            min_length = min(len(plan1_premiums), len(plan2_premiums))
            combined_premiums = [plan1_premiums[i] + plan2_premiums[i] for i in range(min_length)]
            if inflation_rate > 0:
                min_length = min(len(plan1_inflated), len(plan2_inflated))
                combined_inflated = [plan1_inflated[i] + plan2_inflated[i] for i in range(min_length)]
            else:
                combined_inflated = None
            
            # Store combined premiums for both tabs
            st.session_state['premium_data'] = {
                'premiums': combined_premiums,
                'inflated_premiums': combined_inflated,
                'current_age': current_age,
                'inflation_rate': inflation_rate
            }
        else:
            # Store only plan 1 premiums
            st.session_state['premium_data'] = {
                'premiums': plan1_premiums,
                'inflated_premiums': plan1_inflated if inflation_rate > 0 else None,
                'current_age': current_age,
                'inflation_rate': inflation_rate
            }

        # Store first year premium for display
        if plan1_premiums:
            st.session_state.medical_premium = plan1_premiums[0]  # Store the first year's premium
            if plan2 != "None" and plan2_premiums:
                st.session_state.medical_premium += plan2_premiums[0]  # Add second plan's premium if exists

        # Create three columns for results
        col1, col2, col3 = st.columns(3)

        # Column 1: Plan 1
        with col1:
            st.markdown(f'<div class="fixed-height-header"><h2>Plan 1: {plan1}</h2></div>', unsafe_allow_html=True)
            
            # Premium Growth Projection
            st.subheader("醫療保費預計增長情況")
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=list(range(current_age, max_age)),
                y=plan1_premiums,
                name=f"Original Premium ({plan1})",
                line=dict(color='blue')
            ))
            
            if inflation_rate > 0:
                fig1.add_trace(go.Scatter(
                    x=list(range(current_age, max_age)),
                    y=plan1_inflated,
                    name=f"Inflated Premium ({plan1})",
                    line=dict(color='red')
                ))
            
            fig1.update_layout(
                xaxis_title="Age 年齡",
                yaxis_title=f"Premium {currency}",
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Total Premium Calculations
            st.subheader("總保費")
            
            def create_premium_table(premiums, inflated_premiums=None):
                """Create a DataFrame for premium totals"""
                age_ranges = [
                    f"{current_age} to 65",
                    f"{current_age} to 75",
                    f"{current_age} to 85",
                    f"{current_age} to 99"
                ]
                
                data = {
                    'Age Range': age_ranges,
                    f'Current Premium ({currency.strip()})': [
                        f"{calculate_total_premium_to_age(premiums, current_age, 65):,.0f}",
                        f"{calculate_total_premium_to_age(premiums, current_age, 75):,.0f}",
                        f"{calculate_total_premium_to_age(premiums, current_age, 85):,.0f}",
                        f"{calculate_total_premium_to_age(premiums, current_age, 99):,.0f}"
                    ]
                }
                
                if inflated_premiums is not None:
                    data[f'Projected Premium ({currency.strip()}) ({inflation_rate}% inflation)'] = [
                        f"{calculate_total_premium_to_age(inflated_premiums, current_age, 65):,.0f}",
                        f"{calculate_total_premium_to_age(inflated_premiums, current_age, 75):,.0f}",
                        f"{calculate_total_premium_to_age(inflated_premiums, current_age, 85):,.0f}",
                        f"{calculate_total_premium_to_age(inflated_premiums, current_age, 99):,.0f}"
                    ]
                
                return pd.DataFrame(data)
            
            # Plan 1 Premium Table
            df_plan1 = create_premium_table(plan1_premiums, plan1_inflated if inflation_rate > 0 else None)
            st.dataframe(df_plan1, use_container_width=True, hide_index=True)

            # Premium Table
            with st.expander("保費表", expanded=True):
                premium_table = pd.DataFrame({
                    'Age': [str(age) for age in range(current_age, max_age)],
                    f'Original ({currency.strip()})': [f"{p:,.0f}" for p in plan1_premiums]
                })
                if inflation_rate > 0:
                    premium_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in plan1_inflated]
                st.dataframe(premium_table, use_container_width=True, hide_index=True)

            # 5-Year Average Premiums
            with st.expander("每5年平均保費", expanded=True):
                avg_premiums = calculate_5year_averages(plan1_premiums)
                age_ranges = [f"{i}-{i+4}" for i in range(current_age+5, current_age+5+len(avg_premiums)*5, 5)]
                avg_table = pd.DataFrame({
                    'Age Range': age_ranges,
                    f'Original ({currency.strip()})': [f"{p:,.0f}" for p in avg_premiums]
                })
                if inflation_rate > 0:
                    avg_inflated = calculate_5year_averages(plan1_inflated)
                    avg_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in avg_inflated]
                st.dataframe(avg_table, use_container_width=True, hide_index=True)

        # Column 2: Plan 2 (if selected)
        with col2:
            if plan2 != "None":
                st.markdown(f'<div class="fixed-height-header"><h2>Plan 2: {plan2}</h2></div>', unsafe_allow_html=True)
                
                # Premium Growth Projection
                st.subheader("醫療保費預計增長情況")
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=list(range(current_age, max_age)),
                    y=plan2_premiums,
                    name="Original",
                    line=dict(color='green')
                ))
                
                if inflation_rate > 0:
                    fig2.add_trace(go.Scatter(
                        x=list(range(current_age, max_age)),
                        y=plan2_inflated,
                        name="Inflated",
                        line=dict(color='orange')
                    ))
                
                fig2.update_layout(
                    xaxis_title="Age 年齡",
                    yaxis_title=f"Premium {currency}",
                    showlegend=True
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Total Premium Calculations for Plan 2
                st.subheader("總保費")
                df_plan2 = create_premium_table(plan2_premiums, plan2_inflated if inflation_rate > 0 else None)
                st.dataframe(df_plan2, use_container_width=True, hide_index=True)

                # Premium Table
                with st.expander("保費表", expanded=True):
                    premium_table = pd.DataFrame({
                        'Age': [str(age) for age in range(current_age, max_age)],
                        f'Original ({currency.strip()})': [f"{p:,.0f}" for p in plan2_premiums]
                    })
                    if inflation_rate > 0:
                        premium_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in plan2_inflated]
                    st.dataframe(premium_table, use_container_width=True, hide_index=True)

                # 5-Year Average Premiums
                with st.expander("每5年平均保費", expanded=True):
                    avg_premiums = calculate_5year_averages(plan2_premiums)
                    age_ranges = [f"{i}-{i+4}" for i in range(current_age+5, current_age+5+len(avg_premiums)*5, 5)]
                    avg_table = pd.DataFrame({
                        'Age Range': age_ranges,
                        f'Original ({currency.strip()})': [f"{p:,.0f}" for p in avg_premiums]
                    })
                    if inflation_rate > 0:
                        avg_inflated = calculate_5year_averages(plan2_inflated)
                        avg_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in avg_inflated]
                    st.dataframe(avg_table, use_container_width=True, hide_index=True)

        # Column 3: Combined (if Plan 2 is selected)
        with col3:
            if plan2 != "None":
                st.markdown('<div class="fixed-height-header combined"><h2>Combined Total</h2></div>', unsafe_allow_html=True)
                
                # Combined Premium Growth Projection
                st.subheader("醫療保費預計增長情況")
                fig_combined = go.Figure()
                
                # Calculate combined premiums
                combined_premiums = []
                combined_inflated = []
                
                # Adjust lengths to match
                min_length = min(len(plan1_premiums), len(plan2_premiums))
                plan1_premiums_adj = plan1_premiums[:min_length]
                plan2_premiums_adj = plan2_premiums[:min_length]
                
                # Calculate combined premiums
                for p1, p2 in zip(plan1_premiums_adj, plan2_premiums_adj):
                    combined_premiums.append(round(p1 + p2))
                
                # Update ages array to match the length
                ages_combined = list(range(current_age, current_age + min_length))
                
                fig_combined.add_trace(go.Scatter(
                    x=ages_combined,
                    y=combined_premiums,
                    name="Original",
                    line=dict(color='purple')
                ))
                
                if inflation_rate > 0:
                    # Calculate combined inflated premiums
                    plan1_inflated_adj = plan1_inflated[:min_length]
                    plan2_inflated_adj = plan2_inflated[:min_length]
                    
                    for p1, p2 in zip(plan1_inflated_adj, plan2_inflated_adj):
                        combined_inflated.append(round(p1 + p2))
                    
                    fig_combined.add_trace(go.Scatter(
                        x=ages_combined,
                        y=combined_inflated,
                        name="Inflated",
                        line=dict(color='red')
                    ))
                
                fig_combined.update_layout(
                    xaxis_title="Age 年齡",
                    yaxis_title=f"Premium {currency}",
                    showlegend=True
                )
                st.plotly_chart(fig_combined, use_container_width=True)

                # Total Premium Calculations for Combined
                st.subheader("總保費")
                df_combined = create_premium_table(combined_premiums, combined_inflated if inflation_rate > 0 else None)
                st.dataframe(df_combined, use_container_width=True, hide_index=True)

                # Combined Premium Table
                with st.expander("保費表", expanded=True):
                    combined_table = pd.DataFrame({
                        'Age': [str(age) for age in ages_combined],
                        f'Original ({currency.strip()})': [f"{p:,.0f}" for p in combined_premiums]
                    })
                    if inflation_rate > 0:
                        combined_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in combined_inflated]
                    st.dataframe(combined_table, use_container_width=True, hide_index=True)

                # Combined 5-Year Average Premiums
                with st.expander("每5年平均保費", expanded=True):
                    avg_premiums = calculate_5year_averages(combined_premiums)
                    age_ranges = [f"{i}-{i+4}" for i in range(current_age+5, current_age+5+len(avg_premiums)*5, 5)]
                    avg_table = pd.DataFrame({
                        'Age Range': age_ranges,
                        f'Original ({currency.strip()})': [f"{p:,.0f}" for p in avg_premiums]
                    })
                    if inflation_rate > 0:
                        avg_inflated = calculate_5year_averages(combined_inflated)
                        avg_table[f'Inflated ({currency.strip()})'] = [f"{p:,.0f}" for p in avg_inflated]
                    st.dataframe(avg_table, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Medical Financing ")
        st.write("""
        This calculator demonstrates how early savings can help offset future medical premiums through compound interest.
        The plan involves saving for 5 years, followed by interest accrual and premium withdrawals.
        """)
        
        # Savings inputs
        st.markdown("""
        <style>
        div[data-testid="stNumberInput"] > div > div > div > input {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Set default saving amount based on currency
        default_saving = 10000 if st.session_state['currency'] == "USD " else 80000
        
        saving_col1, saving_col2 = st.columns(2)
        with saving_col1:
            annual_saving = st.number_input(
                "Annual Saving for First 5 Years",
                min_value=0,
                value=default_saving,
                step=1000,
                key='annual_saving'
            )
            
            st.write(f"Total Savings after 5 years: {st.session_state['currency'].split()[0]} {annual_saving * 5:,.0f}")
            
            withdrawal_at_65 = st.number_input(
                "Once off withdrawal at 65",
                min_value=0,
                value=0,
                step=1000,
                key='withdrawal_65'
            )
        
        with saving_col2:
            st.info("""
            **Plan Structure:**
            - Save for 5 years
            - 5% annual compound interest from Year 6
            - Annual withdrawal for premium payment from Year 6
            """)
        
        # Determine which premiums to use
        if 'premium_data' in st.session_state:
            premiums_to_use = st.session_state['premium_data']['premiums']
            inflated_premiums = st.session_state['premium_data']['inflated_premiums']
            current_age = st.session_state['premium_data']['current_age']
            inflation_rate = st.session_state['premium_data']['inflation_rate']
        else:
            st.error("Error: Premium data not found in session state.")
            return

        # Calculate savings plan
        total_savings = annual_saving * 5
        
        # Calculate minimum annual savings needed
        min_savings = calculate_minimum_savings(current_age, premiums_to_use, inflated_premiums, 
                                             withdrawal_at_65=withdrawal_at_65)
        
        # Calculate savings plan
        savings_results, insufficient, insufficient_age, insufficient_amount = calculate_savings_plan(annual_saving, current_age, premiums_to_use, 
                                                               inflated_premiums,
                                                               withdrawal_at_65=withdrawal_at_65)
        
        # Display warning if savings are insufficient
        if insufficient:
            # Calculate minimum required savings
            min_required_savings = calculate_minimum_savings(
                current_age,
                premiums_to_use,
                inflated_premiums,
                withdrawal_at_65=withdrawal_at_65
            )
            
            st.error(f"""
            Warning: Insufficient Savings
            
            The proposed saving amount of {st.session_state['currency'].split()[0]} {annual_saving:,.0f} per year is not sufficient 
            to cover the entire medical plan.
            
            Minimum required annual savings: {st.session_state['currency'].split()[0]} {min_required_savings:,.0f}
            Additional savings needed: {st.session_state['currency'].split()[0]} {(min_required_savings - annual_saving):,.0f} per year
            Insufficient at age: {insufficient_age}
            Amount that couldn't be fully withdrawn: {st.session_state['currency'].split()[0]} {insufficient_amount:,.0f}
            """)
        
        # Calculate key metrics
        total_savings = annual_saving * 5
        total_interest = sum([float(result['Interest']) for result in savings_results])
        total_premiums = sum([float(result['Premium']) for result in savings_results])
        final_balance = float(savings_results[-1]['End'])
        
        # Display summary metrics
        st.subheader("Summary")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Savings ", 
                     f"{st.session_state['currency'].split()[0]} {total_savings:,.0f}")
        
        with metric_col2:
            st.metric("Total Interest ",
                     f"{st.session_state['currency'].split()[0]} {total_interest:,.0f}")
        
        with metric_col3:
            st.metric("Total Withdrawal for Medical ",
                     f"{st.session_state['currency'].split()[0]} {total_premiums:,.0f}")
        
        with metric_col4:
            st.metric("Final Balance ",
                     f"{st.session_state['currency'].split()[0]} {final_balance:,.0f}")

        # Create visualization of savings plan
        st.subheader("Savings Plan Visualization")
        
        # Prepare data for visualization
        fig = go.Figure()
        
        # Add Annual Savings bars (green)
        fig.add_trace(go.Bar(
            x=[str(age) for age in range(current_age, current_age + len(savings_results))],
            y=[float(s.replace(st.session_state['currency'].split()[0], '').replace(',', '')) for s in [f"{st.session_state['currency'].split()[0]} {result['Saving']:,.0f}" for result in savings_results]],
            name='Annual Savings',
            marker_color='rgba(75, 192, 75, 0.7)',  # Green
            hovertemplate=st.session_state['currency'] + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
        ))
        
        # Add Interest Earned bars (blue)
        fig.add_trace(go.Bar(
            x=[str(age) for age in range(current_age, current_age + len(savings_results))],
            y=[float(i.replace(st.session_state['currency'].split()[0], '').replace(',', '')) for i in [f"{st.session_state['currency'].split()[0]} {result['Interest']:,.0f}" for result in savings_results]],
            name='Interest Earned',
            marker_color='rgba(66, 133, 244, 0.7)',  # Blue
            hovertemplate=st.session_state['currency'] + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
        ))
        
        # Add Premium Withdrawal bars (red)
        fig.add_trace(go.Bar(
            x=[str(age) for age in range(current_age, current_age + len(savings_results))],
            y=[float(p.replace(st.session_state['currency'].split()[0], '').replace(',', '')) for p in [f"{st.session_state['currency'].split()[0]} {result['Premium']:,.0f}" for result in savings_results]],
            name='Withdrawal for Medical Premium',
            marker_color='rgba(234, 67, 53, 0.7)',  # Red
            hovertemplate=st.session_state['currency'] + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
        ))
        
        # Add Age 65 Withdrawal bars (purple)
        fig.add_trace(go.Bar(
            x=[str(age) for age in range(current_age, current_age + len(savings_results))],
            y=[float(a.replace(st.session_state['currency'].split()[0], '').replace(',', '')) for a in [f"{st.session_state['currency'].split()[0]} {result['Age65']:,.0f}" for result in savings_results]],
            name='Age 65 Withdrawal',
            marker_color='rgba(156, 39, 176, 0.7)',  # Purple
            hovertemplate=st.session_state['currency'] + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
        ))
        
        # Add Balance line (black) on secondary y-axis
        fig.add_trace(go.Scatter(
            x=[str(age) for age in range(current_age, current_age + len(savings_results))],
            y=[float(b.replace(st.session_state['currency'].split()[0], '').replace(',', '')) for b in [f"{st.session_state['currency'].split()[0]} {result['End']:,.0f}" for result in savings_results]],
            name='Balance',
            line=dict(color='rgba(0, 0, 0, 1)', width=3),  # Black
            yaxis='y2',
            hovertemplate=st.session_state['currency'] + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
        ))
        
        # Update layout with dual y-axis
        fig.update_layout(
            title={
                'text': 'Savings Plan Over Time',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Age',
            yaxis_title=f'Amount ({st.session_state["currency"]})',
            yaxis2=dict(
                title=f'Balance ({st.session_state["currency"]})',
                overlaying='y',
                side='right'
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            height=500,
            margin=dict(l=60, r=60, t=50, b=50)
        )
        
        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_width=1, line_color="black", line_dash="dash")
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Create detailed breakdown table
        df_savings = pd.DataFrame(savings_results)
        
        # Convert Age to strings to ensure left alignment
        df_savings['Age'] = df_savings['Age'].astype(str)
        
        # Format currency values for display
        for col in ['Start', 'Interest', 'Saving', 'Premium', 'Age65', 'End']:
            df_savings[col] = df_savings[col].apply(lambda x: f"{st.session_state['currency'].split()[0]} {x:,.0f}")
        
        # Add CSS for table alignment
        st.markdown("""
        <style>
        div[data-testid="stDataFrame"] div[data-testid="stTable"] {
            text-align: left !important;
        }
        div[data-testid="stDataFrame"] td, 
        div[data-testid="stDataFrame"] th {
            text-align: left !important;
            white-space: nowrap !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display results
        st.subheader("Annual Breakdown")
        st.dataframe(
            df_savings,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Age': st.column_config.TextColumn('Age ', help="Current age"),
                'Start': st.column_config.TextColumn('Start ', help="Year-start balance"),
                'Interest': st.column_config.TextColumn('Interest ', help="Interest earned during the year"),
                'Saving': st.column_config.TextColumn('Saving ', help="Amount saved during the year"),
                'Premium': st.column_config.TextColumn('Withdrawal for Medical Premium ', help="Premium paid during the year"),
                'Age65': st.column_config.TextColumn('Age 65 Withdrawal ', help="One-time withdrawal at age 65"),
                'End': st.column_config.TextColumn('End ', help="Year-end balance")
            }
        )

    with tab3:
        display_saving_plan()

if __name__ == "__main__":
    main()
