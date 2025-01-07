import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import io

# Constants for age milestones
AGE_MILESTONES = [65, 75, 85, 99]

# Set page config
st.set_page_config(
    page_title="Medical Premium Calculator",
    page_icon="",
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

# Calculate total premium to age
def calculate_total_premium_to_age(premiums, current_age, target_age):
    """Calculate total premium from current age to target age"""
    if target_age <= current_age:
        return 0
    end_index = min(target_age - current_age, len(premiums))
    return sum(premiums[:end_index])

# Main application
def main():
    st.title("Medical Premium Calculator ")
    
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
    tab1, tab2 = st.tabs(["Premium Calculator ", "Medical Financing "])
    
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
                
                target_ages = [65, 75, 85, 99]
                current_totals = []
                projected_totals = []
                
                for target_age in target_ages:
                    years = target_age - current_age if target_age > current_age else 0
                    current_total = sum(premiums[:years])
                    current_totals.append(current_total)
                    
                    if inflated_premiums:
                        projected_total = sum(inflated_premiums[:years])
                        projected_totals.append(projected_total)
                
                data = {
                    'Age Range': age_ranges,
                    'Current Premium': [f"{currency.split()[0]} {total:,.0f}" for total in current_totals]
                }
                
                if inflated_premiums:
                    data['Projected Premium ({}% Inflation)'.format(inflation_rate)] = [
                        f"{currency.split()[0]} {total:,.0f}" for total in projected_totals
                    ]
                
                return pd.DataFrame(data)
            
            # Plan 1 Premium Table
            df_plan1 = create_premium_table(plan1_premiums, plan1_inflated if inflation_rate > 0 else None)
            st.dataframe(df_plan1, use_container_width=True, hide_index=True)

            # Premium Table
            with st.expander("保費表", expanded=True):
                premium_table = pd.DataFrame({
                    'Age': [str(age) for age in range(current_age, max_age)],
                    'Original': [f"{currency.split()[0]} {p:,.0f}" for p in plan1_premiums]
                })
                if inflation_rate > 0:
                    premium_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in plan1_inflated]
                st.dataframe(premium_table, use_container_width=True, hide_index=True)

            # 5-Year Average Premiums
            with st.expander("每5年平均保費", expanded=True):
                avg_premiums = calculate_5year_averages(plan1_premiums)
                age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
                avg_table = pd.DataFrame({
                    'Age Range': age_ranges,
                    'Original': [f"{currency.split()[0]} {p:,.0f}" for p in avg_premiums]
                })
                if inflation_rate > 0:
                    avg_inflated = calculate_5year_averages(plan1_inflated)
                    avg_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in avg_inflated]
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
                        'Original': [f"{currency.split()[0]} {p:,.0f}" for p in plan2_premiums]
                    })
                    if inflation_rate > 0:
                        premium_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in plan2_inflated]
                    st.dataframe(premium_table, use_container_width=True, hide_index=True)

                # 5-Year Average Premiums
                with st.expander("每5年平均保費", expanded=True):
                    avg_premiums = calculate_5year_averages(plan2_premiums)
                    age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
                    avg_table = pd.DataFrame({
                        'Age Range': age_ranges,
                        'Original': [f"{currency.split()[0]} {p:,.0f}" for p in avg_premiums]
                    })
                    if inflation_rate > 0:
                        avg_inflated = calculate_5year_averages(plan2_inflated)
                        avg_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in avg_inflated]
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
                        'Original': [f"{currency.split()[0]} {p:,.0f}" for p in combined_premiums]
                    })
                    if inflation_rate > 0:
                        combined_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in combined_inflated]
                    st.dataframe(combined_table, use_container_width=True, hide_index=True)

                # Combined 5-Year Average Premiums
                with st.expander("每5年平均保費", expanded=True):
                    avg_premiums = calculate_5year_averages(combined_premiums)
                    age_ranges = [f"{i}-{i+4}" for i in range(current_age+6, current_age+6+len(avg_premiums)*5, 5)]
                    avg_table = pd.DataFrame({
                        'Age Range': age_ranges,
                        'Original': [f"{currency.split()[0]} {p:,.0f}" for p in avg_premiums]
                    })
                    if inflation_rate > 0:
                        avg_inflated = calculate_5year_averages(combined_inflated)
                        avg_table['Inflated'] = [f"{currency.split()[0]} {p:,.0f}" for p in avg_inflated]
                    st.dataframe(avg_table, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Medical Financing ")
        st.write("""
        This calculator demonstrates how early savings can help offset future medical premiums through compound interest.
        The plan involves saving for 5 years, followed by interest accrual and premium withdrawals.
        """)
        
        # Saving inputs
        saving_col1, saving_col2 = st.columns(2)
        
        with saving_col1:
            # Set default saving amount based on currency
            default_saving = 10000 if currency == "USD " else 80000
            annual_saving = st.number_input(
                "Annual Saving Amount ",
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
                Warning: Insufficient Savings
                
                The proposed saving amount of {currency.split()[0]} {annual_saving:,.0f} per year is not sufficient 
                to cover the entire medical plan.
                
                Minimum required annual savings: {currency.split()[0]} {min_required_savings:,.0f}
                Additional savings needed: {currency.split()[0]} {(min_required_savings - annual_saving):,.0f} per year
                """)
            
            # Calculate key metrics
            total_savings = annual_saving * 5
            total_interest = sum([float(result['Interest']) for result in savings_results])
            total_premiums = sum([float(result['Premium']) for result in savings_results])
            final_balance = float(savings_results[-1]['Balance'])
            
            # Display summary metrics
            st.subheader("Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Savings ", 
                         f"{currency.split()[0]} {total_savings:,.0f}")
            
            with metric_col2:
                st.metric("Total Interest ",
                         f"{currency.split()[0]} {total_interest:,.0f}")
            
            with metric_col3:
                st.metric("Total Premiums ",
                         f"{currency.split()[0]} {total_premiums:,.0f}")
            
            with metric_col4:
                st.metric("Final Balance ",
                         f"{currency.split()[0]} {final_balance:,.0f}")

            # Create visualization of savings plan
            st.subheader("Savings Plan Visualization")
            
            # Prepare data for visualization
            fig = go.Figure()
            
            # Add Annual Savings bars (green)
            fig.add_trace(go.Bar(
                x=[str(age) for age in range(current_age, current_age + len(savings_results))],
                y=[float(s.replace(currency.split()[0], '').replace(',', '')) for s in [f"{currency.split()[0]} {result['Savings']:,.0f}" for result in savings_results]],
                name='Annual Savings',
                marker_color='rgba(75, 192, 75, 0.7)',  # Green
                hovertemplate=currency + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
            ))
            
            # Add Interest Earned bars (blue)
            fig.add_trace(go.Bar(
                x=[str(age) for age in range(current_age, current_age + len(savings_results))],
                y=[float(i.replace(currency.split()[0], '').replace(',', '')) for i in [f"{currency.split()[0]} {result['Interest']:,.0f}" for result in savings_results]],
                name='Interest Earned',
                marker_color='rgba(66, 133, 244, 0.7)',  # Blue
                hovertemplate=currency + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
            ))
            
            # Add Premium Withdrawal bars (red)
            fig.add_trace(go.Bar(
                x=[str(age) for age in range(current_age, current_age + len(savings_results))],
                y=[float(p.replace(currency.split()[0], '').replace(',', '')) for p in [f"{currency.split()[0]} {result['Premium']:,.0f}" for result in savings_results]],
                name='Premium Paid',
                marker_color='rgba(234, 67, 53, 0.7)',  # Red
                hovertemplate=currency + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
            ))
            
            # Add Balance line (purple) on secondary y-axis
            fig.add_trace(go.Scatter(
                x=[str(age) for age in range(current_age, current_age + len(savings_results))],
                y=[float(b.replace(currency.split()[0], '').replace(',', '')) for b in [f"{currency.split()[0]} {result['Balance']:,.0f}" for result in savings_results]],
                name='Balance',
                line=dict(color='rgba(156, 39, 176, 1)', width=3),  # Purple
                yaxis='y2',
                hovertemplate=currency + ' %{y:,.0f}<br>Age: %{x}<extra></extra>'
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
                yaxis_title=f'Amount ({currency})',
                yaxis2=dict(
                    title=f'Balance ({currency})',
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
            
            # Convert Age and Year to strings to ensure left alignment
            df_savings['Age'] = df_savings['Age'].astype(str)
            df_savings['Year'] = df_savings['Year'].astype(str)
            
            # Format currency values for display
            for col in ['Savings', 'Interest', 'Premium', 'Balance']:
                df_savings[col] = df_savings[col].apply(lambda x: f"{currency.split()[0]} {x:,.0f}")
            
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
                    'Year': st.column_config.TextColumn('Year ', help="Calendar year"),
                    'Savings': st.column_config.TextColumn('Savings ', help="Amount saved during the year"),
                    'Interest': st.column_config.TextColumn('Interest ', help="Interest earned during the year"),
                    'Premium': st.column_config.TextColumn('Premium ', help="Premium paid during the year"),
                    'Balance': st.column_config.TextColumn('Balance ', help="Year-end balance")
                }
            )

if __name__ == "__main__":
    main()
