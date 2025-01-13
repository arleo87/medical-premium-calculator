import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Clear function caches
@st.cache_data(ttl=0)  # Set TTL to 0 to prevent caching
def calculate_saving_plan(saving_amount=10000, withdrawal_start_year=6, withdrawal_start_age=None, fixed_withdrawal_percentage=None):  
    # Read the logic CSV file for bonus calculations
    logic_df = pd.read_csv('logic.csv')
    
    # Clean up column names by removing extra spaces
    logic_df.columns = logic_df.columns.str.strip()
    
    # Convert percentage strings to float values
    logic_df['SB_Reduction'] = logic_df['SB_Reduction'].apply(lambda x: float(str(x).strip('%')) / 100 if isinstance(x, str) else x)
    
    # Get premium data from session state
    if 'premium_data' not in st.session_state:
        st.error("Error: Premium data not found. Please select plans in the Premium Calculator tab first.")
        return
        
    premium_data = st.session_state['premium_data']
    current_age = premium_data['current_age']
    
    # Always use USD for premiums
    exchange_rate = st.session_state.get('exchange_rate', 7.85)  # Default to 7.85 if not set
    
    # Get premiums and convert to USD if needed
    if st.session_state.get('currency', "USD ") == "HKD ":
        # Convert to USD and prepare premiums array
        raw_premiums = [p / exchange_rate for p in premium_data['premiums']]
        premiums_to_use = raw_premiums  # Will use index offset when withdrawing
        
        if premium_data['inflation_rate'] > 0 and premium_data['inflated_premiums'] is not None:
            raw_inflated = [p / exchange_rate for p in premium_data['inflated_premiums']]
            premiums_to_use = raw_inflated  # Will use index offset when withdrawing
    else:
        # Prepare premiums array
        premiums_to_use = premium_data['premiums']  # Will use index offset when withdrawing
        
        if premium_data['inflation_rate'] > 0 and premium_data['inflated_premiums'] is not None:
            premiums_to_use = premium_data['inflated_premiums']  # Will use index offset when withdrawing

    # Initialize lists to store values
    start_age = current_age
    end_age = 99  # End age
    projection_years = end_age - start_age + 1
    years = list(range(1, projection_years + 1))
    ages = list(range(start_age, end_age + 1))
    
    total_savings = []
    withdrawals = []
    remaining_withdrawals = []  # For debugging
    sb_deductions = []  # For debugging special bonus deductions
    notional_amounts = []
    guaranteed_cash_values = []
    reversionary_bonuses = []
    special_bonuses = []
    surrender_values = []
    gcv_withdrawals = []  # For debugging GCV withdrawals
    
    # Initialize variables
    total_saving = 0
    rev_bonus_balance = 0
    special_bonus_balance = 0
    notional_amount = saving_amount
    insufficient = False
    insufficient_age = None
    insufficient_amount = None
    notional_amount_too_low = False
    notional_amount_too_low_age = None
    notional_amount_too_low_value = None
    
    for year, age in zip(years, ages):
        # For first year, notional amount is the saving amount
        if year == 1:
            notional_amount = saving_amount
            notional_amounts.append(notional_amount)
            gcv = notional_amount * logic_df.loc[year-1, 'GCV']
            guaranteed_cash_values.append(gcv)
            withdrawals.append(0)
            remaining_withdrawals.append(0)
            sb_deductions.append(0)
            reversionary_bonuses.append(0)
            special_bonuses.append(0)
            gcv_withdrawals.append(0)
            total_saving = saving_amount
            total_savings.append(total_saving)
            surrender_value = gcv  # First year surrender value is just GCV
            surrender_values.append(surrender_value)
            continue
        
        # Calculate Total Savings (only for first 5 years)
        if year <= 5:
            total_saving += saving_amount
        total_savings.append(total_saving)
        
        # Calculate beginning reversionary bonus for this year
        if year >= 3:
            # Calculate new reversionary bonus based on previous year's values
            new_rev_bonus = (notional_amounts[-1] * 0.12)  # 12% of previous notional amount
            new_rev_bonus += (rev_bonus_balance * 0.008)  # 0.8% of previous reversionary bonus
            rev_bonus_before_withdrawal = rev_bonus_balance + new_rev_bonus
        else:
            rev_bonus_before_withdrawal = 0
        
        # Calculate withdrawal (start from specified year)
        withdrawal = 0
        remaining_withdrawal = 0  # Initialize remaining withdrawal
        sb_deduction = 0  # Initialize special bonus deduction
        gcv_withdrawal = 0  # Initialize GCV withdrawal
        
        # Calculate total premium paid (5 years of savings)
        total_premium_paid = saving_amount * 5
        
        # Determine if withdrawal should start based on year or age
        should_withdraw = False
        if withdrawal_start_age is not None:
            should_withdraw = age >= withdrawal_start_age
        else:
            should_withdraw = year >= withdrawal_start_year
            
        if should_withdraw:
            if fixed_withdrawal_percentage is not None:
                # Fixed percentage withdrawal
                withdrawal = total_premium_paid * fixed_withdrawal_percentage
            else:
                # Medical premium based withdrawal
                policy_year = year - 1
                if policy_year < len(premiums_to_use):
                    withdrawal = premiums_to_use[policy_year]
            
            # Calculate total available value before withdrawal
            special_bonus_before_withdrawal = special_bonus_balance
            # Calculate GCV using previous year's notional amount
            gcv_before_withdrawal = notional_amounts[-1] * logic_df.loc[min(year-1, len(logic_df)-1), 'GCV']
            total_available = rev_bonus_before_withdrawal + special_bonus_before_withdrawal + gcv_before_withdrawal
            
            # Check if withdrawal would exceed available value
            if withdrawal > total_available:
                if not insufficient:
                    insufficient = True
                    insufficient_age = age
                    insufficient_amount = withdrawal - total_available
                withdrawal = total_available  # Adjust withdrawal to available amount
            
            # Process withdrawal in order: reversionary bonus -> special bonus -> GCV
            remaining_withdrawal = withdrawal
            
            # 1. Deduct from reversionary bonus first
            if remaining_withdrawal > 0 and rev_bonus_before_withdrawal > 0:
                rev_bonus_deduction = min(remaining_withdrawal, rev_bonus_before_withdrawal)
                rev_bonus_balance = rev_bonus_before_withdrawal - rev_bonus_deduction
                remaining_withdrawal -= rev_bonus_deduction
            else:
                rev_bonus_balance = rev_bonus_before_withdrawal
            
            # Store remaining withdrawal for debugging
            remaining_withdrawals.append(remaining_withdrawal)
            
            # 2. Deduct from special bonus if needed
            if remaining_withdrawal > 0 and special_bonus_before_withdrawal > 0:
                sb_reduction_ratio = logic_df.loc[min(year-1, len(logic_df)-1), 'SB_Reduction']
                # Calculate special bonus deduction by multiplying remaining withdrawal with ratio
                sb_deduction = remaining_withdrawal * sb_reduction_ratio
                
                # Check if we have enough special bonus to cover this deduction
                if sb_deduction <= special_bonus_before_withdrawal:
                    special_bonus_balance = special_bonus_before_withdrawal - sb_deduction
                    remaining_withdrawal = remaining_withdrawal * (1 - sb_reduction_ratio)  # Calculate actual remaining withdrawal
                else:
                    # If not enough special bonus, deduct what we can
                    sb_deduction = special_bonus_before_withdrawal
                    special_bonus_balance = 0
                    remaining_withdrawal = remaining_withdrawal * (1 - sb_reduction_ratio)  # Calculate actual remaining withdrawal
            
            # 3. Calculate Guaranteed Cash Value and handle remaining withdrawal
            gcv_ratio = logic_df.loc[min(year-1, len(logic_df)-1), 'GCV']
            # Calculate GCV using previous year's notional amount
            gcv = notional_amounts[-1] * gcv_ratio
            
            # Deduct remaining withdrawal from GCV
            if remaining_withdrawal > 0:
                gcv_withdrawal = remaining_withdrawal
                gcv = max(0, gcv - remaining_withdrawal)
            
            # Update notional amount based on final GCV
            notional_amount = gcv / gcv_ratio if gcv_ratio > 0 else 0
            
        else:
            # If no withdrawal, reversionary bonus balance is the before withdrawal amount
            rev_bonus_balance = rev_bonus_before_withdrawal
            remaining_withdrawals.append(0)  # No remaining withdrawal
            sb_deduction = 0
            gcv_withdrawal = 0
            
            # Calculate GCV with no withdrawal
            gcv_ratio = logic_df.loc[min(year-1, len(logic_df)-1), 'GCV']
            gcv = notional_amounts[-1] * gcv_ratio  # Use previous year's notional amount
            notional_amount = notional_amounts[-1]  # Keep same notional amount if no withdrawal
        
        # Check if notional amount is below minimum
        if notional_amount < 3000 and not notional_amount_too_low:
            notional_amount_too_low = True
            notional_amount_too_low_age = age
            notional_amount_too_low_value = notional_amount
            
        withdrawals.append(withdrawal)
        notional_amounts.append(notional_amount)
        reversionary_bonuses.append(rev_bonus_balance)
        sb_deductions.append(sb_deduction)  # Store special bonus deduction for debugging
        guaranteed_cash_values.append(gcv)
        gcv_withdrawals.append(gcv_withdrawal)  # Store GCV withdrawal for debugging
        
        # Calculate Special Bonus
        if year >= 3:
            sb_ratio = logic_df.loc[min(year-1, len(logic_df)-1), 'SB']
            new_special_bonus = notional_amounts[-2] * sb_ratio
            special_bonus_balance += new_special_bonus
        special_bonuses.append(special_bonus_balance)
        
        # Calculate Surrender Value
        surrender_value = (guaranteed_cash_values[-1] + 
                         reversionary_bonuses[-1] + 
                         special_bonuses[-1])
        surrender_values.append(surrender_value)
    
    # Create DataFrame for saving plan
    savings_list = [saving_amount if year <= 5 else 0 for year in years]  # Only save for first 5 years
    
    saving_plan_df = pd.DataFrame({
        'Policy Year': range(1, len(ages) + 1),  # Add policy year starting from 1
        'Age': ages,
        'Annual Saving (USD)': savings_list,
        'Total Savings (USD)': total_savings,
        'Withdrawal for Medical Premium (USD)': withdrawals,
        'Notional Amount (USD)': notional_amounts,
        'Guaranteed Cash Value (USD)': guaranteed_cash_values,
        'Reversionary Bonus (USD)': reversionary_bonuses,
        'Special Bonus (USD)': special_bonuses,
        'Surrender Value (USD)': surrender_values
    })
    
    # Round all numeric columns to 0 decimal places
    numeric_columns = saving_plan_df.select_dtypes(include=['float64', 'int64']).columns
    saving_plan_df[numeric_columns] = saving_plan_df[numeric_columns].round(0)
    
    return saving_plan_df, insufficient, insufficient_age, insufficient_amount, notional_amount_too_low, notional_amount_too_low_age, notional_amount_too_low_value

def display_saving_plan():
    st.header("Our Saving Plan")
    
    # Add withdrawal scenario selection
    withdrawal_scenarios = [
        {"name": "第6年起提款繳交醫保", "year": 6, "age": None, "percentage": None},
        {"name": "第11年起提款繳交醫保", "year": 11, "age": None, "percentage": None},
        {"name": "66歲起提款繳交醫保", "year": None, "age": 66, "percentage": None},
        {"name": "第6年起每年固定5%", "year": 6, "age": None, "percentage": 0.05},
        {"name": "第6年起每年固定7%", "year": 6, "age": None, "percentage": 0.07}
    ]
    
    selected_scenario = st.selectbox(
        "選擇提款方案",
        options=[s["name"] for s in withdrawal_scenarios],
        index=0
    )
    
    # Get selected scenario details
    selected_scenario_details = next(s for s in withdrawal_scenarios if s["name"] == selected_scenario)
    
    # Get saving amount input
    saving_amount = st.number_input(
        "Annual Saving for First 5 Years (USD)",
        min_value=0,
        value=10000,
        step=1000
    )
    
    # Calculate saving plan with selected withdrawal scenario
    saving_plan_df, insufficient, insufficient_age, insufficient_amount, notional_amount_too_low, notional_amount_too_low_age, notional_amount_too_low_value = calculate_saving_plan(
        saving_amount=saving_amount,
        withdrawal_start_year=selected_scenario_details["year"] if selected_scenario_details["year"] is not None else 999,
        withdrawal_start_age=selected_scenario_details["age"],
        fixed_withdrawal_percentage=selected_scenario_details["percentage"]
    )
    
    # Display warning if savings are insufficient
    if insufficient:
        st.warning(f"Warning: Savings will be insufficient at age {insufficient_age}. The surrender value will be {insufficient_amount:,.0f}, which is not enough to cover the withdrawal needed.")

    # Display warning if notional amount is too low
    if notional_amount_too_low:
        st.warning(f"Warning: The notional amount drops below 3,000 at age {notional_amount_too_low_age}. The notional amount will be {notional_amount_too_low_value:,.0f}. Consider increasing your annual saving amount.")

    # Create visualization
    st.subheader("Annual Saving Plan Breakdown (USD)")
    
    # Annual breakdown chart
    fig = go.Figure()
    
    # Add bars for annual savings only
    fig.add_trace(go.Bar(
        x=saving_plan_df['Age'],
        y=saving_plan_df['Annual Saving (USD)'],
        name='Annual Savings (USD)',
        marker_color='#2ecc71',  # Green
        showlegend=True,
        offsetgroup='savings',  # Separate group for savings
        hovertemplate='Age: %{x}<br>Amount: %{y:,.0f}<extra></extra>'
    ))
    
    # Calculate total annual bonus increment (reversionary + special)
    total_bonus = saving_plan_df['Reversionary Bonus (USD)'] + saving_plan_df['Special Bonus (USD)']
    annual_total_bonus = total_bonus.diff().fillna(total_bonus).round(0)
    
    # Add combined bonus increment
    fig.add_trace(go.Bar(
        name='Annual Total Bonus (USD)',
        x=saving_plan_df['Age'],
        y=annual_total_bonus,
        marker_color='#3498db',  # Blue
        showlegend=True,
        offsetgroup='bonus',  # Group for bonuses
        hovertemplate='Age: %{x}<br>Amount: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Medical Premium Withdrawal (USD)',
        x=saving_plan_df['Age'],
        y=saving_plan_df['Withdrawal for Medical Premium (USD)'],  
        marker_color='#e74c3c',  # Red
        showlegend=True,
        offsetgroup='withdrawal',  # Separate group for withdrawals
        hovertemplate='Age: %{x}<br>Amount: %{y:,.0f}<extra></extra>'
    ))
    
    # Add line for surrender value
    fig.add_trace(go.Scatter(
        name='Surrender Value (USD)',
        x=saving_plan_df['Age'],
        y=saving_plan_df['Surrender Value (USD)'],
        line=dict(color='#2c3e50', width=2),  # Dark blue
        yaxis='y2',
        showlegend=True,
        hovertemplate='Age: %{x}<br>Value: %{y:,.0f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        barmode='group',  # Group bars
        bargap=0.15,  # Gap between bars in a group
        bargroupgap=0.1,  # Gap between bar groups
        xaxis_title='Age',
        yaxis_title='Amount (USD)',
        yaxis2=dict(
            title='Balance (USD)',
            overlaying='y',
            side='right',
            tickformat=',d'  # Format y2-axis ticks without decimals
        ),
        yaxis=dict(
            tickformat=',d'  # Format y-axis ticks without decimals
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add vertical lines every 5 years
    min_age = min(saving_plan_df['Age'])
    max_age = max(saving_plan_df['Age'])
    
    # Calculate ages that are multiples of 5
    vertical_line_ages = list(range(min_age + (5 - min_age % 5), max_age + 1, 5))
    
    for age in vertical_line_ages:
        fig.add_vline(
            x=age,
            line_width=2,
            line_dash="solid",
            line_color="rgba(128, 128, 128, 0.8)",
        )

    st.plotly_chart(fig, use_container_width=True)

    # Display summary metrics
    st.subheader("Summary (USD)")
    
    # Calculate key metrics
    total_withdrawals = saving_plan_df['Withdrawal for Medical Premium (USD)'].sum()
    total_savings = saving_plan_df['Total Savings (USD)'].max()
    final_surrender_value = saving_plan_df['Surrender Value (USD)'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Savings (USD)", f"{total_savings:,.0f}")
    with col2:
        st.metric("Total Withdrawals (USD)", f"{total_withdrawals:,.0f}")
    with col3:
        st.metric("Final Surrender Value (USD)", f"{final_surrender_value:,.0f}")

    # Display annual breakdown table
    st.subheader("Annual Breakdown (USD)")
    
    # Format numeric columns
    formatted_df = saving_plan_df.copy()
    numeric_columns = [
        'Annual Saving (USD)', 
        'Withdrawal for Medical Premium (USD)', 
        'Total Savings (USD)',
        'Notional Amount (USD)',
        'Guaranteed Cash Value (USD)', 
        'Reversionary Bonus (USD)', 
        'Special Bonus (USD)',
        'Surrender Value (USD)'
    ]
    
    # Format numbers with USD and align left
    for col in numeric_columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"USD {x:,.0f}")
    
    # Set index first
    formatted_df = formatted_df.set_index('Policy Year')
    
    # Create a styled dataframe with left alignment for all cells
    styled_df = formatted_df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre',
        'padding-left': '10px'  # Add some padding for better readability
    })
    
    # Display the table with full width
    st.dataframe(styled_df, use_container_width=True)
