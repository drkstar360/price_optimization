import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds

# --- Configuration ---
CSV_FILE_PATH = 'pricing_matrix_30x30.csv'
CUTOFFS = np.array([2, 3, 4, 5, 6, 7, 14, 28])
N_DAYS = 30
N_CUTOFFS = len(CUTOFFS)
EPSILON_STRICT_INCREASE = 1e-7 # For strict inequality in discounts

# --- Helper Functions ---

def load_target_prices(file_path):
    """Loads the pricing matrix from CSV."""
    try:
        # Read CSV, assuming the first row is a header and the first column is an index/header
        df = pd.read_csv(file_path, header=0, index_col=0)
        
        # We expect a 30x30 matrix of actual prices.
        # If headers were present, df might be 30x30 already, or 31x31 then sliced.
        # The problem implies the *data* is 30x30.
        # header=0 means row 0 is header, index_col=0 means col 0 is index.
        # So df.values should give the numeric data.

        if df.shape == (N_DAYS, N_DAYS):
            prices = df.to_numpy(dtype=float)
        elif df.shape == (N_DAYS + 1, N_DAYS + 1) and file_path == CSV_FILE_PATH:
             print(f"Warning: CSV shape is {df.shape}. Assuming first row/col are headers and slicing to {N_DAYS}x{N_DAYS}.")
             prices = df.iloc[:N_DAYS, :N_DAYS].to_numpy(dtype=float)
        elif df.shape[0] >= N_DAYS and df.shape[1] >= N_DAYS:
            print(f"Warning: CSV shape is {df.shape}. Taking top-left {N_DAYS}x{N_DAYS} block.")
            prices = df.iloc[:N_DAYS, :N_DAYS].to_numpy(dtype=float)
        else:
            print(f"Error: CSV shape is {df.shape}, which is smaller than expected ({N_DAYS}, {N_DAYS}). Cannot proceed.")
            # Fallback to dummy for demonstration purposes if critical error
            dummy_prices = np.full((N_DAYS, N_DAYS), 100.0)
            for i in range(N_DAYS):
                for j in range(N_DAYS):
                    if i + j + 1 > N_DAYS:
                        dummy_prices[i,j] = np.nan
            if file_path == CSV_FILE_PATH: # Only save if it's the original user file being problematic
                 np.savetxt(f"dummy_{file_path}", dummy_prices, delimiter=",")
                 print(f"Created a dummy 'dummy_{file_path}' for demonstration as original was problematic.")
            return dummy_prices

        # Convert empty strings or specific placeholders for blanks to NaN
        # The problem says "Cells that would overflow the month are blank."
        # pd.read_csv usually handles empty strings as NaN if dtype is float, but explicit is safer.
        # If blanks are represented by something else, conversion might be needed here.
        # For now, assuming standard blank/empty string handling by read_csv to float conversion.

        # Ensure upper triangular as per problem for valid prices
        # (i,j) -> check-in Day i+1, length j+1 nights
        # Stay ends on day (i+1) + (j+1) - 1 = i+j+1
        # This must be <= N_DAYS
        # So, if i+j+1 > N_DAYS, price is NaN
        cleaned_prices = np.full((N_DAYS, N_DAYS), np.nan)
        for r in range(N_DAYS): # 0 to 29
            for c in range(N_DAYS): # 0 to 29
                # r: check-in day index (0 to 29) -> Day (r+1)
                # c: length of stay index (0 to 29) -> LOS (c+1)
                # Last day of stay (1-indexed) = (r+1) + (c+1) - 1 = r + c + 1
                if (r + c + 1) <= N_DAYS:
                    if r < prices.shape[0] and c < prices.shape[1]:
                         cleaned_prices[r,c] = prices[r,c]
                    # else: it remains NaN, which is correct for out-of-bounds from source data
                # else: it remains NaN, which is correct (overflows month)

        return cleaned_prices
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        print("Please ensure 'pricing_matrix_30x30.csv' is in the same directory as the script.")
        # Create a dummy file for execution to proceed for demonstration if not found
        dummy_prices = np.full((N_DAYS, N_DAYS), 100.0)
        # Make it upper triangular like
        for i in range(N_DAYS):
            for j in range(N_DAYS):
                if i + j +1 > N_DAYS : # check_in_day (i+1) + stay_length (j+1) -1 > 30 (day in month)
                     dummy_prices[i,j] = np.nan
        np.savetxt(file_path, dummy_prices, delimiter=",")
        print(f"Created a dummy '{file_path}' for demonstration.")
        return dummy_prices


def get_discount_for_stay_length(nightly_discounts_for_day, stay_length):
    """
    Determines the discount percentage for a given stay length for a specific day.
    nightly_discounts_for_day: array of 8 discount percentages for that day.
    stay_length: length of stay in nights.
    """
    if stay_length == 1:
        return 0.0  # No LOS discount for 1-night stay

    applicable_cutoff_indices = np.where(CUTOFFS >= stay_length)[0]
    
    if len(applicable_cutoff_indices) > 0:
        chosen_tier_index = applicable_cutoff_indices[0]
        return nightly_discounts_for_day[chosen_tier_index]
    else:
        # stay_length is greater than the largest cutoff, use the discount for the largest cutoff
        return nightly_discounts_for_day[-1]

def calculate_pms_price(base_rates, nightly_discounts_matrix, check_in_day_idx, stay_length):
    """
    Calculates the total price for a stay using the PMS pricing formula.
    base_rates: 1D array of 30 nightly base rates.
    nightly_discounts_matrix: 2D array (30 days x 8 cutoffs) of discount percentages.
    check_in_day_idx: 0-indexed check-in day.
    stay_length: Number of nights.
    """
    total_price = 0.0
    if check_in_day_idx + stay_length > N_DAYS: # Stay exceeds 30 days period for rates/discounts
        # This case should ideally be filtered out by NaN in target_matrix
        # or handled based on how such overflows are priced if they can occur
        return np.nan # Or a very high penalty if it can be chosen by optimizer

    for night_offset in range(stay_length):
        current_night_idx = check_in_day_idx + night_offset
        
        base_rate_for_night = base_rates[current_night_idx]
        discounts_for_current_night = nightly_discounts_matrix[current_night_idx, :]
        
        discount_percentage = get_discount_for_stay_length(discounts_for_current_night, stay_length)
        
        adjusted_price_for_night = base_rate_for_night * (1 - discount_percentage)
        total_price += adjusted_price_for_night
        
    return total_price

# --- Optimization Objective and Constraints ---

def objective_function(params, target_prices, non_blank_indices):
    """
    Objective function to minimize (sum of squared errors).
    params: Flattened array of [base_rates (30), nightly_discounts (30*8)].
    """
    base_rates = params[:N_DAYS]
    nightly_discounts_matrix = params[N_DAYS:].reshape((N_DAYS, N_CUTOFFS))
    
    total_squared_error = 0.0
    
    for r_idx, c_idx in non_blank_indices:
        check_in_day_idx = r_idx  # 0-indexed
        stay_length = c_idx + 1   # stay_length is 1-indexed column
        
        target_price = target_prices[r_idx, c_idx]
        
        # Ensure the stay is valid according to problem (ends within month)
        # This is already encoded by NaNs in target_prices for invalid cells
        # calculated_price should also respect this (e.g. check_in_day_idx + stay_length <= N_DAYS)

        pms_price = calculate_pms_price(base_rates, nightly_discounts_matrix, check_in_day_idx, stay_length)
        
        if np.isnan(pms_price): # Should not happen if inputs are correct and stay is valid based on non_blank_indices
            total_squared_error += 1e12 # High penalty
            continue

        total_squared_error += (pms_price - target_price)**2
        
    return total_squared_error

def get_constraints():
    """Defines the constraints for the optimization problem."""
    constraints = []
    
    # Constraint: Discounts must be strictly increasing for each dayPPP
    # p_dj < p_d,j+1  => p_d,j+1 - p_dj >= EPSILON_STRICT_INCREASE
    for day_idx in range(N_DAYS):
        for cutoff_j_idx in range(N_CUTOFFS - 1):
            # Constraint function c(x) >= 0
            # params are [base_rates (30), discounts (30*8)]
            # discount index in params: N_DAYS (offset for base_rates) + day_idx * N_CUTOFFS + cutoff_idx
            def constraint_func(params, day_idx=day_idx, cutoff_j_idx=cutoff_j_idx):
                idx1 = N_DAYS + day_idx * N_CUTOFFS + cutoff_j_idx
                idx2 = N_DAYS + day_idx * N_CUTOFFS + cutoff_j_idx + 1
                return params[idx2] - params[idx1] - EPSILON_STRICT_INCREASE
            
            constraints.append({'type': 'ineq', 'fun': constraint_func})
            
    return constraints

def get_bounds():
    """Defines the bounds for decision variables."""
    # Bounds for base rates: (e.g., 1.0 to a high value like 1000.0 or max_price_in_csv)
    # Smallest price could be low, largest ~900. Let's try (1, 1500)
    base_rate_bounds = [(1.0, 1500.0)] * N_DAYS 
    
    # Bounds for discounts: [0.0, 1.0]
    # To be safe with strict inequality and 1-discount term, use [0.0, 0.99]
    # But problem states percentages, 100% (1.0) could be valid. Let's use [0, 0.999] to ensure 1-disc > 0 slightly
    discount_bounds = [(0.0, 0.999)] * (N_DAYS * N_CUTOFFS)
    
    return Bounds([b[0] for b in base_rate_bounds + discount_bounds],
                  [b[1] for b in base_rate_bounds + discount_bounds])

# --- Main Execution ---
def main():
    print("Loading target prices...")
    target_prices = load_target_prices(CSV_FILE_PATH)
    
    # Identify non-blank cells (where optimization should focus)
    non_blank_indices = []
    for r in range(N_DAYS):
        for c in range(N_DAYS):
            if not np.isnan(target_prices[r, c]):
                non_blank_indices.append((r, c))

    if not non_blank_indices:
        print("Error: No valid data points found in the target price matrix. Exiting.")
        return

    print(f"Found {len(non_blank_indices)} non-blank target prices to optimize against.")

    # Initial guess (x0)
    # Base rates: from 1-night stays if available, else average.
    initial_base_rates = np.full(N_DAYS, 100.0) # Default average
    for i in range(N_DAYS):
        if not np.isnan(target_prices[i, 0]): # Col 0 is 1-night stay
             # Assuming 1 night stay has 0 discount, base_rate = price
            initial_base_rates[i] = target_prices[i, 0]
    initial_base_rates = np.clip(initial_base_rates, 1.0, 1500.0) # Ensure within bounds

    # Discounts: small, increasing values
    initial_discounts_flat = []
    for _ in range(N_DAYS):
        # Spread initial discounts more widely, e.g., up to 50%
        day_discounts = np.sort(np.random.uniform(0.01, 0.5, N_CUTOFFS)) # Random increasing
        # Ensure strict increase for initial guess too
        for k in range(N_CUTOFFS -1):
            if day_discounts[k+1] < day_discounts[k] + EPSILON_STRICT_INCREASE:
                 day_discounts[k+1] = day_discounts[k] + EPSILON_STRICT_INCREASE
        day_discounts = np.clip(day_discounts, 0.0, 0.999) # Ensure within bounds
        initial_discounts_flat.extend(day_discounts)
    
    x0 = np.concatenate([initial_base_rates, np.array(initial_discounts_flat)])

    print("Setting up bounds and constraints...")
    bounds = get_bounds()
    constraints = get_constraints()

    print("Starting optimization (this may take a while)...")
    options = {'maxiter': 2000, 'disp': True, 'ftol': 1e-8} # ftol for SLSQP

    result = minimize(objective_function, 
                      x0, 
                      args=(target_prices, non_blank_indices),
                      method='SLSQP',
                      bounds=bounds, 
                      constraints=constraints,
                      options=options)

    if result.success:
        print("\nOptimization successful!")
    else:
        print("\nOptimization failed or did not converge.")
        print(f"Message: {result.message}")
        print(f"Current SSE: {result.fun:.4f}")
        print("Showing and saving the best result found so far.")

    optimized_params = result.x
    final_sse = result.fun
    num_points = len(non_blank_indices)
    final_rmse = np.sqrt(final_sse / num_points) if num_points > 0 else 0

    optimized_base_rates = optimized_params[:N_DAYS]
    optimized_discounts = optimized_params[N_DAYS:].reshape((N_DAYS, N_CUTOFFS))

    print("\n--- Optimized Nightly Base Rates ---")
    for i, rate in enumerate(optimized_base_rates):
        print(f"Day {i+1:2d}: {rate:8.2f}")
    # Save base rates to CSV with Day column
    pd.DataFrame({'nights': np.arange(1, N_DAYS+1), 'BaseRate': optimized_base_rates}).to_csv('optimized_base_rates.csv', index=False)

    print("\n--- Optimized Nightly Discount Percentages (for cutoffs: " + ", ".join(map(str, CUTOFFS)) + " nights) ---")
    for i in range(N_DAYS):
        discount_strs = [f"{d*100:5.2f}%" for d in optimized_discounts[i,:]]
        print(f"Day {i+1:2d}: " + " | ".join(discount_strs))
    # Save discounts to CSV with Day column
    discount_cols = [f'cutoff_{c}' for c in CUTOFFS]
    discounts_df = pd.DataFrame(optimized_discounts, columns=discount_cols)
    discounts_df.insert(0, 'nights', np.arange(1, N_DAYS+1))
    discounts_df.to_csv('optimized_discounts.csv', index=False)

    print(f"\nFinal Sum of Squared Errors (SSE): {final_sse:.4f}")
    print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.4f}")

    # Validate discount monotonicity
    valid_monotonicity = True
    for i in range(N_DAYS):
        for j in range(N_CUTOFFS - 1):
            if optimized_discounts[i,j+1] - optimized_discounts[i,j] < EPSILON_STRICT_INCREASE / 2: # check with tolerance
                print(f"Warning: Monotonicity violated or too close for Day {i+1}, Tier {j+1} vs {j+2}")
                print(f"Values: {optimized_discounts[i,j]*100:.2f}% vs {optimized_discounts[i,j+1]*100:.2f}%")
                valid_monotonicity = False
    if valid_monotonicity:
        print("\nDiscount monotonicity constraint appears to be satisfied.")

    # Post-process to enforce monotonicity
    for i in range(N_DAYS):
        for j in range(1, N_CUTOFFS):
            if optimized_discounts[i, j] <= optimized_discounts[i, j-1]:
                optimized_discounts[i, j] = optimized_discounts[i, j-1] + 1e-6  # or a small epsilon

    # The explanation text will be printed after this script's output by the calling agent.

if __name__ == '__main__':
    main() 