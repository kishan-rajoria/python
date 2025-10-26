# Inferential Statistics Example: Independent Samples t-test

# Import the t-test function from scipy.stats library
from scipy.stats import ttest_ind
import numpy as np # Used for calculating means

# --- Sample Data ---
# Imagine we have test scores from two different teaching methods (Group A and Group B)
# These are independent samples because the students in Group A are different from Group B.
group_a_scores = [85, 90, 78, 92, 88, 76, 89]
group_b_scores = [79, 82, 75, 81, 85, 80, 77, 83]

# --- Hypothesis Setup ---
# Null Hypothesis (H₀): There is no significant difference between the mean scores of Group A and Group B.
# (i.e., μ_A = μ_B)
# Alternative Hypothesis (H₁): There is a significant difference between the mean scores of Group A and Group B.
# (i.e., μ_A ≠ μ_B) - This is a two-tailed test.

significance_level = 0.05 # Standard alpha level

# --- Perform the t-test ---
# ttest_ind calculates the T-test for the means of TWO INDEPENDENT samples of scores.
# It returns the calculated t-statistic and the two-tailed p-value.
# By default, it assumes equal population variances. If variances might be different,
# you can add the argument: equal_var=False
t_statistic, p_value = ttest_ind(group_a_scores, group_b_scores)

# --- Output Results and Interpretation ---
print("--- Independent Samples t-test Results ---")
print(f"Group A Mean Score: {np.mean(group_a_scores):.2f}")
print(f"Group B Mean Score: {np.mean(group_b_scores):.2f}")
print(f"\nCalculated t-statistic: {t_statistic:.4f}")
print(f"Calculated p-value (two-tailed): {p_value:.4f}")

# --- Decision ---
print(f"\nSignificance Level (alpha): {significance_level}")
if p_value < significance_level:
    print(f"Decision: Reject the Null Hypothesis (H₀). P-value ({p_value:.4f}) is less than alpha ({significance_level}).")
    print("Conclusion: There is a statistically significant difference between the mean scores of the two groups.")
else:
    print(f"Decision: Fail to Reject the Null Hypothesis (H₀). P-value ({p_value:.4f}) is greater than or equal to alpha ({significance_level}).")
    print("Conclusion: There is not enough evidence to conclude a statistically significant difference between the mean scores of the two groups.") 