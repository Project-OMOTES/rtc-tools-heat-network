import numpy as np
import matplotlib.pyplot as plt

def calculate_annuity_factor(discount_factor, life):
    return (1 - (1 + discount_factor) ** -life) / discount_factor

# Generate a range of discount factor values
discount_factors = np.linspace(0.0001, 0.99999, 100)

# Generate the plot
plt.figure(figsize=(10, 6))

# Loop over different years in a range
for life in range(1, 30, 5):
    # Calculate the corresponding annuity factors
    annuity_factors = calculate_annuity_factor(discount_factors, life)
    # Plot the line for each year
    plt.plot(discount_factors, annuity_factors, label=f'Life = {life} years')

plt.xlabel('Discount Factor')
plt.ylabel('Annuity Factor')
plt.title('Discount Factor vs Annuity Factor')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('discount_factor_vs_annuity_factor.png')

