def compound_interest(principal, rate, time, n):
    amount = principal * (1 + rate/n) ** (n*time)
    return amount

# Example usage
print(compound_interest(1000, 0.05, 10, 1))
