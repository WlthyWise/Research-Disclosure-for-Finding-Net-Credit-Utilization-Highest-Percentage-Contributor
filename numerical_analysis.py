import random
from typing import List, Tuple, Optional
import csv

NUM_TRAIL_ACCOUNTS = 5

output_filename = "trails/numerical_analysis_results.csv"
with open(output_filename, 'w', newline='') as csvfile:
    pass

# Time Complexity: O(nlogn), Space Compexity: O(n)
class CreditUtilizationSorterDirectRatioLoopBase:
    """
    A system to sort credit accounts based on their utilization ratio.
    The implementation is based on the provided proof of correctness.
    """

    def net_utilization_ratio(self, accounts: list[tuple[float, float]]):
        b_total = 0
        l_total = 0
        for a in accounts:
            b_total += a[0]
            l_total += a[1]
        return b_total / l_total

    def simulate_payoff_utilization_percentage_delta(self, sorted_accounts):
        original_ratio = self.net_utilization_ratio(sorted_accounts)
        # Create a copy of the list before popping
        sorted_accounts_copy = list(sorted_accounts)
        sorted_accounts_copy[0] = (0, sorted_accounts_copy[0][1])
        resultant_ratio = self.net_utilization_ratio(sorted_accounts_copy)
        return resultant_ratio - original_ratio

    def sort(self, accounts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sorts accounts using the simpler, equivalent utilization ratio r_i.

        This method is more pragmatic as sorting by r_i = b_i / l_i
        is mathematically equivalent and computationally simpler.

        Args:
            accounts: A list of tuples, where each tuple is (balance, limit).

        Returns:
            A new list of account tuples sorted in descending order of utilization.
        """
        if not accounts:
            return []
            
        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        # Sort directly by the utilization ratio b_i / l_i
        sorted_accounts = sorted(
            accounts,
            key=lambda acc: acc[0] / acc[1],
            reverse=True
        )
        return sorted_accounts
    
    def find_largest_utilization_account(self, accounts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not accounts:
            return None

        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        sorted_accounts = self.sort(list(accounts))
        return sorted_accounts[0], sorted_accounts

# Time Complexity: O(nlogn), Space Compexity: O(n)
class CreditUtilizationSorterSumRatioLoopInvariantA(CreditUtilizationSorterDirectRatioLoopBase):

    def simulate_payoff_utilization_percentage_delta(self, sorted_accounts):
        original_ratio = self.net_utilization_ratio(sorted_accounts)
        # Create a copy of the list before popping
        sorted_accounts_copy = list(sorted_accounts)
        sorted_accounts_copy[len(sorted_accounts_copy) - 1] = (0, sorted_accounts_copy[len(sorted_accounts_copy) - 1][1])
        resultant_ratio = self.net_utilization_ratio(sorted_accounts_copy)
        return resultant_ratio - original_ratio

    def sort(self, accounts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sorts accounts using the exact key R_i from the proof.

        This method is a direct implementation of the sorting key:
        R_i = (b_i * L_total) / (l_i * B_total)

        Args:
            accounts: A list of tuples, where each tuple is (balance, limit).

        Returns:
            A new list of account tuples sorted in descending order of utilization.
        """
        if not accounts:
            return []

        # Validate that all limits are positive, as assumed in the proof
        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        # Step 1: Calculate total limits (L_total) and total balances (B_total)
        l_total = sum(limit for _, limit in accounts)

        # Handle edge case where total balance is zero to avoid division by zero
        if l_total == 0:
            # All utilizations are 0, order doesn't matter. Return a copy.
            return list(accounts)

        # Step 2: Sort C using the key R_i in descending order
        # The key is a lambda function that calculates R_i for each account `acc`.
        # acc[0] is the balance (b_i) and acc[1] is the limit (l_i).
        sorted_accounts = sorted(
            accounts,
            key=lambda acc: acc[0] / l_total
        )
        return sorted_accounts
    
    def find_largest_utilization_account(self, accounts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not accounts:
            return None

        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        sorted_accounts = self.sort(accounts)
        return sorted_accounts[0], sorted_accounts

# Time Complexity: O(nlogn), Space Compexity: O(n)
class CreditUtilizationSorterSumRatiosWeightedRatioLoopInvariantB(CreditUtilizationSorterDirectRatioLoopBase):
    def sort(self, accounts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sorts accounts using the exact key R_i from the proof.

        This method is a direct implementation of the sorting key:
        R_i = (b_i * L_total) / (l_i * B_total)

        Args:
            accounts: A list of tuples, where each tuple is (balance, limit).

        Returns:
            A new list of account tuples sorted in descending order of utilization.
        """
        if not accounts:
            return []

        # Validate that all limits are positive, as assumed in the proof
        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        # Step 1: Calculate total limits (L_total) and total balances (B_total)
        l_total = sum(limit for _, limit in accounts)
        b_total = sum(balance for balance, _ in accounts)

        # Handle edge case where total balance is zero to avoid division by zero
        if b_total == 0:
            # All utilizations are 0, order doesn't matter. Return a copy.
            return list(accounts)

        # Step 2: Sort C using the key R_i in descending order
        # The key is a lambda function that calculates R_i for each account `acc`.
        # acc[0] is the balance (b_i) and acc[1] is the limit (l_i).
        sorted_accounts = sorted(
            accounts,
            key=lambda acc: (acc[0] * l_total) / (acc[1] * b_total),
            reverse=True
        )
        return sorted_accounts

    def find_largest_utilization_account(self, accounts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not accounts:
            return None

        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        sorted_accounts = self.sort(accounts)
        return sorted_accounts[0], sorted_accounts

# Time Complexity: O(nlogn), Space Compexity: O(n)
class CreditUtilizationSorterRatiosSumWeightedRatioLoopInvariantC(CreditUtilizationSorterDirectRatioLoopBase):
    def sort(self, accounts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Sorts credit accounts using the explicit comparison predicate from the proof.

        This function directly implements the sorting key specified in the proof:
        R'_i = (b_i / l_i) * sum(b_j / l_j).

        It pre-calculates the sum of all ratios (the "normalizer") and then uses
        this full key in each comparison within a Bubble Sort algorithm. This method
        is less efficient than using the simplified ratio but perfectly matches
        the formal predicate.

        Args:
            accounts: A list of tuples, where each tuple is (balance, credit_limit).

        Returns:
            A new list of account tuples sorted by the key R'_i in descending order.
        """
        n = len(accounts)
        if n < 2:
            return list(accounts)

        # Create a mutable copy of the list to sort.
        sorted_accounts = list(accounts)

        # 1. Calculate the constant normalizer: R_sum = sum(b_i / l_i).
        # This corresponds to R_normalizer in the pseudocode.
        try:
            r_sum = sum(balance / limit for balance, limit in accounts)
        except ZeroDivisionError:
            raise ValueError("Error: A credit limit cannot be zero.")

        # If r_sum is 0 (e.g., all balances are 0), the sorting keys for all
        # accounts will be 0. Their relative order is undefined and doesn't
        # matter, so we can return the list as is.
        if r_sum == 0:
            return sorted_accounts

        # 2. Implement Bubble Sort using the complex key for comparison.
        for i in range(n):
            # The inner loop applies the comparison to adjacent elements (k, k+1).
            for k in range(0, n - i - 1):
                balance_k, limit_k = sorted_accounts[k]
                balance_k_plus_1, limit_k_plus_1 = sorted_accounts[k+1]

                # Calculate the full sorting keys R'_k and R'_{k+1} as defined.
                key_k = (balance_k / limit_k) * r_sum
                key_k_plus_1 = (balance_k_plus_1 / limit_k_plus_1) * r_sum

                # The MAINTENANCE step:
                # To sort in DESCENDING order, if the key for the current element
                # is less than the key for the next element, swap them.
                # This action enforces the invariant: R'_k > R'_{k+1}.
                if key_k < key_k_plus_1:
                    sorted_accounts[k], sorted_accounts[k+1] = sorted_accounts[k+1], sorted_accounts[k]

        return sorted_accounts

    def find_largest_utilization_account(self, accounts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not accounts:
            return None

        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        sorted_accounts = self.sort(accounts)
        return sorted_accounts[0], sorted_accounts

# Time Complexity: O(nlogn), Space Compexity: O(n)
class CreditUtilizationSorterRatiosProductSumLoopInvariantD(CreditUtilizationSorterDirectRatioLoopBase):
    def sort(self, accounts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sorts accounts using the balance-limit product from the proof.

        This method is a direct implementation of the sorting key:
        R_i = (b_i * l_i) / sum(b_j * l_j)

        Args:
            accounts: A list of tuples, where each tuple is (balance, limit).

        Returns:
            A new list of account tuples sorted in descending order of balance-limit product.
        """
        if not accounts:
            return []

        # Validate that all limits are positive, as assumed in the proof
        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        # Step 1: Calculate the sum of all individual balance-limit products
        bl_sum = sum(balance * limit for balance, limit in accounts)

        # Handle edge case where the sum is zero to avoid division by zero
        if bl_sum == 0:
            # All products are 0, order doesn't matter. Return a copy.
            return list(accounts)

        # Step 2: Sort C using the key R_i in descending order
        # The key is a lambda function that calculates R_i for each account `acc`.
        # acc[0] is the balance (b_i) and acc[1] is the limit (l_i).
        # Since the denominator is constant, we can just sort by the numerator b_i * l_i
        sorted_accounts = sorted(
            accounts,
            key=lambda acc: ((acc[0] * acc[1]) / bl_sum),
            reverse=True
        )
        return sorted_accounts

    def find_largest_utilization_account(self, accounts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not accounts:
            return None

        if any(limit <= 0 for _, limit in accounts):
            raise ValueError("Credit limits must be positive.")

        sorted_accounts = self.sort(accounts)
        return sorted_accounts[0], sorted_accounts

# Time Complexity: O(n^2), Space Compexity: O(n)
def best_fit_bin_packing_simulate_payoff_utilization_percentage_delta_for_account_n(accounts: list[tuple[float, float]], n):
    s = CreditUtilizationSorterDirectRatioLoopBase()
    original_ratio = s.net_utilization_ratio(accounts)
    # Create a copy of the list before popping
    sorted_accounts_copy = list(accounts)
    sorted_accounts_copy[n] = (0, sorted_accounts_copy[n][1])
    resultant_ratio = s.net_utilization_ratio(sorted_accounts_copy)
    return resultant_ratio - original_ratio

def generate_random_accounts(num_accounts: int) -> List[Tuple[float, float]]:
    """Generates a list of pseudo-random credit accounts."""
    accounts = []
    for _ in range(num_accounts):
        limit = random.uniform(1000, 20000)
        balance = random.uniform(0, limit)
        accounts.append((balance, limit))
    return accounts

def main():
    """Main function to run the analysis in a loop."""
    analyzers = [
        ("Direct Ratio based", CreditUtilizationSorterDirectRatioLoopBase),
        ("Sum Ratio based", CreditUtilizationSorterSumRatioLoopInvariantA),
        ("Sum Ratios based", CreditUtilizationSorterSumRatiosWeightedRatioLoopInvariantB),
        ("Ratios Sum based", CreditUtilizationSorterRatiosSumWeightedRatioLoopInvariantC),
        ("Ratios Product Sum based", CreditUtilizationSorterRatiosProductSumLoopInvariantD)
    ]

    with open(output_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Trail #1", "Analyzer Name", "Balance", "Limit", "L-Shifted Delta"])

    i = 0        
    while True:
        i += 1
        print("\n" + "="*50)
        print("Generating new random accounts...")
        my_accounts = generate_random_accounts(NUM_TRAIL_ACCOUNTS)
        print("Generated Accounts:", my_accounts)
        print("-"*50)

        for n in range(NUM_TRAIL_ACCOUNTS):
            brute_forced_delta = best_fit_bin_packing_simulate_payoff_utilization_percentage_delta_for_account_n(my_accounts, n)
            print(f"  - Account n={n+1} 0-balance Delta:   {brute_forced_delta:.2%}")
            balance, limit = my_accounts[n]
            with open(output_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([i, f"Account n={n+1}", f"{balance:,.2f}", f"{limit:,.2f}", f"{brute_forced_delta:.2%}"])

        results = []
        for name, analyzer_class in analyzers:
            analyzer = analyzer_class()
            highest_util_account, sorted_accounts = analyzer.find_largest_utilization_account(my_accounts)
            if highest_util_account:
                results.append((name, highest_util_account, sorted_accounts, analyzer))        

        for name, highest_util_account, sorted_accounts, analyzer in results:
            balance, limit = highest_util_account
            delta = analyzer.simulate_payoff_utilization_percentage_delta(sorted_accounts)
            print(f"{name} Account with the highest utilization:")
            print(f"  - Balance: ${balance:,.2f}")
            print(f"  - Limit:   ${limit:,.2f}")
            print(f"  - L-Shift Delta:   {delta:.2%}")
            with open(output_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([i, name, f"{balance:,.2f}", f"{limit:,.2f}", f"{delta:.2%}"])
            
        print("="*50)

if __name__ == "__main__":
    main()