import numpy as np
from scipy.stats import norm


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence interval for a given dataset.

    Args:
        data (list or numpy array): The dataset.
        confidence (float): The confidence level (e.g., 0.95 for 95% confidence).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
               Returns (None, None) if the input data is invalid.
    """
    if not isinstance(data, (list, np.ndarray)):
        print("Error: Input data must be a list or a NumPy array.")
        return None, None

    if len(data) == 0:
        print("Error: Input data must not be empty.")
        return None, None

    try:
        data = np.array(data)
        mean = np.mean(data)
        std_dev = np.std(data)
        n = len(data)
        standard_error = std_dev / np.sqrt(n)
        z = norm.ppf((1 + confidence) / 2)  # Z-score for the given confidence level
        margin_of_error = z * standard_error
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        return lower_bound, upper_bound
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


if __name__ == '__main__':
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    confidence_level = 0.95
    lower, upper = calculate_confidence_interval(data, confidence_level)

    if lower is not None and upper is not None:
        print(f"Confidence Interval ({confidence_level*100}%): ({lower:.2f}, {upper:.2f})")

    data2 = [10, 12, 14, 15, 18, 20, 22, 25, 28, 30]
    confidence_level2 = 0.99
    lower2, upper2 = calculate_confidence_interval(data2, confidence_level2)

    if lower2 is not None and upper2 is not None:
        print(f"Confidence Interval ({confidence_level2*100}%): ({lower2:.2f}, {upper2:.2f})")

    data3 = []
    lower3, upper3 = calculate_confidence_interval(data3, 0.95)

    data4 = [1,2,3]
    lower4, upper4 = calculate_confidence_interval(data4, "hello")
