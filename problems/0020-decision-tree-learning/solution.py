import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    label_counts = Counter(labels)
    total_counts = len(labels)
    entropy = 0.0
    for count in label_counts.values():
        prob = count/total_counts
        entropy -= prob * math.log(prob)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    total_entropy = calculate_entropy([ex[target_attr] for ex in examples])
    counts = Counter(ex[attr] for ex in examples)

    weighted_entropy = 0.0
    for value, count in counts.items():
        subset_labels = [ex[target_attr] for ex in examples if ex[attr] == value]
        weighted_entropy += (count / len(examples)) * calculate_entropy(subset_labels)

    return total_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    counts = Counter(example[target_attr] for example in examples)

    return sorted(counts, key=lambda x: (-counts[x], x))[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    labels = {ex[target_attr] for ex in examples}

    # Pure node: return the label directly
    if len(labels) == 1:
        return labels.pop()
    # No attributes left: fall back to the majority class
    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = max(
        attributes,
        key=lambda a: calculate_information_gain(examples, a, target_attr),
    )

    remaining_attrs = [a for a in attributes if a != best_attr]
    branches = {}

    for value in sorted({ex[best_attr] for ex in examples}):
        subset = [ex for ex in examples if ex[best_attr] == value]
        branches[value] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return {best_attr: branches}