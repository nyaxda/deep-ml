import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    model = [math.exp(scores[i] - max(scores)) / (sum(math.exp(scores[j] - max(scores)) for j in range(len(scores)))) for i in range (len(scores))]
    return model