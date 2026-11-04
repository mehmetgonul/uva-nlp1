import torch


def bag_of_words(token_ids, vocab_size):
    """Count each token's occurrences in one sentence."""
    ids = torch.tensor(token_ids, dtype=torch.long)
    bags = torch.bincount(ids, minlength=vocab_size)

    return bags

def token_indicators(token_ids, vocab_size):
    """Mark each token's presence in one sentence."""

    indicators = bag_of_words(token_ids, vocab_size)

    # bag of words already keep the count ex. (3 4 0)
    # we convert that to binary indicator ex. (1 1 0)
    indicators = (indicators > 0)

    return indicators



def overlap(token_ids, vocab_size):
    """Concatenate premise, hypothesis, and shared-token indicators."""
    premise, hypothesis = token_ids

    # get the token indicators for the premise
    premise_indicators = token_indicators(premise, vocab_size)

    # get the token indicators for the hypothesis
    hypothesis_indicators = token_indicators(hypothesis, vocab_size)

    # get the common tokens
    common_indicators = premise_indicators * hypothesis_indicators

    return torch.cat([
        premise_indicators,
        hypothesis_indicators,
        common_indicators
    ])
