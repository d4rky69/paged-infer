"""Autoregressive token sampler with temperature, top-k, and top-p (nucleus) filtering."""

import numpy as np


class Sampler:
    """Samples next token IDs from unnormalized vocabulary logits."""

    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> int:
        """Samples a single token index from logits.
        
        Args:
            logits: 1D array of shape (vocab_size,).
            temperature: Softmax temperature (>0). If <= 1e-4, greedy argmax is used.
            top_p: Nucleus cumulative probability threshold [0.0, 1.0].
            top_k: Top-k filtering threshold (>0).
            
        Returns:
            Selected token ID integer.
        """
        # Greedy sampling
        if temperature <= 1e-4:
            return int(np.argmax(logits))

        # Scale by temperature
        logits = logits / temperature

        # Top-K filtering
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits = logits.copy()
            logits[indices_to_remove] = -float("Inf")

        # Softmax computation
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right so we keep at least the first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs = probs.copy()
            probs[indices_to_remove] = 0.0
            sum_probs = np.sum(probs)
            if sum_probs > 0:
                probs /= sum_probs
            else:
                return int(np.argmax(logits))

        # Multinomial sample
        return int(np.random.choice(len(probs), p=probs))
