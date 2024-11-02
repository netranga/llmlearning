# LLM Token Prediction Explorer

## Overview
LLM Token Prediction Explorer is a Python library created to deepen my understanding of next-token generation process in language models. I've created helepr tools and visualizations for experimenting with next-token predictions, probability distributions, token entropy, and other insights into how autoregressive language models, such as GPT-2, generate text.

This library is ideal for learners and developers who want to explore how language models predict text, understand probability distributions over tokens, and analyze how context and various parameters affect token generation.

## Key Features
- **Vocabulary Check**: Check if a word is in the model’s vocabulary.
- **Next-Token Prediction**: Given a text, get the next most probable words.
- **Probability Distributions**: Visualize probability distributions over vocabulary tokens.
- **Entropy Calculation**: Quantify model uncertainty during token generation.
- **Temperature and Sampling Adjustments**: Explore how parameters like temperature, top-k, and top-p affect prediction.
- **Cumulative Probability Cutoff**: Find the minimum token set needed to reach a confidence threshold.
- **Token History Comparison**: Track how the probability of specific tokens changes over time.
- **Rare and Common Token Analysis**: Observe which tokens have the highest and lowest probabilities.
  
## Installation
To install the required dependencies:
```bash
pip install transformers torch
