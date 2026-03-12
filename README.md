# Interpreting social bias in autoregressive models

## v1

- Modeling the probability of the final token ($\text{IG}^2$ templates which end with mask)

### To start

- `cd` in the `data_ig2/v1` directory and run the `generate_data_v1.py` script.
- `cd` out to root of this repository and run the `ig2_gpt2_analyze_bias.py`

#### Acknowledgments

- GPT2 implementation is largely borrowed from Karpathy's NanoGPT tutorial.
- The code style and analysis done is largely borrowed from and inspired from the [[paper]](https://arxiv.org/abs/2406.10130)'s implementation.
- Pieces of code (certain functions) generated using AI has been labeled (through comments) appropriately.
