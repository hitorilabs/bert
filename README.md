# Overview

This repo houses the experiments for BERT my beloved. 

Initially, I was just looking for a nice clean torch implementation of BERT
where I can easily check against the one provided by HuggingFace. Meaning, map
the weights from HF onto the custom implementation and check if it can produce
roughly the same results.

It ended up being a bit more tedious than I thought, so I thought it would be
worth writing up some notes about it.

There's a slight difference (about `2e-06`) in results you would get compared
to HF, but it's probably still within range of floating point precision
problems... right?

some things I didn't know about or didn't expect:
- HF implementation is actually seems to handwrite many of the modules vs.
  being able to use a built-in torch module
- MHA naturally expects qkv to be concatenated (was a bit tedious to write the
  mapping, but this is quite common)
- not sure what configuration of TransformerEncoder would get me a BERT
  implementation that lines up w/ HF... but tbh it's cleaner now that I've
  pulled out only the specific implementation details that I need.
- still not 100% sure, but I think the key difference in my first (incorrect)
  implementation is in the way the skip connection is computed in
  TransformerEncoderLayer vs. my BertEncoderLayer

BERT as we know it today has changed a lot over the years (pre-training,
implementation details, etc.), so it's actually not really worth your time to
try and get the "foundations" down. There's a lot of outdated slop on the
internet, so IMO your best bet is doing something similar to this - a top-down
approach.

# Usage

Download the weights from HF (https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking)

remap + load in weights from HF to the custom implementation
```
python3 convert_hf.py /path/to/google-bert/bert-large-uncased-whole-word-masking/model.safetensors /path/to/output/dir
```

check the results
```
python3 check.py  --hf_model_path=/path/to/google-bert/bert-large-uncased-whole-word-masking --custom_model_path=/path/to/output/dir
```
