# gghfez/ik_llama.cpp

A personal fork of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) focused on representation engineering: Jacobian-lens readouts, residual-stream interventions and control vectors, plus some audio/vision model support.

> NOTE:
> This is a research/hobby fork, not affiliated with upstream. Things may change or break without notice. Please report bugs in upstream features to ikawrakow/ik_llama.cpp, and bugs in the features below here.

Build exactly like upstream. See the [upstream README](../README.md) and
[build docs](../docs/build.md).

```bash
git clone https://github.com/gghfez/ik_llama.cpp
cd ik_llama.cpp
git checkout jacobian-lens
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

---

## Jacobian lens

A Jacobian lens is a per-layer matrix `J_l` (`d_model × d_model`) that maps the residual stream at
layer `l` into the final-layer basis. Reading out at layer `l` is:

```
logits_l = lm_head( output_norm( J_l · h_l ) )
```

It reuses the model's own final norm and unembedding, so readouts are directly comparable to the model's real output distribution. With `jlens_use_jacobian: false` it falls back to a plain logit lens.

### Features

- **Per-token readout:** Retrieve the top-k lens predictions for every generated token across all fitted layers.
- **Prompt probe:** Execute readouts at every position of the prompt, including audio and image embeddings.
- **Interventions:** Modify the residual stream during the decoding process via steering, ablation, swapping and replacement.
- **Masks:** Filter out special tokens or attractor tokens from readouts.


### Usage

| Flag | Description |
|---|---|
| `--jlens FNAME` | Load a pre-fit Jacobian lens (GGUF) for interpretability analysis and steering |
| `--jlens-mask FNAME` | JSON token mask to filter unwanted tokens from jlens output (repeatable) |
| `--jlens-device N` | Where the `J` matrices live: `-2` = same as `lm_head` (default, fastest), `-1` = CPU, `>=0` = GPU index |
| `--no-jlens-offload` | Shorthand for `--jlens-device -1` (saves VRAM, slower) |

```bash
#Without attractor mask
./build/bin/llama-server -m model.gguf --jlens model.jlens.gguf
#With attractor mask
./build/bin/llama-server -m model.gguf --jlens model.jlens.gguf --jlens-mask attractors.json
```

#### Lens file format

A GGUF containing:

- `jlens.d_model` (u32): must equal the model's `n_embd`
- `jlens.n_prompts` (u32, optional): number of prompts used to fit the lens
- `jlens.J.<layer>`: one `[d_model, d_model]` tensor per fitted layer

#### Request fields

**Readout**
> NOTE:
> 
> Currently only works on the `/completion` endpoint
> 
> `jlens_mask_special` is likely to change soon

| Field | Default | Description |
|---|---|---|
| `jlens_n_probs` | `0` | Top-k tokens per layer. `0` disables the readout |
| `jlens_layers` | all fitted | `18`, `"18,24,30-44"`, or `[18, "30-44"]` |
| `jlens_use_jacobian` | `true` | `false` = plain logit lens |
| `jlens_mask` | `true` | `true` = server's `--jlens-mask`, `false` = none, or an inline mask object |
| `jlens_mask_special` | none | `true`, or any of `"control"`, `"unused"`, `"user_defined"`, `"byte"`, `"unknown"` |
| `jlens_mask_renorm` | `false` | Renormalise probabilities over the unmasked vocab |

**Prompt probe**

| Field | Default | Description |
|---|---|---|
| `jlens_prompt` | `false` | Also read out at prompt positions (needs `jlens_n_probs > 0`, disables prompt cache) |
| `jlens_prompt_pos_start` | `0` | Inclusive |
| `jlens_prompt_pos_end` | `-1` | Exclusive, `-1` = end of prompt |

#### Example: readout

```bash
curl -s localhost:8080/completion -d '{  
"prompt": "In one word, the biggest lie humanity has ever told is:",  
"n_predict": 4,  
"jlens_n_probs": 5,  
"jlens_layers": "42,44-45",  
"jlens_mask_special": true  
}' |jq '.jlens_probabilities[0].layers'
```

Each generated token gets one entry:

```json
{
  "layer": 45,
  "probs": [
    {
      "id": 93462,
      "tok_str": " MONEY",
      "prob": 0.06255664676427841
    },
    /* … */
  ]
}
```
Prompt-probe results come back in `jlens_prompt_probabilities` with an extra `pos` field. Media positions have `id: -1` and `"media": "image"` or `"audio"`.

#### Interventions

> Works on /completions and /chat/completions endpoints

`jlens_interventions` is an array of edits applied to the residual stream (`l_out-<layer>`) before the readout captures it. 

| Mode | Fields | Effect |
|---|---|---|
| `steer` | `token`, `alpha` | `h += α·‖h‖·v̂` |
| `ablate` | `token`  | Project out `v̂` |
| `swap` | `from`, `to` | Exchange the two tokens' lens coordinates |
| `replace` | `from`, `to` | Add the `from` coordinate -> `to` then zero the `from` |

Token strings must be exactly one token. Try a leading space (`" Paris"`).

```bash
curl -s localhost:8080/completion -d '{
  "prompt": "My favourite city is",
  "n_predict": 20,
  "jlens_interventions": [
    { "mode": "swap", "from": " Paris", "to": " Tokyo", "layers": [24, 36] }
  ]
}'
```

(TODO: Document mask file format)

---

## Credits

- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- *(Jacobian lens paper citation pending)*

