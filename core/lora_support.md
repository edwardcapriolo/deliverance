# LoRA / PEFT Adapter Support

Deliverance is adding support for LoRA (Low-Rank Adaptation) adapters in the standard
HuggingFace PEFT format. The goal is to let a single resident base model take on specialized
behavior — a particular skill, style, or persona — by loading a small adapter file on top of
it, instead of downloading and holding a second full multi-gigabyte checkpoint per variant.
This page is written for readers who know Deliverance but have never worked with LoRA/PEFT
adapters before.

## What Is A LoRA Adapter?

A LoRA adapter is not a second model. It is a small set of extra weights, trained separately
from the base model, that nudge the base model's behavior in a particular direction when
combined with it at inference time.

Full fine-tuning produces a new copy of every weight in the model — for a multi-billion
parameter model, that is another multi-gigabyte file per variant. LoRA instead trains a pair of
much smaller matrices per targeted layer (a low-rank decomposition, hence the name) that
approximate the *change* fine-tuning would have made, without touching the base weights
directly. The practical result is what matters here, not the linear algebra: an adapter file is
typically single-digit to a few tens of megabytes, it is swappable independently of the base
model, and it does not duplicate the base model's weights.

## Why This Matters

- **Cheap specialization**: get a specialized assistant, tone, or skill on top of one base
  model without paying the storage and load-time cost of a full model per variant.
- **Fast iteration**: adapters are small enough to train, share, and swap far more quickly than
  retraining or redistributing a full checkpoint.
- **One resident base model, many behaviors**: the eventual goal (Phase 2 below) is to keep one
  base model loaded and switch which adapter is applied per request, rather than loading a
  different full model for each behavior.
- **Standard, portable format**: adapters trained anywhere in the HuggingFace PEFT ecosystem
  should work with Deliverance's support, without a Deliverance-specific training or export
  step.

## Current Status

**This is not usable for inference today.** As of this doc, only adapter *parsing* has landed
([#153](https://github.com/edwardcapriolo/deliverance/pull/153)): Deliverance can fetch a PEFT
adapter repository, parse its `adapter_config.json`, and read the low-rank weight deltas out of
its `adapter_model.safetensors` file. Nothing in model loading or generation consumes an adapter
yet — there is no way to actually apply one to a running model. Treat the feature as
in-progress, not available.

## Roadmap

1. **Adapter parsing** — done. Deliverance can read an adapter's config and weights into memory
   and validate them against the adapter's own declared shape.
2. **Merge-at-load (Phase 1)** — planned next. The adapter's weights are merged into a copy of
   the base model's weights when the model is loaded, producing a single adapted model. This is
   the simpler of the two integration modes and proves the format and math end-to-end before
   tackling anything more dynamic.
3. **Runtime hot-swap (Phase 2)** — planned after Phase 1. Adapters are loaded once and can be
   switched per request without reloading or duplicating the base model, enabling the "one base
   model, many behaviors" use case described above.

## Terminology

- **Adapter**: the small set of trained weights (plus a config file) that make up a LoRA
  fine-tune, distributed separately from the base model.
- **Rank (`r`)**: the size of the low-rank decomposition used by the adapter. Lower rank means
  fewer parameters and a smaller adapter file; higher rank can capture more complex changes to
  behavior.
- **Alpha**: a scaling factor applied to the adapter's contribution. Combined with rank, it
  determines how strongly the adapter's weights influence the base model's output.
- **Target modules**: the specific layers/weight matrices within the base model that the
  adapter modifies. An adapter does not necessarily touch every layer.
- **Base model**: the original, unmodified model an adapter is trained against and applied on
  top of.
- **Merge**: combining an adapter's weights into a copy of the base model's weights, producing
  a single adapted set of weights (Phase 1).
- **Hot-swap**: applying (or switching) an adapter against a resident base model at inference
  time, without merging into a persistent copy of the weights (Phase 2).
- **PEFT**: HuggingFace's "Parameter-Efficient Fine-Tuning" library and adapter format;
  Deliverance's adapter support targets PEFT's standard output format.

## Where This Comes From

Deliverance's adapter support targets the standard HuggingFace PEFT adapter layout: an
`adapter_config.json` file alongside an `adapter_model.safetensors` file. For the deeper
mechanics of LoRA and PEFT — the math behind the low-rank decomposition, how adapters are
trained — see [HuggingFace's PEFT documentation](https://huggingface.co/docs/peft/index),
which this doc deliberately does not try to reproduce.
