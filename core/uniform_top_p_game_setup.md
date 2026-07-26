# Stop Generating Ten Answers To Pick One

`uniform_top_p` exists because local games and tools often need one random plausible answer. They do not need a ranked list of answers.

## The Problem

In Dead to Rights, the game setup needs varied but ordinary details:

- a suspect name
- a place
- an item
- a small theft case

Before `uniform_top_p`, the game asked the model for a guided JSON list of options, then Java picked one.

The old flow looked like this:

```text
Prompt: Generate exactly 10 grounded ordinary present-day places where a small theft mystery could happen.
Model: returns a JSON array with 10 places.
Java: randomly picks one place.
Java: discards the other 9 places.
```

That is easy application code, but it makes the model do unnecessary work. The model has to decode the full list, including every discarded answer and the JSON array syntax.

For local inference, generated tokens are the expensive part. If the application only needs one place, paying to generate ten places is wasted time.

## The New Flow

`uniform_top_p` moves the random choice into the sampler.

The new flow is:

```text
Prompt: Generate one grounded ordinary present-day place where a small theft mystery could happen.
Sampler: build a nucleus with top_p=0.95.
Sampler: pick uniformly from that plausible candidate set.
Model: returns one JSON field.
```

The model now decodes one answer instead of a list of answers that user code later throws away.

## Measured Run

This was measured with a temporary embedded Java main program outside the repository. The program loaded the real local inference engine and cached model:

```text
model=edwardcapriolo/Qwen3-0.6B-JQ4
```

The old path used guided JSON for a list of 10 places, then represented the current Java behavior of picking one from that list.

The new path used guided JSON for one place with `top_p=0.95` and `uniform_top_p=1.0`.

### Old Flow Output

```text
=== Old flow: guided_json list of 10, then Java picks one ===
wall_ms=26335
prompt_tokens=58
generated_tokens=94
prompt_time_ms=2731
generate_time_ms=26332
finish_reason=STOP_TOKEN
output={ "places": [ "The grocery store in downtown New York", "The coffee shop in the city park", "The library on a quiet street", "The bakery in a small town", "The park with a fountain and a cafe", "The bookstore in a quiet alley", "The cafe in a historic building", "The library in a small building", "The bakery in a market area", "The park with a fountain and a cafe" ] }
```

The model generated 10 places. The application only needed one.

### New Flow Output

```text
=== New flow: one answer with uniform_top_p in sampler ===
wall_ms=4627
prompt_tokens=55
generated_tokens=14
prompt_time_ms=1665
generate_time_ms=4627
finish_reason=STOP_TOKEN
output={ "place": "Cambridge, MA, USA" }
```

The model generated one place. The sampler handled the randomness.

## Result

```text
RESULT speedup=5.69x saved_ms=21708 saved_generated_tokens=80
```

For this run:

- old list flow: `26.335 s`
- new sampler flow: `4.627 s`
- time saved: `21.708 s`
- generated tokens avoided: `80`
- speedup: `5.69x`

The exact number will vary by model, hardware, prompt, schema, and output length. The important point is not that every run is exactly `5.69x` faster. The important point is that the old design asks the model to generate many alternatives in text, then throws most of them away. `uniform_top_p` lets the inference engine make the random plausible choice while generating one answer.

## When To Use It

Use `uniform_top_p` when:

- the application needs one random plausible choice
- variety matters more than picking the highest-probability answer
- you would otherwise ask the model for a list and pick one in code
- the output is setup data, not an ongoing conversation

Good examples:

- game places
- suspect names
- ordinary objects
- random scenario seeds
- short creative setup facts

Avoid it when:

- the model should give the most likely answer
- conversation consistency matters
- interrogation or agent behavior should stay stable
- you already need the whole list for display

## API Shape

Use `uniform_top_p` with `top_p`:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "uniform_top_p": 1.0
}
```

`top_p` defines the nucleus. `uniform_top_p` changes the final draw from weighted to uniform:

- `top_p`: weighted draw from the nucleus
- `uniform_top_p`: uniform draw from the nucleus

`top_k` can still be used as a safety cap before `top_p` is applied.
