# ELI5: Thermodynamic Constraints in Transformers

*An intuitive explanation of the main findings.*

**Paper:** [Thermodynamic Constraints in Transformer Architectures](https://doi.org/10.5281/zenodo.18165365)

---

## The Amplifier Chain

Imagine a long chain of audio amplifiers connected in series. A signal passes through the first, then the second, and so on.

Now imagine two versions of this setup:

**Version A (The Dampeners):** Each amplifier is set to 99% volume. The signal gets slightly quieter at every step. After 32 steps, it's calm and controlled.

**Version B (The Shouters):** Each amplifier is set to 101% volume. The signal gets slightly louder at every step. After 32 steps, it's roaring.

Here's the thing: **transformers work exactly like this.**

---

## What We Found

We looked inside 23 different AI models — Pythia, LLaMA, GPT-2, Mistral, and others — and measured the "volume" of information as it flows through the layers.

The result surprised us.

Some model families are **whisperers**. Each layer makes the signal a bit quieter (Gain < 1). These models gradually suppress energy to stay stable.

Other model families are **shouters**. Each layer makes the signal a bit louder (Gain > 1). These models actively amplify information as it flows deeper.

And here's the kicker: **this has almost nothing to do with model size.**

A tiny 160 million parameter model and a massive 12 billion parameter model from the same lab behave almost identically. But two models of the exact same size from different labs can behave completely opposite.

---

## The Family Tree

So who are the whisperers and who are the shouters?

**The Whisperers (Dampening):**
- Pythia (EleutherAI)
- GPT-NeoX (EleutherAI)

**The Shouters (Amplifying):**
- LLaMA (Meta)
- OPT (Meta)
- GPT-2 (OpenAI)

It's not random. It's determined by *who trained the model* — their data recipes, their optimizer settings, their training choices. We call this **training heritage**.

---

## Why Deep Models Can't Shout

Now, if you're building a very deep model — say, 80 layers instead of 32 — you have a problem.

If each layer amplifies even a tiny bit, the signal explodes. It's like compound interest or audio feedback: 1.05 × 1.05 × 1.05... repeated 80 times becomes enormous.

So deep models are *forced* to be more neutral. They can't amplify much per layer, or everything blows up.

We found this follows a precise mathematical law:

> The maximum amplification per layer decreases as models get deeper.

It's eerily similar to Kleiber's Law in biology — how metabolic rate slows as animals get larger to prevent overheating. Transformers, it seems, have their own "physics."

---

## Why This Matters for Fine-Tuning

Here's where it gets practical.

When you fine-tune a model — with LoRA, RLHF, or any other technique — you're adjusting the weights after pre-training.

You can turn the volume knob up or down a bit. But you **cannot** change whether the system was built as a dampener or an amplifier.

A whisperer will always tend to dampen. A shouter will always tend to amplify.

This explains something practitioners have felt for years: some models feel "stiff" and resist being pushed away from their base behavior. Others feel "responsive" or even unstable.

It's not your fault. It's not your hyperparameters. It's baked into the model's DNA.

---

## The Knob Inside the Model

Where does this behavior come from?

Inside every attention layer, there are two key weight matrices:
- **W_V** controls how much information is extracted
- **W_O** controls how strongly it's written back

The ratio between them — `||W_V|| / ||W_O||` — acts like the gain knob on an amplifier.

When we measured this across different labs, we found **10× differences**. That's not noise. That's a fundamental design choice, whether intentional or not.

---

## The Bottom Line

We discovered a hierarchy:

> **Heritage > Geometry > Scale**

Who trained the model matters more than its architecture. Architecture matters more than raw parameter count.

Training recipes leave permanent fingerprints that fine-tuning cannot erase.

---

*For the full technical details, see the [paper](https://doi.org/10.5281/zenodo.18165365) and [notebooks](notebooks/).*
