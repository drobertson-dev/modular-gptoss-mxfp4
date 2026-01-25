---
title: "MAX CLI Command Reference and Usage Guide"
description: "Documentation for the Modular MAX command line tool, including usage, global options, and available commands for model serving and benchmarking."
---

# MAX CLI Command Reference and Usage Guide

Documentation for the Modular MAX command line tool, including usage, global options, and available commands for model serving and benchmarking.

The `max` command line tool allows you to create an OpenAI-compatible endpoint with a simple `max serve` command. It
also includes a command to benchmark your endpoint using built-in datasets or your own dataset.

To install the `max` CLI, install the `modular` package as shown in the install guide.

## Usage

```shell
max [OPTIONS] COMMAND [ARGS]...
```

## Options

- ### `--log-level `

Set logging level explicitly (ignored if -verbose or -quiet is used).

**Options:**

DEBUG | INFO | WARNING | ERROR

- ### `--version`

Show the MAX version and exit.

## Commands

- `benchmark`: Run benchmark tests on a serving model.
- `encode`: Encode text input into model embeddings.
- `generate`: Generate text using the specified model.
- `list`: List available pipeline configurations and more.
- `serve`: Start a model serving endpoint for inference.
- `warm-cache`: Load and compile the model to prepare caches.
