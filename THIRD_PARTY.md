# Third-party components

Nothing below is redistributed in this repository; build and setup scripts download it.

| component | license | use |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | MIT | inference engine, fetched and patched at build time. The patch contains a few lines of llama.cpp as context and is MIT-licensed. |
| [Ollama](https://github.com/ollama/ollama) | MIT | optional, fetched by `scripts/setup_ollama.sh` |
| [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) | **CC BY-NC 4.0** | optional, fetched by `scripts/setup_python.sh --reference`; imported by `tools/` and the comparison tests. Not needed to convert or run models. |

Models trained with the LittleBit code are usually published under CC BY-NC 4.0 and remain subject to the license of
their base model. Check before using one commercially.

This is an independent project, not affiliated with or endorsed by Samsung, Ollama or ggml-org.
