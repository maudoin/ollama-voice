# ollama-voice
Plug whisper audio transcription to a local ollama server and ouput tts audio responses

This is just a simple combination of three tools in offline mode:
 - Speech recognition: [whisper](https://github.com/openai/whisper) running local models in offline mode
 - Large Language Mode: [ollama](https://github.com/jmorganca/ollama) running local models in offline mode
 - Offline Text To Speech: [pyttsx3](https://pypi.org/project/pyttsx3/)

## Prerequisites

whisper dependencies are setup to run on GPU so Install Cuda before running `pip install`.

## Running

Install [ollama](https://ollama.ai/) and ensure server is started locally first (in WLS under windows) (e.g. `curl https://ollama.ai/install.sh | sh`)

Download a [whisper](https://github.com/openai/whisper) [model](https://github.com/openai/whisper#available-models-and-languages) and place it in the `whisper` subfolder (e.g. https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)

Configure `assistant.yaml` settings. (It is setup to work in french with ollama [mistral](https://ollama.ai/library/mistral) model by default...)

Run `assistant.py`

Leave `space` key pressed to talk, the AI will interpret the query when you release the key.

## Todo

- Rearrange code base
- Multi threading to overlap tts and speed recognition (ollama is already running remotely in parallel)