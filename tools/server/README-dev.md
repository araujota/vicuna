# Server Development Notes

This server is now the host runtime only.

Retained responsibilities:

- route between `standard` and `experimental`
- send standard traffic to DeepSeek
- relay experimental traffic to RunPod
- own tool exposure and Telegram-facing request construction
- persist experimental transitions, decode traces, and emotive traces
- expose the retained RL data surfaces used by offline training

Removed from this repository:

- local `llama.cpp` inference runtime
- in-repo embedding backends
- upstream inference-engine plumbing

Key files:

- [server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)
- [server-runpod.cpp](/Users/tyleraraujo/vicuna/tools/server/server-runpod.cpp)
- [server-emotive-runtime.cpp](/Users/tyleraraujo/vicuna/tools/server/server-emotive-runtime.cpp)
- [server-runtime-control.cpp](/Users/tyleraraujo/vicuna/tools/server/server-runtime-control.cpp)

Build:

```bash
cmake -S /Users/tyleraraujo/vicuna -B /Users/tyleraraujo/vicuna/build
cmake --build /Users/tyleraraujo/vicuna/build --target llama-server -j8
```
