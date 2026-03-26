# Quickstart: Runtime Latency Hot Path Reduction

1. Build the server:
   - `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
2. Run provider tests:
   - `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q`
3. Run bridge tests:
   - `node --test /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs`
4. Verify the runtime still reports the expected provider defaults and the
   bridge keeps serving Telegram turns with the thinner transport path.
