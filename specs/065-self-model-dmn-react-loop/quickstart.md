# Quickstart: Self-Model-Translated DMN ReAct Loop

## 1. Review the current baseline

Inspect the current DMN versus active-loop split before implementation:

```bash
sed -n '545,620p' ARCHITECTURE.md
sed -n '4010,4255p' src/llama-cognitive-loop.cpp
sed -n '5080,5345p' src/llama-cognitive-loop.cpp
```

Confirm the existing DMN still relies on endogenous seed assembly and
winner-action scoring before starting code changes.

## 2. Build targeted runtime binaries

Use the normal project build flow from `docs/build.md`, then build the runtime
and targeted tests:

```bash
cmake -B build
cmake --build build --config Release -j 8
```

If a narrower local loop is needed while iterating on the feature, rebuild the
affected tests and runtime targets after each change.

## 3. Run targeted validation

After implementation, run the native tests covering the new translation and DMN
lifecycle:

```bash
ctest --test-dir build --output-on-failure -R "test-cognitive-loop|test-self-state"
```

If local target names differ in the active build tree, run the corresponding
test binaries directly from `build/`.

## 4. Validate live runtime behavior

After rebuilding the runtime on a host:

```bash
curl http://127.0.0.1:8080/health
journalctl --user -u vicuna-runtime.service --no-pager -n 200
```

Confirm all of the following in the logs or traces:

- a self-model revision produced a new DMN prompt revision
- a DMN episode started from that prompt revision
- tool observations can supersede the current DMN episode through a newer prompt
  revision
- DMN-origin Telegram relay appears as a tool action rather than a background
  emit winner action

## 5. Validate Telegram relay semantics

With the bridge running, trigger a DMN-origin relay and verify:

```bash
journalctl -u vicuna-telegram-bridge.service --no-pager -n 200
```

Check that:

- the relay is recorded as a tool request/result
- active engagement response accounting does not advance because of the relay
- DMN cognition continues after delivery or failure
