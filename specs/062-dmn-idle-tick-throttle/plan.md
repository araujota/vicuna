# Plan: DMN Idle Tick Throttle

1. Inspect the server idle loop and existing DMN timing state.
2. Add a minimum idle interval gate ahead of `llama_dmn_tick(...)`.
3. Add a targeted server test that proves idle startup no longer accumulates
   rapid DMN provenance events.
4. Rebuild the runtime, restart the service, and verify the new live cadence.
