# Feature Spec: DMN Idle Tick Throttle

## Summary

Temporarily reduce the background DMN idle tick rate so repeated repository
inspection/tool churn is easier to observe and reason about without saturating
logs or self-state updates.

## Requirements

1. Idle server loops must enforce a minimum interval between admitted DMN ticks.
2. The throttle must live in explicit CPU-side control code so operators can
   inspect and adjust it easily.
3. Foreground request handling and pending tool dispatch must continue to run
   while the DMN tick is throttled.
4. Add or update tests alongside the runtime change.
5. Rebuild and restart the runtime after the patch, then verify the live DMN
   behavior changed.
