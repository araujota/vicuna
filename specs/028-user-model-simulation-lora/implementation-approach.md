# Implementation Approach

## Step 1: Expand User Model Surfaces

Add typed user-preference summaries to the public API and compute them during
self-state postwrite updates using existing lexical and social features.

## Step 2: Add User Personality Adapter

Reuse the runtime adapter factory and Adam update path to create a dedicated
user-personality adapter managed by `llama_active_lora_manager`.

## Step 3: Train Only From User Evictions

Thread a user-role-aware ingestion path so only user-authored evicted spans
reach the user-personality adapter.

## Step 4: Add Scoped Serving Override

Implement a context-level scoped override that detaches or bypasses temporal
runtime layers, attaches the user-personality adapter, and restores the stack
after simulation.

## Step 5: Add DMN Simulation Pass

Extend DMN counterfactual comparison so message-variant candidates can simulate
user reply, apply the simulated event on the counterfactual channel, and score
the resulting self-state delta.

## Step 6: Document and Test

Update docs and add focused tests for archival, adapter updates, stack policy,
and simulation traces.
