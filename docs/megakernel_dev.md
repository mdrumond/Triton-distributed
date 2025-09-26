# Mega Kernel Development Guide

## High-Level Mega Kernel Architecture

Triton Distributed's mega kernel (MK) concept encapsulates a complete multi-stage model pipeline in a single, device-resident control loop. An MK bundles the following responsibilities:

* **Persistent scheduler loop** that keeps GPU SMs resident to minimize launch overhead.
* **Task queues** shared across warps/blocks to accept work items representing model operations.
* **Communication fabric hooks** (NVLink, SHMEM, NCCL, etc.) to exchange tensors and metadata with peer devices.
* **Profiling collectors** that record runtime counters, execution latencies, and inter-kernel timings for host-side aggregation.

An MK is structured into three layers:

1. **Front-end task API** — exported to CPU code for enqueuing work items and orchestrating execution sequences.
2. **Device runtime** — persistent cooperative groups managing task dispatch, synchronization, and data movement.
3. **Backend micro-kernels** — per-operation routines (matmul, attention, layer-norm, etc.) invoked through an indirection table within the MK.

## Scheduler Architecture

### Device-Resident Components

* **Persistent Scheduler Kernel:** Runs as a long-lived grid occupying the GPU. Each block owns a portion of the global task queue and executes a cooperative loop: dequeue task → execute → update completion counters.
* **Warp-Level Dispatchers:** Within a block, warps pull subtasks and invoke specialized micro-kernels through encoded op IDs. A single queue entry can front a *bundle* of tiled micro-kernel launches; the code generator lowers each task into an inline branch so composite operators (for example, a fused attention block) can run without relinquishing residency.【F:python/triton_dist/mega_triton_kernel/core/code_generator.py†L22-L118】
* **Device Scoreboarding:** Dependency tracking lives entirely inside the persistent kernel. The dispatcher instantiates `Scoreboard` helpers that gate task execution on GPU-visible fences, update tile-completion state, and ensure downstream tasks observe data hazards without CPU intervention.【F:python/triton_dist/mega_triton_kernel/kernels/task_context.py†L71-L151】【F:python/triton_dist/mega_triton_kernel/core/scheduler.py†L63-L104】
* **Profiling Hooks:** At task boundaries, dispatchers collect timestamps and SM metadata by calling the Triton `Profiler` utility, which stamps start/stop events into a GPU buffer for later decoding.【F:python/triton_dist/mega_triton_kernel/core/code_generator.py†L22-L118】【F:python/triton_dist/tools/profiler/language.py†L20-L120】

### CPU-Resident Components

* **Host Scheduler:** Maintains global execution plans, fills device queues, and monitors completion via host-visible fences. Scoreboard memory is allocated and reset by the host (`ModelBuilder.compile`) but is never polled while the device loop is running; all fine-grained scoreboarding lives with the persistent kernel.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L547-L609】
* **Command Submission:** Uses CUDA streams and cooperative launch APIs to initialize MK grids and to issue control messages (e.g., updates to device-side memory, shared queues).
* **Profiling Aggregator:** Polls device-generated profiling buffers, correlates with host-side events, and exports traces/logs.

The scheduler is bifurcated: persistent scheduling logic runs entirely on the device once launched, while the CPU handles coarse-grained orchestration, queue refills, and teardown.

## Device-Resident Mega Kernels

### Inter-Kernel Communication

Mega kernels communicate through GPU-resident shared memory regions and device-level queues:

* **Global Work Queues:** Allocated in device memory, accessible to all resident MKs via atomic operations. Entries encode task IDs, data pointers, and dependency information that the device-side dispatcher reads directly before scheduling any micro-kernel calls.【F:python/triton_dist/mega_triton_kernel/core/scheduler.py†L28-L104】【F:python/triton_dist/mega_triton_kernel/core/code_generator.py†L22-L118】
* **Peer-to-Peer Buffers:** Tensor-parallel builders allocate NVSHMEM symmetric tensors through the model builder (`create_symm_tensor`) and pass them to communication tasks. The generated all-reduce kernel dereferences peer pointers with `libshmem_device.remote_mc_ptr` or `dl.symm_at` to fetch remote tiles before writing the aggregated slice back into the shared buffer, so dependent ranks observe the update without a host round trip.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L146-L167】【F:python/triton_dist/mega_triton_kernel/kernels/allreduce.py†L29-L80】【F:python/triton_dist/mega_triton_kernel/models/layers/tp_attn.py†L68-L118】
* **Event Flags:** Lightweight device-visible flags or semaphores synchronize producer/consumer MKs without returning to the host.

### Model Development Workflow

Developing a model for MK execution requires translating high-level model graphs into MK-compatible task descriptions:

1. **Model Partitioning:** Identify kernels/operators that can run within the persistent environment. Operators that require host callbacks, in-flight CPU synchronization, or dynamic tensor shapes that violate the contiguous/aligned assumptions enforced by the builders (for example, `check_tensor_dim`, `check_alignment`, `check_tensor_dtype`) must be rewritten or executed outside of the mega kernel.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L520-L569】 Convert each supported operator into a micro-kernel or call into existing library functions.
2. **Task Specification:** For each operator, define a task descriptor containing opcode, tensor metadata, launch parameters, and dependency links.
3. **Kernel-to-Task Conversion:** Use Triton Distributed's lowering pipeline to transform Triton or CUDA kernels into MK callables. The `TaskBuilder` abstraction wraps Triton-generated kernels, but builders can also emit stubs that launch precompiled CUDA micro-kernels through custom codegen blocks registered with `registry.register_task`—the integration cost is limited to providing metadata and a codegen snippet that calls into the cubin or CUDA graph you already own.【F:python/triton_dist/mega_triton_kernel/core/builder.py†L19-L86】【F:python/triton_dist/mega_triton_kernel/core/registry.py†L19-L74】 The pipeline produces:
   * Compiled binary code for Triton-based micro-kernels or hooks into custom CUDA binaries.
   * Metadata (register usage, shared memory requirements) for the dispatcher.
   * Task descriptors mapping model graph nodes to callable slots.
4. **Integration:** Register tasks with the MK runtime by populating the device-side dispatch table and uploading task descriptors to the queue initialization buffer.

### Host-Device Communication Lifecycle

* **Issuing Mega Kernels:** The CPU launches the persistent MK grid via cooperative kernel launch APIs. Initialization parameters include queue pointers, profiling buffer addresses, and dispatch tables.
* **Runtime Communication:** During execution, the CPU does not intervene in the MK inner loop. Communication occurs exclusively through device memory writes/reads:
  * Host enqueues new tasks by writing to queue memory mapped to the GPU.
  * Device signals completion by updating fences or status buffers consumed asynchronously by the host.
* **No Mid-Execution Host Intervention:** Once running, MKs operate autonomously on the device. Host involvement is limited to queue maintenance and final teardown.

### Profiling Data Collection

Device MKs collect profiling data through:

* **Cycle Counters:** Using `clock64()` or equivalent to timestamp task start/finish.
* **Hardware Performance Counters:** The generated kernel writes timestamped events via the Triton `Profiler`. When deeper metrics are needed, host-side collectors can attach CUPTI sampling to the same execution window, but availability depends on GPU/driver support for profiling long-running kernels; fall back to timestamp analysis when CUPTI cannot attach to persistent launches.【F:python/triton_dist/tools/profiler/language.py†L20-L120】【F:python/triton_dist/tools/profiler/viewer.py†L1-L120】
* **Per-Task Logs:** Each task writes summary statistics (latency, occupancy, memory throughput) into ring buffers. The host periodically maps and aggregates these buffers.

## CPU-Resident Code Paths

### Compilation Flow

1. **Front-End Definition:** Users typically author task bodies as Triton kernels (for example the linear, flash attention, and decode tasks under `python/triton_dist/mega_triton_kernel/tasks`). The registry interface can wrap external CUDA call-sites, but the in-tree builders lower Triton code paths today.【F:python/triton_dist/mega_triton_kernel/tasks/linear.py†L1-L247】【F:python/triton_dist/mega_triton_kernel/tasks/flash_attn.py†L1-L210】【F:python/triton_dist/mega_triton_kernel/core/registry.py†L19-L74】
2. **Lowering Pipeline:** The `ModelBuilder` drives task lowering; each registered builder produces task instances and enqueues them through `enque_tasks`, which materializes queue tensors and the scoreboard layout.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L520-L609】【F:python/triton_dist/mega_triton_kernel/core/scheduler.py†L28-L104】 Triton kernels JIT-compile to PTX/SASS on first launch as usual.
3. **Emitter Kernel Generation:** `CodeGenerator.generate_code` stitches all task branches into a single Triton mega kernel source file and imports it dynamically, providing the device control loop that calls each micro-kernel in-line rather than linking through a separate binary. No CUDA Graph compilation is required—the output is a persistent Triton kernel that the runtime launches directly. Resource usage is capped by host-provided launch metadata: `ModelBuilder` exposes a `num_warps` knob and honors the `MEGAKERNEL_MAX_NUM_STAGES` environment variable, while task builders such as the linear implementation clamp per-operator pipeline stages to respect shared-memory and staging limits reported by `DeviceProp` before the mega kernel is generated.【F:python/triton_dist/mega_triton_kernel/core/code_generator.py†L1-L118】【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L88-L144】【F:python/triton_dist/mega_triton_kernel/tasks/linear.py†L150-L219】 The environment knob does **not** alter how many devices participate in a model replica; instead it caps the Triton software-pipelining depth (the number of stages kept in flight per kernel) that builders are allowed to request. When `MEGAKERNEL_MAX_NUM_STAGES` is set, the parsed integer propagates into `DeviceProp.MAX_PIPELINE_STAGES`, and helpers such as `LinearTaskBuilder._adjust_num_stages` clamp each task's `NUM_STAGES` to that ceiling before code generation, preventing excessive register and shared-memory pressure on GPUs with limited resources.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L103-L133】【F:python/triton_dist/mega_triton_kernel/tasks/linear.py†L169-L209】
4. **Packaging:** The host builds a launch package containing the generated kernel callable, dispatch tables, queue initialization data, and profiling buffer layouts.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L547-L609】
5. **Deployment:** Package is transferred to the target device and launched via the scheduler.

### Profiling Flow

1. **Device Logging:** MK writes profiling records to device memory buffers (structured as circular buffers or per-block logs).
2. **Host Collection:** CPU threads periodically map/copy the profiling buffers (e.g., through `ProfilerBuffer` or manual tensor copies) using asynchronous CUDA memcpys or unified memory reads.【F:python/triton_dist/tools/profiler/context.py†L18-L63】
3. **Aggregation:** The host-side profiler decodes buffer entries with `export_to_perfetto_trace`, correlates task IDs with timestamps, and exports traces consumable by Perfetto/Chrome trace viewers. The model builder wires this path through `dump_trace`, while `ProfilerBuffer` drives the same export when used as a context manager.【F:python/triton_dist/tools/profiler/viewer.py†L1-L120】【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L560-L576】【F:python/triton_dist/tools/profiler/context.py†L28-L63】
4. **Feedback:** Aggregated profiling data feeds back into autotuners or scheduling heuristics.

## Dependencies within Triton Distributed

Mega kernel infrastructure relies on multiple components across the project:

* **`python/` Front-End:** Provides APIs for model authors to define tasks, configure MK schedulers, and trigger compilation—the `ModelBuilder` in `python/triton_dist/mega_triton_kernel/models/model_builder.py` is the primary entry point.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L1-L609】
* **`python/triton_dist/mega_triton_kernel/core/`:** Houses the scheduler, task builders, registry, and code generator that assemble the mega kernel.【F:python/triton_dist/mega_triton_kernel/core/scheduler.py†L1-L104】【F:python/triton_dist/mega_triton_kernel/core/builder.py†L1-L86】【F:python/triton_dist/mega_triton_kernel/core/code_generator.py†L1-L118】
* **`python/triton_dist/mega_triton_kernel/kernels/`:** Implements device micro-kernels and synchronization primitives such as the scoreboarding helpers.【F:python/triton_dist/mega_triton_kernel/kernels/task_context.py†L1-L151】
* **`python/triton_dist/tools/profiler/`:** Provides device-side instrumentation utilities and host decoders used by mega kernel profiling flows.【F:python/triton_dist/tools/profiler/language.py†L20-L120】【F:python/triton_dist/tools/profiler/viewer.py†L1-L120】
* **`shmem/` and `asset/`:** Supply NVSHMEM bindings and communication utilities for inter-device data exchange.
* **`tutorials/` and `docs/getting-started/megakernel/`:** Include runnable examples (`model_server.py`, `bench_qwen3.py`) that demonstrate end-to-end model deployment on the mega kernel runtime.【F:python/triton_dist/mega_triton_kernel/test/models/model_server.py†L1-L160】【F:docs/getting-started/megakernel/megakernel.md†L1-L80】

## Qwen3 Mega Kernel Implementation

The reference Qwen3 integration illustrates how an existing PyTorch model is adapted for the mega kernel runtime by pushing layer execution into the persistent task pipeline.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L57-L198】 Key changes include:

* **Layer builders wrap Triton tasks:** `Qwen3LayerBuilder` replaces the eager `Qwen3DecoderLayer` forward path with calls to tensor-parallel attention/MLP builders. Each builder emits MK tasks (RMSNorm, attention, fused matmuls) instead of launching standalone kernels, ensuring the generated mega kernel contains the fused control flow.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L57-L139】
* **Persistent buffers and KV cache ownership:** The model allocates device-resident hidden state buffers and a `PagedKVCache` so the mega kernel can read/write state without CPU interaction, mirroring the expectations of the device scheduler.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L100-L134】
* **Builder-driven forward graph:** `build_fwd` stitches layer builders together, invoking `ModelBuilder` helpers (`make_rms_norm`, `make_add`, `make_linear`) to enqueue tasks that will execute inside the persistent loop rather than calling PyTorch ops on the host.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L118-L180】
* **Mega kernel runtime entry point:** `mega_forward` copies inputs into the persistent buffer and calls `ModelBuilder.run()` to trigger the resident mega kernel, while retaining a functional fallback that reuses the emitted buffers for post-processing. This demonstrates how inference entry points shift from per-request kernel launches to reusing a compiled MK instance.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L181-L200】
* **Parallelism configuration:** Tensor-parallel builders accept rank and world-size arguments, allocate NVSHMEM-backed output shards, and call `make_allreduce` to merge results across devices, enabling pipelined execution across GPUs without extra host APIs. The Qwen3 attention builder shows the pattern by sharding projection weights and issuing an all-reduce through the mega kernel tasks.【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L64-L123】【F:python/triton_dist/mega_triton_kernel/models/layers/tp_attn.py†L32-L118】 Pipeline parallel building blocks (NVSHMEM-backed P2P queues and handshakes) live in `triton_dist.layers.nvidia.p2p.CommOp`, which provides the stage-to-stage rendezvous used by non-mega-kernel runtimes; the in-tree Qwen3 mega kernel currently runs the entire decoder stack per rank in a single stage (see the sequential layer loop in `Qwen3Model.build_fwd`), so pipeline partitioning has not yet been wired into the device-resident scheduler.【F:python/triton_dist/layers/nvidia/p2p.py†L34-L140】【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L134-L180】

## Code Entry Points and Examples

* **Building models:** `python/triton_dist/mega_triton_kernel/models/model_builder.py` exposes `ModelBuilder`, which assembles graph tasks, allocates scoreboards, and launches the generated mega kernel. The Qwen3 reference model illustrates how to wire attention, MLP, and communication operators together.【F:python/triton_dist/mega_triton_kernel/models/model_builder.py†L1-L609】【F:python/triton_dist/mega_triton_kernel/models/qwen3.py†L1-L200】
* **Implementing new tasks:** Extend `TaskBuilderBase` and register it via `registry.register_task` to plug in Triton or CUDA micro-kernels. Examples include the linear and attention tasks under `python/triton_dist/mega_triton_kernel/tasks/` which emit code fragments consumed by the generated mega kernel.【F:python/triton_dist/mega_triton_kernel/core/builder.py†L1-L86】【F:python/triton_dist/mega_triton_kernel/core/registry.py†L19-L74】【F:python/triton_dist/mega_triton_kernel/tasks/linear.py†L70-L150】
* **Runtime usage:** The test model server (`python/triton_dist/mega_triton_kernel/test/models/model_server.py`) shows how to instantiate a `ModelBuilder`, compile the mega kernel, and serve requests inside Triton Distributed, while unit tests under `python/triton_dist/mega_triton_kernel/test/ops/` provide operator-level validation patterns.【F:python/triton_dist/mega_triton_kernel/test/models/model_server.py†L1-L160】【F:python/triton_dist/mega_triton_kernel/test/ops/test_flash_attn.py†L1-L160】

This document should be maintained in tandem with compiler, runtime, and communication subsystem updates to ensure accuracy across the Triton Distributed stack.
