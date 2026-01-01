# nn.pl - Formal Specifications and Technical Architecture

## 📋 Quick Start

This repository contains a neural network library with implementations in **Lua/Torch**, **Prolog**, and **C**. The `docs/` directory contains comprehensive formal specifications and architectural documentation.

**Choose your path:**
- 🏗️ **Architecture Overview**: Start with [`docs/architecture_overview.md`](docs/architecture_overview.md) for visual diagrams and system design
- 📐 **Formal Specifications**: Dive into Z++ specs starting with [`docs/data_model.zpp`](docs/data_model.zpp)
- 📖 **Reading Guide**: See [`docs/README.md`](docs/README.md) for a complete guide

## 📚 Documentation Overview

### Complete Documentation Suite (184 KB total)

| Document | Size | Purpose | Start Here If... |
|----------|------|---------|------------------|
| [**architecture_overview.md**](docs/architecture_overview.md) | 16.6 KB | Visual architecture with 20+ Mermaid diagrams | You want the big picture |
| [**data_model.zpp**](docs/data_model.zpp) | 36.0 KB | Formal data structure specifications | You need precise type definitions |
| [**system_state.zpp**](docs/system_state.zpp) | 31.7 KB | System state and global invariants | You're tracking state consistency |
| [**operations.zpp**](docs/operations.zpp) | 52.5 KB | Operational semantics (forward/backward/training) | You're implementing algorithms |
| [**integrations.zpp**](docs/integrations.zpp) | 37.3 KB | External interfaces and contracts | You're integrating with the system |
| [**README.md**](docs/README.md) | 10.1 KB | Documentation guide and Z++ reference | You're new to the docs |

## 🎯 What's Included

### Architecture Documentation
- **System Architecture**: Multi-layer, multi-language design
- **Component Diagrams**: Module hierarchy, containers, layers
- **Data Flow**: Training loop, forward/backward propagation
- **State Machines**: Module lifecycle and mode transitions
- **Integration Points**: External APIs and boundaries
- **20+ Mermaid Diagrams**: Class diagrams, sequence diagrams, flowcharts, state diagrams

### Formal Z++ Specifications

#### 1. Data Model (36 KB)
```
📦 Basic Types (Real, Natural, Integer, Boolean)
├── 📊 Tensors (Shape, Data, Operations)
├── ⚙️ Parameters (Learnable tensors with gradients)
├── 🧩 Modules (Base abstraction)
│   ├── Containers (Sequential, Parallel, Concat)
│   ├── Simple Layers (Linear, Reshape, Mean, Max)
│   ├── Transfer Functions (Sigmoid, Tanh, ReLU, Softmax)
│   ├── Loss Criterions (MSE, NLL, BCE, L1)
│   ├── Convolutional Layers (Spatial, Temporal)
│   └── Table Layers (Split, Join, Select)
└── ✓ Invariants (Shape compatibility, value ranges)
```

#### 2. System State (32 KB)
```
🌐 System State
├── 📋 Module Registry (Unique IDs, centralized management)
├── 🌳 Network Topology (DAG structure, parent-child)
├── 🎓 Training State (Dataset, batching, epochs, optimizer)
├── 💾 Computation State (Forward/backward caches)
├── 🔄 Mode Transitions (Training ↔ Evaluation)
└── ✓ Global Invariants (Consistency across all components)
```

#### 3. Operations (53 KB)
```
⚡ Neural Network Operations
├── ➡️ Forward Propagation (Input → Output)
│   ├── Module-specific forwards (Linear, Sigmoid, ReLU, etc.)
│   ├── Sequential forward (chaining)
│   └── Complete forward pass
├── 📉 Loss Computation (MSE, ClassNLL, BCE)
├── ⬅️ Backward Propagation (Gradient computation)
│   ├── Module-specific backwards (gradients)
│   ├── Sequential backward (reverse chaining)
│   └── Complete backward pass
├── 🔄 Parameter Updates (SGD, Momentum)
├── 🎓 Training Loop (Batch → Epoch → Validation)
└── 🔮 Inference (Prediction without gradients)
```

#### 4. Integration Contracts (37 KB)
```
🔌 External Interfaces
├── 🔢 Tensor Operations (Add, multiply, reshape, slice, concat)
├── 📐 Math Functions (Activations with derivatives)
├── 💾 Serialization (JSON, Binary, Lua, Prolog)
├── 📁 File I/O (Save/load models and checkpoints)
├── 📊 Dataset Operations (Load, shuffle, split)
├── ⚠️ Numerical Stability (NaN/Inf detection, gradient clipping)
└── 🚨 Error Handling (Result types, error propagation)
```

## 🎓 Learning Path

### Beginners (30 minutes)
1. Read [`docs/architecture_overview.md`](docs/architecture_overview.md) sections 1-3
2. Look at the Mermaid diagrams
3. Skim the summaries in each `.zpp` file

### Intermediate (2-3 hours)
1. Study all diagrams in `architecture_overview.md`
2. Read [`data_model.zpp`](docs/data_model.zpp) sections 1-5
3. Read [`system_state.zpp`](docs/system_state.zpp) sections 1-2, 5
4. Read [`operations.zpp`](docs/operations.zpp) sections 1-3

### Advanced (1-2 days)
1. Complete reading of all specifications
2. Trace through operation sequences manually
3. Verify invariants with sample data
4. Study integration contracts for your use case

## 🔑 Key Features

### Formal Specifications
- ✅ **Rigorous**: Z++ formal notation with complete semantics
- ✅ **Verifiable**: All invariants explicitly stated
- ✅ **Modular**: Clear dependencies between specifications
- ✅ **Complete**: Covers data, state, operations, and integrations

### Architecture Documentation
- 📊 **Visual**: 20+ Mermaid diagrams
- 🎯 **Practical**: Design patterns, tech stack, performance tips
- 🔒 **Secure**: Security considerations documented
- 🚀 **Future-proof**: Evolution roadmap included

## 🛠️ Use Cases

### For Developers
- **Implementing features**: Consult `operations.zpp` for precise semantics
- **Adding modules**: Follow patterns in `data_model.zpp`
- **Debugging**: Verify invariants from `system_state.zpp`
- **Integration**: Use contracts in `integrations.zpp`

### For Researchers
- **Understanding design**: Study `architecture_overview.md`
- **Formal verification**: Use Z++ specs for theorem proving
- **Property checking**: Validate invariants hold
- **Algorithm analysis**: Trace operations through specs

### For QA/Testing
- **Test generation**: Derive test cases from pre/post-conditions
- **Coverage**: Ensure all operations tested
- **Validation**: Check invariants during testing
- **Error handling**: Verify error contracts

## 📊 Specification Statistics

```
Total Documentation:     184 KB
Specification Files:     4 files (157 KB)
Architecture Docs:       1 file (17 KB)
Guide & Index:           2 files (10 KB)

Schemas Defined:         150+
Invariants Specified:    200+
Operations Formalized:   40+
Diagrams Created:        20+
```

## 🔗 Navigation

### By Topic
- **Tensors**: [`data_model.zpp`](docs/data_model.zpp) Section 2
- **Modules**: [`data_model.zpp`](docs/data_model.zpp) Sections 4-10
- **Training**: [`system_state.zpp`](docs/system_state.zpp) Section 3, [`operations.zpp`](docs/operations.zpp) Section 5
- **Forward/Backward**: [`operations.zpp`](docs/operations.zpp) Sections 1-3
- **Serialization**: [`integrations.zpp`](docs/integrations.zpp) Section 3-4

### By Activity
- **Learning the system**: [`architecture_overview.md`](docs/architecture_overview.md) → [`docs/README.md`](docs/README.md)
- **Implementing**: [`operations.zpp`](docs/operations.zpp) → [`data_model.zpp`](docs/data_model.zpp)
- **Integrating**: [`integrations.zpp`](docs/integrations.zpp) → [`architecture_overview.md`](docs/architecture_overview.md)
- **Verifying**: [`system_state.zpp`](docs/system_state.zpp) → [`operations.zpp`](docs/operations.zpp)

## 📖 Z++ Notation Quick Reference

| Symbol | Meaning | Symbol | Meaning |
|--------|---------|--------|---------|
| `ℕ` | Natural numbers | `∀` | For all |
| `ℤ` | Integers | `∃` | Exists |
| `ℝ` | Real numbers | `∧` | And |
| `𝔹` | Booleans | `∨` | Or |
| `seq T` | Sequence | `⇒` | Implies |
| `A ⇸ B` | Partial function | `Δ` | State change |
| `A → B` | Total function | `Ξ` | Read-only |
| `#S` | Cardinality | `?` | Input |
| `∈` | Element of | `!` | Output |
| `⊆` | Subset | `'` | After state |

See [`docs/README.md`](docs/README.md) for complete notation guide.

## 🤝 Contributing

When contributing:
1. **Update specs first**: Changes should be reflected in formal specifications
2. **Maintain consistency**: Keep all documents synchronized
3. **Verify invariants**: Ensure global invariants still hold
4. **Update diagrams**: Keep Mermaid diagrams current
5. **Add examples**: Include concrete examples when helpful

## 📜 License

See `COPYRIGHT.txt` in the repository root.

## 🔍 Related Documentation

- **README_PROLOG.md**: Prolog implementation guide
- **IMPLEMENTATION_SUMMARY.md**: Implementation details
- **CONTRIBUTING.md**: Contribution guidelines
- **doc/**: Original Torch/nn documentation

---

**Last Updated**: 2025-12-31  
**Specification Version**: 1.0  
**Authors**: Repository contributors and formal methods team

For questions or clarifications, please open an issue on GitHub.
