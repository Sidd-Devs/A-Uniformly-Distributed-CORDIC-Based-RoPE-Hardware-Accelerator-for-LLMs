<p align="center">
  <h1 align="center">A Uniformly Distributed CORDIC-Based<br>RoPE Hardware Accelerator for LLMs</h1>
</p>

<p align="center">
  <strong>Hardware-Efficient Rotary Positional Encoding Accelerator for Edge AI</strong><br>
  International Institute of Information Technology, Bangalore (IIIT-B)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-UD--CORDIC-blue?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Technology-45nm_CMOS-critical?style=flat-square" alt="Technology">
  <img src="https://img.shields.io/badge/Application-Edge_LLMs-informational?style=flat-square" alt="Application">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" alt="Status">
</p>

---

## Table of Contents

* [Overview](#overview)
* [Motivation](#motivation)
* [Proposed Architectures](#proposed-architectures)
* [Key Features](#key-features)
* [Hardware Architecture](#hardware-architecture)
* [Fixed-Point Quantization](#fixed-point-quantization)
* [ASIC Synthesis Results](#asic-synthesis-results)
* [LLM Validation](#llm-validation)
* [Repository Structure](#repository-structure)
* [Tools and Technology](#tools-and-technology)
* [Authors](#authors)
* [References](#references)
* [License](#license)

---

## Overview

This project presents a hardware-efficient Rotary Positional Encoding (RoPE) accelerator for transformer-based Large Language Models (LLMs) using Uniformly Distributed (UD) CORDIC architectures.

The proposed design eliminates expensive trigonometric computations and removes the conventional CORDIC Z-path control logic by directly extracting rotation directions from binary angle representations. Two optimized architectures are explored:

* Binary UD-CORDIC
* Canonical Signed Digit (CSD) UD-CORDIC

The accelerator is implemented as a fully pipelined fixed-point RTL architecture and synthesized using a 45nm ASIC flow for area, power, and timing evaluation.

---

## Motivation

Rotary Positional Encoding (RoPE) is a critical component in transformer-based LLMs, enabling positional awareness through trigonometric vector rotations.

However, conventional implementations suffer from:

* Large LUT overheads
* High memory bandwidth usage
* Expensive floating-point computations
* High power consumption

This project investigates hardware-efficient alternatives suitable for:

* Edge AI accelerators
* Low-power transformer inference
* Embedded LLM deployment

---

## Proposed Architectures

| Architecture         | Key Idea                                                     |
| :------------------- | :----------------------------------------------------------- |
| **Standard CORDIC**  | Conventional iterative micro-rotation architecture           |
| **Binary UD-CORDIC** | Eliminates Z-path using uniformly distributed angles         |
| **CSD UD-CORDIC**    | Merges adjacent stages using Canonical Signed Digit encoding |

The CSD UD-CORDIC architecture reduces hardware complexity by collapsing two Binary UD stages into a single hardware stage.

---

## Key Features

| Feature                      | Description                                |
| :--------------------------- | :----------------------------------------- |
| **Multiplier-Free Design**   | Shift-and-add based datapath               |
| **Z-Path Elimination**       | Removes iterative angle accumulation logic |
| **Fully Pipelined RTL**      | High-throughput architecture               |
| **Fixed-Point Quantization** | Q(1,F) configurable datapath               |
| **ASIC Implementation**      | 45nm CMOS synthesis flow                   |
| **LLM Validation**           | Perplexity evaluation on modern LLMs       |

---

## Hardware Architecture

### Binary UD-CORDIC

* Uniform angle set: αᵢ = 2⁻ⁱ
* Rotation direction directly extracted from angle bits
* Open-loop feed-forward architecture
* Eliminates iterative control feedback

### CSD UD-CORDIC

* Uses Canonical Signed Digit representation
* Merges consecutive rotation stages
* Reduces datapath depth from N to N/2 stages
* Significantly lowers switching activity and silicon area

---

## Fixed-Point Quantization

The accelerator uses:

* Q(1,F) fixed-point representation
* Configurable fractional precision
* Fully parameterized RTL generators

A precision sweep from F=6 to F=14 was performed to identify the optimal hardware–accuracy tradeoff.

### Optimal Precision Range

* F = 7–9 fractional bits
* Minimal perplexity degradation
* Significant area and power reduction

---

## ASIC Synthesis Results

| Architecture     | Area Reduction | Power Reduction |
| :--------------- | :------------- | :-------------- |
| Binary UD-CORDIC | Up to 12.6%    | 33–37%          |
| CSD UD-CORDIC    | 27.1–31.4%     | 62.3–64.5%      |

### Technology Details

| Parameter       | Value                        |
| :-------------- | :--------------------------- |
| Technology Node | 45nm CMOS                    |
| Supply Voltage  | 1.2V                         |
| Clock Frequency | 500 MHz                      |
| Flow            | Cadence Genus ASIC Synthesis |

---

## LLM Validation

The hardware models were integrated directly into the RoPE computation pipeline of multiple transformer-based LLMs.

### Evaluated Models

* LLaMA-2 7B / 13B
* Mistral-7B
* Falcon-7B
* Gemma-2 9B
* PHI-3
* Qwen2-7B

### Dataset

* WikiText-2

Results show negligible perplexity degradation for F ≥ 7 fractional bits while achieving substantial hardware savings.

---

## Repository Structure

```text
.
├── RTL/
├── Binary_UD_CORDIC/
├── CSD_UD_CORDIC/
├── ASIC_Synthesis/
├── Fixed_Point_Analysis/
├── LLM_Evaluation/
├── Reports/
├── Scripts/
├── Figures/
└── README.md
```

---

## Tools and Technology

| Component          | Details                  |
| :----------------- | :----------------------- |
| RTL Design         | Verilog HDL              |
| Synthesis Tool     | Cadence Genus            |
| Technology Node    | 45nm CMOS                |
| Numerical Format   | Fixed-Point Q(1,F)       |
| Evaluation Dataset | WikiText-2               |
| LLM Frameworks     | HuggingFace Transformers |

---

## Authors

| Name           | Affiliation    |
| :------------- | :------------- |
| Siddhant Deore | IIIT Bangalore |
| Pratham Shetty | IIIT Bangalore |
| Madhav Rao | IIIT Bangalore |

---

## References

* Garrido et al., *Uniformly Distributed CORDIC*
* RoPE-based Transformer Architectures
* IEEE APCCAS 2026 Submission

---

## License

This project is intended for academic and research purposes.

---

<p align="center">
  <sub>© 2026 · IIIT Bangalore · Edge AI Hardware Accelerator Project</sub>
</p>
