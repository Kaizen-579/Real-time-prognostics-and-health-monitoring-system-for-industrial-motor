# Real-Time Prognostics and Health Monitoring System for Industrial Motor Using Hybrid Analytics

## Overview

This project presents a Real-Time Prognostics and Health Monitoring (PHM) system for industrial motors. The system continuously monitors motor operating conditions, detects anomalies, evaluates motor health, estimates efficiency, and predicts Remaining Useful Life (RUL) using a combination of physics-based modeling and machine learning techniques.

The objective is to move beyond traditional reactive and preventive maintenance strategies by enabling predictive maintenance through real-time analytics.

---

## Features

- Real-time motor condition monitoring
- Adaptive baseline estimation
- Health Index calculation
- Isolation Forest based anomaly detection
- Rule-based fault diagnosis
- Efficiency estimation
- Stress and degradation modeling
- Remaining Useful Life (RUL) prediction
- MATLAB Simulink and Python integration
- UDP-based real-time communication
- Dashboard visualization

---

## Problem Statement

Industrial motors are exposed to:

- Thermal stress
- Electrical loading
- Mechanical wear

Traditional maintenance approaches:

- Detect faults only after failure
- Follow fixed maintenance schedules
- Ignore actual equipment condition

This project addresses these limitations through continuous health assessment and predictive analytics.

---

## System Architecture

```text
Industrial Motor Model (MATLAB Simulink)
                │
                ▼
      Real-Time Parameter Extraction
                │
                ▼
          UDP Communication
                │
                ▼
       Python Analytics Engine
                │
 ┌──────────────┼──────────────┐
 ▼              ▼              ▼
Health      Anomaly       Efficiency
Index       Detection     Estimation
Calculation
 │
 ▼
Stress & Damage Modeling
 │
 ▼
RUL Prediction
 │
 ▼
Dashboard Visualization
