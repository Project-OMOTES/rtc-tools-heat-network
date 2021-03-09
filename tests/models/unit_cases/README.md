## WarmingUP Test Cases

This file lists all the test cases from WarmingUP.

### 1a

Geothermal well providing for greenhouse baseloads in a tree network.

```mermaid
graph TD
    S[Geothermal Well] --> D1{Greenhouse}
    S --> D2{Greenhouse}
    S --> D3{Greenhouse}
```

### 2a

Geothermal well and boiler providing for greenhouses and residential areas in a ring network.

```mermaid
graph TD
    S1[Geothermal Well] --- D1{Residental Area} --- S2[Boiler]
    S1 --- D2{Greenhouse} --- S2
    S1 --- D3{Residential area} --- S2
```

### 3a

Geothermal well with storage providing for greenhouses in a tree network.

```mermaid
graph TD
    S[Geothermal Well] --> N( )
    B[Buffer] --- N
    N --> D1{Greenhouse}
    N --> D2{Greenhouse}
    N --> D3{Greenhouse}
```
