# LLM Agent Frameworks for Geospatial Data Visualization

This repository contains implementations of three different LLM agent frameworks (ReAct, MRKL, and Chain-of-Thought) applied to the task of creating a choropleth map of US population density using census data.

It is still a work in progeess!!!!

## Overview

Modern LLM agent frameworks offer different approaches to organizing complex tasks that require multiple steps of reasoning and execution. This project implements and compares three prominent frameworks:

1. **Chain-of-Thought**
2. **ReAct** (Reasoning + Acting)
3. **MRKL** (Modular Reasoning, Knowledge and Language) 

All three frameworks tackle the same task: fetching US Census data, retrieving geographic boundary information, calculating population density, and creating a choropleth visualization.

## Framework Comparison

### ReAct Framework

ReAct combines reasoning traces with task-specific actions in an iterative loop of thinking, planning, executing, and observing.

```
Think → Act → Observe → Think → ...
```

**Key characteristics:**
- Explicit reasoning steps using the `think()` method
- Iterative observation-action loop
- Tools mapped to specific capabilities
- LLM-guided decision making between steps
- Adapts based on results of previous actions

The ReAct agent is particularly good at handling tasks with potential errors or unexpected results because it can observe outcomes and adjust its approach.

### MRKL Framework

MRKL integrates neural and symbolic modules into a cohesive system where each module specializes in a specific type of task.

```
Router → Module 1 → Module 2 → ... → Module N
```

**Key characteristics:**
- Router determines module execution sequence
- Specialized modules for different task components
- Working memory maintains state across modules
- Domain-specific LLM consultations within modules
- Highly modular and compartmentalized design

The MRKL approach shines when dealing with tasks that require different types of specialized knowledge or capabilities.

### Chain-of-Thought Framework

Chain-of-Thought focuses on explicit intermediate reasoning steps to solve complex problems step by step.

```
Task Decomposition → Step 1 (Reason + Execute) → Step 2 (Reason + Execute) → ...
```

**Key characteristics:**
- Initial task decomposition with LLM
- Explicit reasoning before each execution step
- Records complete reasoning chain
- Sequential, linear approach
- Step-specific LLM consultations

Chain-of-Thought excels at problems requiring clear, logical progression through predefined steps, with strong emphasis on explainability.

## Implementation Details

All three frameworks:
1. Use Claude 3.7 Sonnet for guidance and reasoning
2. Fetch data from the US Census API
3. Retrieve geographic boundaries from Census TIGER/Line files
4. Calculate population density (people per square km)
5. Create a choropleth visualization

### Project Structure

```
├── react_agent.py          # ReAct framework implementation
├── mrkl_agent.py           # MRKL framework implementation
├── cot_agent.py            # Chain-of-Thought implementation
├── requirements.txt        # Project dependencies
├── cot.log     # Print of the Chain-of-Thought Agent
├── react.log   # Print of ReAct Agent 
├── mrkl.log   # Print of MRKL Agent 
├── us_population_density_react.png  # ReAct output map
├── us_population_density_mrkl.png   # MRKL output map
└── us_population_density_cot.png    # Chain-of-Thought output map

### Technology Stack

- **Data Acquisition**: requests
- **Data Processing**: pandas, geopandas
- **Visualization**: matplotlib
- **LLM Integration**: anthropic Python SDK

## Key Differences

| Feature | ReAct | MRKL | Chain-of-Thought |
|---------|-------|------|-----------------|
| **Structure** | Iterative | Modular | Sequential |
| **Decision Flow** | Think-Act-Observe Loop | Module Routing | Predefined Steps |
| **LLM Usage** | Action Selection | Routing + Module Guidance | Task Decomposition + Step Reasoning |
| **Best For** | Uncertain Tasks | Multi-domain Tasks | Structured Problems |
| **Adaptability** | High | Medium | Low |
| **Specialization** | Medium | High | Medium |
| **Explainability** | Good | Good | Excellent |
| **Extensibility** | Medium | High | Low |

## Practical Applications

These frameworks can be adapted for various LLM-driven data tasks:

- **Data analysis workflows**
- **Multi-step research tasks**
- **Content creation pipelines**
- **Decision making systems**
- **Automated reporting**

## Running the Agents

Each agent is implemented as a standalone Python file that can be run independently:

```bash
python react_agent.py &> react.log &
python mrkl_agent.py &> mrkl.log &
python cot_agent.py &> cot.log &
```

You'll need to set the following environment variables:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `CENSUS_API_KEY`: US Census API key (or use 'demo_key' for testing)

### Comparing the Agents

This repository includes a comprehensive agent comparison module that allows you to:
1. Run all three agents and collect performance metrics
2. Compare execution time, success rates, and reasoning approaches
3. Visualize the differences between frameworks
4. Analyze the output maps side-by-side

To run the comparison (not yet implemented):

```bash
python agent_comparison.py
```

Or to analyze existing results without re-running the agents:

```bash
python agent_comparison.py --analysis-only
```

The comparison module provides:
- Performance metrics tables
- Framework comparison analysis
- Visualizations of execution time and reasoning steps
- Side-by-side comparison of output maps
- Persistent storage of results for future reference

## Agent Comparison Analysis (Not Yet Implemented)

The included `agent_comparison.py` module provides detailed analysis of how these frameworks differ:

```
========================================================================================
FRAMEWORK SIMILARITIES AND DIFFERENCES
========================================================================================

Framework Comparison:
+-------------------+--------------------------------+--------------------------------+------------------------------------+
| Aspect            | ReAct                          | MRKL                           | Chain-of-Thought                   |
+===================+================================+================================+====================================+
| Reasoning Approach | Iterative Think-Act-Observe    | Modular with specialized       | Sequential reasoning with explicit |
|                   | loop                           | components                     | steps                              |
+-------------------+--------------------------------+--------------------------------+------------------------------------+
| Task Decomposition | Dynamic, based on current      | Router determines module       | Upfront decomposition before       |
|                   | state                          | sequence                       | execution                          |
+-------------------+--------------------------------+--------------------------------+------------------------------------+
| Error Handling    | Adaptive through observation   | Module-specific error handling | Sequential, stops on first error   |
|                   | loop                           |                                |                                    |
+-------------------+--------------------------------+--------------------------------+------------------------------------+
| LLM Integration   | Per-action guidance            | Router and module-specific     | Task decomposition and step        |
|                   |                                | guidance                       | reasoning                          |
+-------------------+--------------------------------+--------------------------------+------------------------------------+
| Execution Style   | Iterative and adaptive         | Modular and compartmentalized  | Linear and sequential              |
+-------------------+--------------------------------+--------------------------------+------------------------------------+
```

The comparison analysis automatically generates:
- Performance metrics tables comparing execution time and success rates
- Framework comparison tables showing approach differences
- Visualizations of metrics for easy comparison
- Side-by-side comparison of output maps

[Metrics Comparison](https://github.com/yourusername/llm-agent-frameworks/raw/main/agent_results/metrics_visualization_example.png)

## Future Work

- Implement hybrid approaches combining the strengths of multiple frameworks
- Enhance error recovery in the ReAct framework
- Improve explainability in all frameworks
- Expand the comparison module to evaluate additional metrics

## Conclusion

When choosing an LLM agent framework, consider:

- **Task complexity and structure**
- **Explainability expectations**
- **Error handling needs**

No framework is universally superior, each has strengths for different situations. Understanding these differences helps in selecting the most appropriate approach for specific use cases.
