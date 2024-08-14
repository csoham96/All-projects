# Agent-Based Language Model System

## Overview

The Agent-Based Language Model System is designed to process and complete tasks using specialized worker agents. These agents interact to handle complex tasks and provide the final output in text format. The system can be extended to support various types of tasks such as summarization, translation, and more.

## Features

- Modular architecture for easy extension
- Task decomposition into manageable subtasks
- Coordination between worker agents
- Evaluation using ROUGE and BLEU scores

## Installation and Setup

### Prerequisites

- Python 3.7+
- Conda (for environment management)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/agent-based-lm-system.git
   cd agent-based-lm-system
    ```
2. Create a virtual environment:
    ```bash
    conda create -n agent_lm python=3.8
    conda activate agent_lm
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install additional libraries for evaluation:
    ```bash
    pip install nltk rouge-score
    ```

### Usage Instructions

Basic Usage

To use the system, provide a task prompt through a script or command line:
```bash
python
Copy code
from agent_lm_system import TaskPlanner

# Example task prompt
task_prompt = "Summarize the document and translate it to French."

# Initialize the Task Planner
planner = TaskPlanner(task_prompt)

# Execute the task
output = planner.execute_task()
print(output)
```
Running in Batch Mode
You can run multiple tasks in batch mode:


```bash
task_prompts = [
    "Summarize the first document.",
    "Translate the second document to Spanish.",
    "Summarize and translate the third document to French."
]

outputs = [TaskPlanner(prompt).execute_task() for prompt in task_prompts]
for output in outputs:
    print(output)
```
## Components Description
Task Planner

The Task Planner analyzes the input prompt and breaks it down into subtasks. It assigns these tasks to the Worker Agents.

### Key Methods

- analyze_task()
- create_subtasks()
- assign_tasks()

### Worker Agents
Worker Agents handle specific tasks such as summarization or translation.

Key Methods

- perform_task()
- get_result()

### Coordinator Agent

The Coordinator Agent manages the workflow and compiles outputs from Worker Agents.

Key Methods

- coordinate_tasks()
- compile_results()

### Task Examples

Example 1: Simple Summarization
Input Task: "Summarize the following text: ..."

Output: "This text provides an overview of ..."

Example 2: Summarization and Translation
Input Task: "Summarize the document and translate it to French."

Output: "Résumé: ... Traduction: ..."

## Evaluation
## Accuracy Assessment

ROUGE Score Calculation for Summarization:

run 
```bash
python eval_summary.py
```

BLEU Score Calculation for Translation:

run
```bash
python eval_translation.py
```

### Efficiency Metrics

Latency: Time taken to complete a single task.

Throughput: Number of tasks completed per unit of time.

Resource Usage: Monitor CPU, memory, and other resources during task execution.

#### Reliability Testing

Error Rate: Track frequency of task failures.

Consistency: Verify consistency of outputs for repeated tasks.

Recovery: Test the system’s ability to recover from errors.

#### Testing and Validation

Unit Tests
Unit tests are implemented to test individual components.

#### Integration Tests
Integration tests ensure that all components work together.

