AGENT_LIST= """
database_agent: Expert in fetching images from a database!
map_agent: Expert in performing all kinds of operations on a map!
detector_agent: Expert in processing images fetched from a database, such as object detection!
data_agent: Expert in all kinds of image analyzing tasks!
"""


INIT_WORKFLOW_TEMPLATE = """"
For query: "Fetch BigEarthNet in Switzerland for and run the ResNet-32 classifier. Please plot on the map the 'Vineyards' and 'Fruit trees and berry plantations' LCC classes!"
The result should be in the following format:
{
  tasks": {
    "task0": {
      "id": "task0",
      "objective": "Fetch images from the BigEarthNet dataset and filter the images specifically from Switzerland.",
      "agent_id": 0,
      "next": [
        "task1"
      ],
      "prev": [],
      "status": "pending",
      "history": "",
      "remaining_dependencies": 0,
      "agent": "database_agent"
    },
    "task1": {
      "id": "task1",
      "objective": "Run the ResNet-32 classifier on BigEarthNet images to detect 'Vineyards' and 'Fruit trees and berry plantations'.",
      "agent_id": 2,
      "next": [
        "task2"
      ],
      "prev": [
        "task0"
      ],
      "status": "pending",
      "history": "",
      "remaining_dependencies": 1,
      "agent": "detector_agent"
    },
    "task2": {
      "id": "task2",
      "objective": "Plot ResNet-32 classification results highlighting the 'Vineyards' and 'Fruit trees and berry plantations' classes on the map.",
      "agent_id": 3,
      "next": [],
      "prev": [
        "task1"
      ],
      "status": "pending",
      "history": "",
      "remaining_dependencies": 1,
      "agent": "map_agent"
    }
  }
}

"""

INIT_WORKFLOW_PROMPT = f'''
You are a workflow planner. Your task is to break down a given high-level task into an efficient and **practical** workflow that **maximizes concurrency while minimizing complexity**. 

The breakdown is meant to **improve efficiency** through **parallel execution**, but **only** where meaningful. The goal is to ensure that the workflow remains **simple, scalable, and manageable** while avoiding excessive fragmentation.

---

# **Guidelines for Workflow Design**
## **1. Subtask Clarity and Completeness**
- **Each subtask must be well-defined, self-contained, and easy to execute by a single agent.**
- **Ensure that the workflow meets all requirements of the task.**
- **Keep descriptions concise but informative.** Clearly specify the subtask's purpose, the operation it performs, and its role in the overall workflow.
- **Avoid unnecessary subtasks.** If a task can be handled efficiently in one step without blocking others, do not split it further.


## **2. Dependency Optimization and Parallelization**
- **Identify only necessary dependencies.** Do not introduce dependencies unless a subtask *genuinely* requires the output of another.
- **Encourage parallel execution, but do not force it.** If tasks can run independently without affecting quality, prioritize concurrency. However, avoid excessive parallelization that may lead to synchronization issues.
- **Keep the dependency graph simple.** Avoid deep dependency chains that increase complexity.

## **3. Efficient Agent Assignment**
- **Assign exactly one agent per subtask.** Every subtask must have a responsible agent.
- **Use sequential agent IDs starting from "Agent 0".** Assign agents in a clear, structured way.
- **Ensure logical role assignments.** Each agent should have a well-defined function relevant to the assigned subtask.

## **4. Workflow Simplicity and Maintainability**
- **Do not overcomplicate the workflow.** A well-balanced workflow has an optimal number of subtasks that enhance efficiency without adding unnecessary coordination overhead.
- **Maintain clarity and logical flow.** The breakdown should be intuitive, avoiding redundant or trivial steps.
- **Prioritize quality over extreme concurrency.** Do not split tasks into too many small fragments if it negatively impacts output quality.

## Below is an Output Format Template:
```json
{INIT_WORKFLOW_TEMPLATE}
```
'''
