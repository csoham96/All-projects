from task import InputHandler,TaskPlannerAgent,CoordinatorAgent,OutputHandler
if __name__ == "__main__":
    # Initialize components
    input_handler = InputHandler()
    task_planner = TaskPlannerAgent()
    coordinator = CoordinatorAgent()
    output_handler = OutputHandler()

    # Simulate receiving a task
    task = "Summarize and translate this article into French."
    received_task = input_handler.receive_task(task)

    # Plan the task
    subtasks = task_planner.plan_task(received_task)

    # Execute the task
    results = coordinator.execute(received_task, subtasks)

    # Return the output
    final_output = output_handler.return_output(results)
    print(final_output)