import os

def create_output_directory(task_name: str = "Task-1") -> str:
    output_dir = f"results/{task_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir
