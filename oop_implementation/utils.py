import asyncio
from typing import List, Coroutine

class ProgressTracker:
    """
    Utility class for tracking the progress of asynchronous tasks.
    """
    async def track_progress(self, tasks: List[asyncio.Task]) -> List:
        """
        Tracks progress while tasks are running and gathers results.

        Args:
            tasks (List[asyncio.Task]): A list of asyncio tasks to track.

        Returns:
            List: A list of results from the completed tasks.
        """
        total_tasks = len(tasks)
        while not all(task.done() for task in tasks):
            completed = sum(task.done() for task in tasks)
            print(f"Progress: {completed}/{total_tasks} tasks completed", end="\r")
            await asyncio.sleep(1)

        print("\nAll tasks completed!")
        return await asyncio.gather(*tasks)