from transformers import pipeline
class InputHandler:
    def __init__(self):
        pass

    def receive_task(self, task: str):
        # Receives the task and sends it to the Task Planner
        return task
    
class TaskPlannerAgent:
    def __init__(self):
        pass

    def plan_task(self, task: str):
        # Break down the task into subtasks based on keywords or task analysis
        subtasks = []
        if "summarize" in task.lower():
            subtasks.append("summarize")
        if "translate" in task.lower():
            subtasks.append("translate")
        return subtasks

class SummarizationAgent:
    def __init__(self):
        self.summarizer = pipeline("summarization")

    def summarize(self, text: str):
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']


class TranslationAgent:
    def __init__(self):
        self.translator = pipeline("translation_en_to_fr")

    def translate(self, text: str):
        translation = self.translator(text)
        return translation[0]['translation_text']

class CoordinatorAgent:
    def __init__(self):
        self.summarization_agent = SummarizationAgent()
        self.translation_agent = TranslationAgent()

    def execute(self, task: str, subtasks: list):
        results = {}

        if "summarize" in subtasks:
            # Assuming we have some text to summarize
            text_to_summarize = "This is a sample text to summarize."  # Replace with actual input
            summary = self.summarization_agent.summarize(text_to_summarize)
            results['summary'] = summary

        if "translate" in subtasks and 'summary' in results:
            translation = self.translation_agent.translate(results['summary'])
            results['translation'] = translation

        return results

class OutputHandler:
    def __init__(self):
        pass

    def return_output(self, results: dict):
        # Compile results into a final output format
        output = "\n".join([f"{key.capitalize()}: {value}" for key, value in results.items()])
        return output
