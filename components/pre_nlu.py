import yaml
import json
import re
from typing import List, Dict, Optional

import ollama

from utils import params
from utils.history import History


class PRE_NLU:
    def __init__(self, model: str, prompt_path: str, eval_mode: bool = False):
        """
        Initializes the intent classifier (Pre-NLU step).

        Args:
            model (str): Name of the LLM to query.
            prompt_path (str): Path to the YAML prompt file.
            eval_mode (bool): Whether evaluation mode is enabled.
        """
        self.model = model
        self.prompt_path = prompt_path
        self.eval_mode = eval_mode
        self.history = History()
        self.history.history_limit = 6

        self.valid_intents = params.VALID_INTENTS
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        with open(self.prompt_path, "r") as f:
            return yaml.safe_load(f)

    def __call__(self, user_input: str = " ", system_response: str = " ") -> List[Dict]:
        """
        Runs the intent classification step.

        Args:
            user_input (str): User message.
            system_response (str): Last system message.

        Returns:
            List[Dict]: Detected intents with cleaned structure.
        """
        self.history.push("system", system_response)
        system_prompt = self.prompts["pre_nlu"]["prompt"]

        while True:
            raw_output = self._query_model(user_input, system_prompt)

            try:
                cleaned_text = self._extract_json_array(raw_output)
                intents = json.loads(cleaned_text)

                if isinstance(intents, dict):
                    intents = [intents]

                parsed = self._sanitize_intents(intents)

                if all(intent.get("intent") in self.valid_intents for intent in parsed):
                    break

            except Exception as e:
                print(f"[PRE-NLU] Invalid output: {raw_output}. Retrying...")

        self.history.push("user", user_input)
        return parsed

    def _query_model(self, user_input: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if self.eval_mode:
            messages.append({
                "role": "system",
                "content": f"History: {self.history.get_history()}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "What can I help you with?"
            })

        messages.append({
            "role": "user",
            "content": user_input
        })

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": 200}
        )

        return response["message"]["content"]

    def _extract_json_array(self, raw_text: str) -> str:
        """
        Extracts a JSON array or object string from model output.
        """
        pattern = r"\[\s*{.*?}\s*\]"
        match = re.search(pattern, raw_text, re.DOTALL)

        if match:
            return match.group(0)
        return raw_text

    def _sanitize_intents(self, intents: List[Dict]) -> List[Dict]:
        """
        Ensures intent values are not None. Converts them to 'null' if needed.
        """
        sanitized = []

        for intent in intents:
            entry = {}
            for key, value in intent.items():
                entry[key] = value if value is not None else "null"
            sanitized.append(entry)

        return sanitized
