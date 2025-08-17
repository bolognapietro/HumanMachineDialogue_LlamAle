import json
import yaml
from copy import deepcopy
from typing import List, Dict

import ollama

from utils.history import History


class NLU:
    def __init__(self, model: str, prompt_path: str, eval_mode: bool = False):
        """
        Extracts slots from user intents using prompt-based LLM guidance.

        Args:
            model (str): Model identifier to query.
            prompt_path (str): YAML path containing NLU prompts.
            eval_mode (bool): Toggles inclusion of system-user history.
        """
        self.model = model
        self.prompt_path = prompt_path
        self.eval_mode = eval_mode
        self.history = History()
        self.history.history_limit = 6
        self.prompts = self._load_prompts()

        self.prompt_map = {
            "get_beer_recommendation": "prompt_get_beer_recommendation",
            "get_beer_info": "prompt_get_beer_info",
            "list_beers_by_brewery": "prompt_list_beers_by_brewery",
            "get_top_rated": "prompt_get_top_rated",
            "rate_beer": "prompt_rate_beer",
            "give_evaluation": "prompt_out_of_context",
            "out_of_context": "prompt_out_of_context"
        }

    def _load_prompts(self) -> Dict:
        with open(self.prompt_path, "r") as f:
            return yaml.safe_load(f)

    def __call__(self, pre_nlu_result: List[Dict], user_utterance: str, system_utterance: str) -> List[Dict]:
        """
        Performs slot filling given intents and dialogue history.

        Args:
            pre_nlu_result (List[Dict]): Intent list from pre-NLU.
            user_utterance (str): Last user message.
            system_utterance (str): Last system message.

        Returns:
            List[Dict]: Extracted slot structures.
        """
        cleaned_outputs = []

        self.history.push("system", system_utterance)

        for intent_chunk in pre_nlu_result:
            intent_name = intent_chunk.get("intent")
            prompt_key = self.prompt_map.get(intent_name, "prompt_out_of_context")
            system_prompt = self.prompts["nlu"][prompt_key]

            # Skip give_evaluation if multi-intent detected
            if intent_name == "give_evaluation" and len(pre_nlu_result) > 1:
                continue

            while True:
                raw_response = self._query_model(intent_chunk, system_prompt)

                try:
                    parsed = json.loads(raw_response)
                    result = self._clean_slots(parsed)
                    cleaned_outputs.append(result)
                    break
                except Exception as e:
                    print(f"[NLU] JSON error. Retrying. Response was: {raw_response}")

        self.history.push("user", user_utterance)
        return cleaned_outputs

    def _query_model(self, intent_block: Dict, system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        if self.eval_mode:
            for msg in self.history.get_history():
                messages.append({
                    "role": msg["role"],
                    "content": f"History {msg['role']}: {msg['content']}"
                })
        else:
            messages.append({
                "role": "system",
                "content": "Hi, what can I help you with today?"
            })

        messages.append({
            "role": "user",
            "content": str(intent_block)
        })

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": 200}
        )

        return response["message"]["content"]

    def _clean_slots(self, response: Dict) -> Dict:
        cleaned = deepcopy(response)

        for key, value in response.items():
            if value is None:
                cleaned[key] = "null"
            elif isinstance(value, dict):
                nested = self._clean_slots(deepcopy(value))
                if len(nested) == 0:
                    del cleaned[key]
                else:
                    cleaned[key] = nested

        return cleaned
