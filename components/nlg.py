import yaml
from typing import List, Dict

import ollama


class NLG:
    def __init__(self, model: str, prompt_path: str):
        """
        Generates system responses from structured DM outputs.

        Args:
            model (str): LLM model name to query.
            prompt_path (str): YAML prompt definitions.
        """
        self.model = model
        self.prompts = self._load_prompts(prompt_path)

        # Maps (action, intent) or just action to a prompt key
        self.prompt_map = {
            "request_info": "prompt_request_info",
            "check_info": "prompt_check_info",
            "merge": "prompt_merge_responses",
            ("confirmation", "get_beer_recommendation"): "prompt_confirmation_get_beer_recommendation",
            ("confirmation", "get_beer_info"): "prompt_confirmation_get_beer_info",
            ("confirmation", "list_beers_by_brewery"): "prompt_confirmation_list_beers_by_brewery",
            ("confirmation", "get_top_rated"): "prompt_confirmation_get_top_rated",
            ("confirmation", "rate_beer"): "prompt_confirmation_rate_beer",
            ("confirmation", "out_of_context"): "prompt_confirmation_out_of_context"
        }

    def __call__(self, decisions: List[Dict]) -> str:
        """
        Converts DM actions into a final system utterance.

        Args:
            decisions (List[Dict]): DM outputs.

        Returns:
            str: Verbalized system response.
        """
        responses = []

        for action_obj in decisions:
            if not isinstance(action_obj, dict):
                return "error"

            action = action_obj["action"]
            param = action_obj["parameter"]
            payload = action_obj["data"]

            key = (action, param) if (action, param) in self.prompt_map else action
            prompt_key = self.prompt_map.get(key)

            if not prompt_key:
                return "error"

            system_prompt = self.prompts["nlg"][prompt_key]
            response = self._query_model(action, param, payload, system_prompt)
            responses.append(response)

        return responses[0] if len(responses) == 1 else self._merge_responses(responses)

    def _merge_responses(self, responses: List[str]) -> str:
        prompt = self.prompts["nlg"][self.prompt_map["merge"]]
        combined = "\n\n".join([f"response {i+1}: {text}" for i, text in enumerate(responses)])
        return self._query_model("merge", "responses", combined, prompt)

    def _query_model(self, action: str, param: str, data: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"action: {action}, parameter: {param}\ndata = {data}"}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": 200}
        )

        return response["message"]["content"]

    def _load_prompts(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
