import yaml
import json
import re
from typing import List, Dict

import ollama

from utils.history import History
from dataset.dataset import BeerDataset


class StateTracker:
    def __init__(self, intent: str):
        """
        Represents the state of a specific user intent.
        Tracks slot-filling progress.
        """
        self.intent = intent
        self.slots = self._initialize_slots(intent)

    def _initialize_slots(self, intent: str) -> Dict[str, str | None]:
        intent_slots = {
            "get_beer_recommendation": ["style", "abv", "ibu", "rating"],
            "get_beer_info": ["name", "brewery"],
            "list_beers_by_brewery": ["brewery"],
            "get_top_rated": ["style"],
            "rate_beer": ["name", "rating", "comment"],
            "out_of_context": []
        }

        fields = intent_slots.get(intent, [])
        return {field: None for field in fields}

    def update(self, nlu_input: Dict) -> str:
        for _, value in nlu_input.items():
            if isinstance(value, dict):
                for slot, val in value.items():
                    self.slots[slot] = None if val in ["null", None] else val
        return self.serialize()

    def serialize(self) -> str:
        safe = {k: (v if v is not None else "null") for k, v in self.slots.items()}
        return json.dumps({"intent": self.intent, "slots": safe}, indent=2)

    def get_intent(self) -> str:
        return self.intent


class DM:
    def __init__(self, model: str, prompt_path: str, eval_mode: bool = False):
        """
        Decides next actions based on slot states and intent type.
        """
        self.model = model
        self.eval_mode = eval_mode
        self.dataset = BeerDataset()
        self.history = History()
        self.history.history_limit = 6
        self.prompts = self._load_prompts(prompt_path)
        self.state_stack: List[StateTracker] = []

    def _load_prompts(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def __call__(self, nlu_inputs: List[Dict]) -> List[Dict]:
        if len(nlu_inputs) == 1 and nlu_inputs[0].get("intent") == "terminate_system":
            return ["terminate_system"]

        self.state_stack = self._update_states(nlu_inputs)

        final_actions = []
        remove_indices = []

        for idx, state in enumerate(self.state_stack):
            state_str = state.serialize()
            repeat = True

            while repeat:
                model_reply = self._query_model(state_str)
                try:
                    parsed = json.loads(self._extract_json(model_reply))
                except:
                    print(f"[DM] Retry on invalid JSON: {model_reply}")
                    continue

                repeat = not self._check_response_validity(parsed, state) if not self.eval_mode else False

            if parsed["action"] == "confirmation":
                result = self._handle_confirmation(parsed)
                if result:
                    remove_indices.append(idx)
                else:
                    parsed = {"action": "check_info", "parameter": parsed["parameter"]}
                    result = state.serialize()

            elif parsed["action"] == "request_info":
                result = f"intent: {state.get_intent()}"

            final_actions.append({
                "action": parsed["action"],
                "parameter": parsed["parameter"],
                "data": result
            })

        self.state_stack = [s for i, s in enumerate(self.state_stack) if i not in remove_indices]
        return final_actions

    def _update_states(self, nlu_inputs: List[Dict]) -> List[StateTracker]:
        for nlu in nlu_inputs:
            intent = nlu["intent"]
            found = False
            for state in self.state_stack:
                if state.get_intent() == intent:
                    state.update(nlu)
                    found = True
                    break
            if not found:
                new_state = StateTracker(intent)
                new_state.update(nlu)
                self.state_stack.append(new_state)
        return self.state_stack

    def _query_model(self, user_payload: str) -> str:
        system_prompt = self.prompts["dm"]["prompt"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": 200}
        )

        return response["message"]["content"]

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

    def _check_response_validity(self, nba: Dict, state: StateTracker) -> bool:
        action, parameter = nba.get("action"), nba.get("parameter")
        if action not in ["request_info", "confirmation"]:
            return False
        if not parameter or str(parameter).lower() in ["none", "null"]:
            return False

        if action == "request_info" and state.slots.get(parameter) is not None:
            return False

        full_slots = all(v is not None for v in state.slots.values())
        if full_slots and action != "confirmation":
            return False
        if not full_slots and action != "request_info":
            return False

        if state.get_intent() == "get_beer_info":
            return (parameter == "name" and action == "request_info") or \
                   (state.slots.get("name") is not None and action == "confirmation")

        return True

    def _handle_confirmation(self, nba_confirm: Dict) -> str | None:
        target_intent = nba_confirm["parameter"]

        target_state = next((s for s in self.state_stack if s.get_intent() == target_intent), None)
        if not target_state:
            return None

        if target_intent in {
            "get_beer_recommendation",
            "get_beer_info",
            "list_beers_by_brewery",
            "get_top_rated"
        }:
            return self.dataset.filter_by_intent(slots=target_state.slots, intent=target_intent)

        elif target_intent == "rate_beer":
            return self.dataset.record_user_rating(target_state.slots)

        elif target_intent == "out_of_context":
            return "Intent out_of_context"

        return None
