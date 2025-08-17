from typing import List, Dict


class History:
    def __init__(self, limit: int = 5):
        """
        Maintains a rolling memory of recent dialogue turns.

        Args:
            limit (int): Maximum number of exchanges to store.
        """
        self._memory: List[Dict[str, str]] = []
        self.max_turns = limit

    def reset(self) -> None:
        """Clears the entire memory buffer."""
        self._memory.clear()

    def push(self, role: str, content: str) -> None:
        """
        Adds a new turn to the memory.

        Args:
            role (str): Either 'user' or 'system'.
            content (str): Text content of the message.
        """
        if len(self._memory) >= self.max_turns:
            self._memory.pop(0)

        self._memory.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the full dialogue history."""
        return self._memory

    def get_history_str(self) -> str:
        """
        Returns the memory as a readable dialogue string.

        Returns:
            str: Formatted conversation.
        """
        return "\n".join(f"{entry['role']}: {entry['content']}" for entry in self._memory)

    def clean(self, user_text: str) -> None:
        """
        Clears all turns and keeps only one from the user.

        Args:
            user_text (str): Message to preserve.
        """
        self.reset()
        self.push("user", user_text)
