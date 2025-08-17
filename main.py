import argparse
from utils.chat_controller import DialogueOrchestrator


def parse_arguments() -> dict:
    """
    Parse CLI arguments for configuring the dialogue system.

    Returns:
        dict: A dictionary with model name and prompt file path.
    """
    parser = argparse.ArgumentParser(
        description="Run the AleAgent conversational system."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="LLM model identifier (default: 'llama3')"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts/prompts.yaml",
        help="Path to the prompt configuration file (default: 'prompts/prompts.yaml')"
    )

    args = parser.parse_args()

    return {
        "model": args.model,
        "prompts_path": args.prompts
    }


def launch() -> None:
    """
    Entry point to start the dialogue agent.
    """
    config = parse_arguments()
    assistant = DialogueOrchestrator(config)
    assistant.run()


if __name__ == "__main__":
    launch()
