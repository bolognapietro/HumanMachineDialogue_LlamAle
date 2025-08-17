from termcolor import colored

from components import pre_nlu, nlu, dm, nlg


class DialogueOrchestrator:
    def __init__(self, config: dict):
        """
        Initializes the dialogue system components.
        """
        self.model = config["model"]
        self.prompts_path = config["prompts_path"]
        self.eval_mode = True
        self.chat_active = True

        self.pre_nlu = pre_nlu.PRE_NLU(self.model, self.prompts_path, self.eval_mode)
        self.nlu = nlu.NLU(self.model, self.prompts_path, self.eval_mode)
        self.dm = dm.DM(self.model, self.prompts_path, self.eval_mode)
        self.nlg = nlg.NLG(self.model, self.prompts_path)

        self.bot_name = colored("BeerBot", "red")
        self.user_name = colored("User", "green")

    def run(self):
        """
        Starts the interactive chat loop.
        """
        self.display_intro()

        system_message = "What can I do for you?"
        print(f"{self.bot_name}: {system_message}")

        while self.chat_active:
            user_input = input(f"{self.user_name}: ").strip()

            if not user_input:
                continue

            pre_nlu_output = self.pre_nlu(user_input, system_message)

            if any(intent.get("intent") == "terminate_system" for intent in pre_nlu_output):
                print(f"{self.bot_name}: Goodbye! 🍻")
                self.chat_active = False
                return

            nlu_output = self.nlu(pre_nlu_output, user_input, system_message)
            dm_output = self.dm(nlu_output)
            nlg_output = self.nlg(dm_output)

            system_message = nlg_output
            print(f"\n{self.bot_name}: {system_message}")

    def display_intro(self):
        """
        Displays the welcome message and ASCII banner.
        """
        print("""
    ⠀⣾⣿⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠹⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠹⣏⢿⣿⣿⣿⣿⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠈⢦⠹⣿⣿⣿⠀⠹⡄⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⣀⣀⠀⠀⠀⠀⠀
    ⠀⠀⠀⢠⣶⣷⣌⠻⣇⠀⢀⡇⣀⠴⠚⠉⠀⠀⠀⠀⠈⠑⠋⠀⠈⠂⢀⠀⠀⠀
    ⠀⠀⠀⣨⡿⠙⣿⣷⠦⣽⡿⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠱⠀⠀
    ⠀⣠⠞⣿⣤⣶⣯⣝⣶⡟⠛⠒⠻⠿⠦⠤⠤⠤⠶⠤⠤⠞⠃⡁⠀⠀⠀⠀⡇⠀
    ⢰⠃⣰⡿⠟⠋⠉⠻⣿⠇⣿⡇⢰⣶⣶⠀⣶⣶⣶⠀⣶⣶⡎⠀⠀⠀⠀⠀⠃⠀
    ⡏⢀⣿⠁⠀⠀⠀⠀⢸⢠⣿⡇⢸⣿⣿⠀⣿⣿⣿⠀⣿⣿⣷⠀⠀⠀⠀⠀⠀⢡
    ⡇⢸⣿⠀⠀⠀⠀⠀⣸⢸⣿⡇⢸⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠇⠀⠀⠀⠀⠀⡆
    ⢹⠈⣿⣇⠀⠀⠀⠀⡏⢸⣿⡇⢸⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⢀⣀⠀⠀⠀⠀
    ⠈⣧⠘⣿⣆⠀⠀⠀⡇⣸⣿⠇⣸⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠒⣿⣿⢡⠀⢸⠀
    ⠀⠘⣧⠘⢿⣦⠀⢰⡇⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⡄⡇⠊⠀
    ⠀⠀⠈⢳⡈⢻⣷⣸⠁⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⡇⡇⠀⠀
    ⠀⠀⠀⠀⠙⣄⢻⣿⢰⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⡇⣿⣿⡇⢹⠀⠀
    ⠀⠀⠀⠀⣠⠟⢸⣿⢸⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⡇⢹⣿⣷⢸⠀⠀
    ⠀⠀⠀⢸⣧⣤⡿⡇⢸⣿⡟⠀⣿⣿⣿⠀⣿⣿⣿⠀⣿⣿⣿⡇⢸⣿⣿⢸⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⡇⣾⣿⡇⠀⣿⣿⡏⠀⣿⣿⣿⠀⣿⣿⣿⡇⢸⣿⣿⡈⡇⠀
    ⠀⠀⠀⠀⠀⠀⢸⡃⠙⠻⠇⠸⣿⣿⡇⠀⣿⣿⣿⠀⣿⣿⣿⡇⢸⡿⠿⠃⡇⠀
    ⠀⠀⠀⠀⠀⠀⠀⠉⠛⠒⠶⠤⠤⢤⣤⣤⣤⣄⣀⣠⣤⣤⠤⠤⠴⠖⠒⠛⠁⠀

    🍺 Welcome to LlamAle 🍺

    Hey there, beer enthusiast! 
    I’m BeerBot, your personal AI beer guide. Here to help you discover crisp lagers, rich stouts, hoppy IPAs, and adventurous brews.

    Here’s what I can do for you:
     🍺 Recommend beers based on your preferred style, strength, or bitterness
     🍺 Share details about any beer you’re curious about
     🍺 Show beers from your favorite brewery
     🍺 List the top-rated beers in our collection
     🍺 Record your rating for a beer you’ve tried
        """)