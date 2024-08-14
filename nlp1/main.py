import config
from chatbot import Chatbot
if __name__ == "__main__":
    
    audio_file = config.audio_file
    chatbot = Chatbot(audio_file)
    query = chatbot.speech_to_text(audio_file)
    response = chatbot.chatbot(query)
    print(response)