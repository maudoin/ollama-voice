import pyttsx3
import numpy as np
import whisper
import pyaudio
import sys
import torch
import requests
import json
import yaml
from yaml import Loader

if sys.version_info[0:3] != (3, 9, 13):
    print('Warning, it was only tested with python 3.9.13, it may fail')

INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json',}
INPUT_CONFIG_PATH ="assistant.yaml"
    

class Assistant:


    def __init__(self):
        self.config = self.initConfig()    
        print("Loading Whisper model...")
        self.model = whisper.load_model(self.config.whisperRecognition.modelPath)
        self.tts = pyttsx3.init()    
        self.audio = pyaudio.PyAudio()
        self.conversation_history = [self.config.conversation.context+self.config.conversation.greeting+"\n"]


    def initConfig(self):
        class Inst:
            pass
        config=Inst();
        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = "whisper/large-v3.pt"
        config.whisperRecognition.lang = "fr"
        config.ollama = Inst()
        config.ollama.url = "http://localhost:11434/api/generate"
        config.ollama.model = 'mistral'
        config.conversation = Inst()
        config.conversation.context = "This is a discussion in french.\n"
        config.conversation.greeting = "Je vous écoute."
        config.conversation.recognitionWaitMsg = "J'interprète votre demande."
        config.conversation.llmWaitMsg = "Laissez moi réfléchir."
        
        stream = open(INPUT_CONFIG_PATH, 'r', encoding="utf-8")
        dic = yaml.load(stream, Loader=Loader)
        #dic depth 2: map values to attributes
        def dic2Object(dic, object):
            for key in dic: 
                setattr(object, key, dic[key]) 
        #dic depth 1: fill depth 2 attributes
        for key in dic: 
             dic2Object(dic[key], getattr(config, key))

        return config

    def waveform_from_mic(self, duration=INPUT_DEFAULT_DURATION_SECONDS) -> np.ndarray:

        stream = self.audio.open(format=INPUT_FORMAT, channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE, input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        for _ in range(0, int(INPUT_RATE / INPUT_CHUNK * duration)):
            data = stream.read(INPUT_CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        print("Finished recording, converting to text...")
        self.text_to_speech(self.config.conversation.recognitionWaitMsg)

        transcript = self.model.transcribe(waveform, language = self.config.whisperRecognition.lang, fp16=torch.cuda.is_available())
        return transcript["text"]
    
    
    def ask_ollama(self, prompt):
        print("Sending: ", prompt)
        self.text_to_speech(prompt+self.config.conversation.llmWaitMsg)    

        self.conversation_history.append(prompt)
        full_prompt = "\n".join(self.conversation_history)
        response = requests.post(self.config.ollama.url, json= {"model": self.config.ollama.model,"stream":False,"prompt":full_prompt}, headers=OLLAMA_REST_HEADERS)
        if response.status_code == 200:
            data = json.loads(response.text)
            response_text = data["response"]
            self.conversation_history.append(response_text)
            print("Received: ", response_text)
            return response_text
        else:
            return "Erreur: " + response.text

    def text_to_speech(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

def main():

    ass = Assistant()

    ass.text_to_speech(ass.config.conversation.greeting)
    print("Recording...")

    speech = ass.waveform_from_mic()

    transcription = ass.speech_to_text(waveform=speech)
    
    response = ass.ask_ollama(transcription)

    ass.text_to_speech(text=response)

if __name__ == "__main__":
    main()

