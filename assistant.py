import pyttsx3
import numpy as np
import whisper
import pyaudio
import sys
import torch
import requests
import json
import yaml
import wave
from yaml import Loader
import pygame, sys
import pygame.locals

BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
REC_SIZE = 80
FONT_SIZE = 24
WIDTH = 320
HEIGHT = 240
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 32



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

        programIcon = pygame.image.load('assistant.png')

        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Assistant")

        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.audio = pyaudio.PyAudio()
        
        self.tts = pyttsx3.init()    

        try:
            self.audio.open(format=INPUT_FORMAT, 
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE, 
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except :        
            self.wait_exit()

        self.text_to_speech(self.config.messages.loadingModel)
        self.display_message(self.config.messages.loadingModel)
        self.model = whisper.load_model(self.config.whisperRecognition.modelPath)
        #self.conversation_history = [self.config.conversation.context,
        #                             self.config.conversation.greeting]
        self.context = []

        self.text_to_speech(self.config.conversation.greeting)
        self.display_message(self.config.messages.pressSpace)

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        self.audio.terminate()
        pygame.quit()
        sys.exit()

    def initConfig(self):
        class Inst:
            pass
        config=Inst();
        config.messages = Inst()
        config.messages.pressSpace = "Pressez sur espace pour parler puis relachez."
        config.messages.loadingModel = "Loading model..."
        config.messages.noAudioInput = "Erreur: Pas d'entrée son"
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
                if hasattr(object, key):
                    setattr(object, key, dic[key]) 
                else:
                    print("Ignoring unknow setting ", key)
        #dic depth 1: fill depth 2 attributes
        for key in dic: 
            if hasattr(config, key):
                dic2Object(dic[key], getattr(config, key))
            else:
                print("Ignoring unknow setting ", key)
                

        return config

    def display_rec_start(self):
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        COL_COUNT = 5
        RED_CENTER = 150
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE*energy)
        hspace, vspace = 2*KWIDTH, int(KHEIGHT/2)
        def rect_coords(x, y):
            return (int(x-KWIDTH/2), int(y-KHEIGHT/2),
                    KWIDTH, KHEIGHT)
        for i in range(-int(np.floor(COL_COUNT/2)), int(np.ceil(COL_COUNT/2))):
            x, y, count = WIDTH/2+(i*hspace), HEIGHT/2, amplitude-2*abs(i)    
            
            mid = int(np.ceil(count/2))
            for i in range(0, mid):
                color = (RED_CENTER+(FACTOR*(i % mid)), 0, 0)
                offset = i*(KHEIGHT+vspace)
                pygame.draw.rect(self.windowSurface, color, 
                                rect_coords(x, y+offset))
                #mirror:
                pygame.draw.rect(self.windowSurface, color, 
                                rect_coords(x, y-offset))
        pygame.display.flip()

    def display_message(self, text):
        self.windowSurface.fill(BACK_COLOR)
        
        label = self.font.render(text 
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY) 
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."), 
                                 1, 
                                 TEXT_COLOR)
        
        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT/2 - size[1]/2))

        pygame.display.flip()

    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:

        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT, 
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE, 
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:            
            pygame.event.pump() # process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        self.text_to_speech(self.config.conversation.recognitionWaitMsg)    

        transcript = self.model.transcribe(waveform, 
                                           language = self.config.whisperRecognition.lang, 
                                           fp16=torch.cuda.is_available())
        text = transcript["text"]
        self.text_to_speech(text)
        return text
    
    
    def ask_ollama(self, prompt, responseCallback):
        #self.conversation_history.append(prompt)
        #full_prompt = "\n".join(self.conversation_history)
        full_prompt = prompt if hasattr(self, "contextSent") else (self.config.conversation.context+"\n"+prompt)
        self.contextSent = True
        jsonParam= {"model": self.config.ollama.model,
                                        "stream":True,
                                        "context":self.context,
                                        "prompt":full_prompt}
        response = requests.post(self.config.ollama.url, 
                                 json=jsonParam, 
                                 headers=OLLAMA_REST_HEADERS,
                                 stream=True)
        response.raise_for_status()

        #print(jsonParam)
        self.text_to_speech(self.config.conversation.llmWaitMsg)    

        tokens = []
        for line in response.iter_lines():
            #print(line)
            body = json.loads(line)
            token = body.get('response', '')
            tokens.append(token)
            # the response streams one token at a time, process only at end of sentences
            if token == "." or token == ":" or token == "!" or token == "?":
                current_response = "".join(tokens)
                #self.conversation_history.append(current_response)
                responseCallback(current_response)
                tokens = []

            if 'error' in body:
                responseCallback("Erreur: " + body['error'])

            if body.get('done', False):
                self.context = body['context']

    def text_to_speech(self, text):
        print(text)
        tempPath = 'temp.wav'
        #self.tts.say(text)
        self.tts.save_to_file(text , tempPath)
        self.tts.runAndWait()
        wf = wave.open(tempPath, 'rb')
        # open stream based on the wave object which has been input.
        stream = self.audio.open(format =
                        self.audio.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)
        chunkSize = 1024
        chunk = wf.readframes(chunkSize)
        while chunk:
            stream.write(chunk)
            tmp = np.array(np.frombuffer(chunk, np.int16), np.float32) * (1 / 32768.0)
            energy_of_chunk = np.sqrt(np.mean(tmp**2))
            self.display_sound_energy(energy_of_chunk)
            chunk = wf.readframes(chunkSize)

        wf.close()    
        self.display_message(text)

def main():

    if sys.version_info[0:3] != (3, 9, 13):
        print('Warning, it was only tested with python 3.9.13, it may fail')
        
    pygame.init()

    ass = Assistant()

    push_to_talk_key = pygame.K_SPACE;
   
    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == push_to_talk_key:
                speech = ass.waveform_from_mic(push_to_talk_key)

                transcription = ass.speech_to_text(waveform=speech)
                
                ass.ask_ollama(transcription, ass.text_to_speech)
                
                ass.display_message(ass.config.messages.pressSpace)

            if event.type == pygame.locals.QUIT:
                ass.shutdown()


if __name__ == "__main__":
    main()

