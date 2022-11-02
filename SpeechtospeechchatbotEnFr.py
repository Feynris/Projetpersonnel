# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:19:33 2022

@author: Feyn
"""
import pyaudio
import wave
import whisper
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22500
modelwhisper = whisper.load_model("small")

"""record permet d'enregistrer nos paroles """
def record():

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)

	print("Start recording")

	frames = []

	try:
		while True:
			data = stream.read(CHUNK)
			frames.append(data)
	except KeyboardInterrupt:
		print("Done recording")
	except Exception as e:
		print(str(e))

	sample_width = p.get_sample_size(FORMAT)
	
	stream.stop_stream()
	stream.close()
	p.terminate()
	
	return sample_width, frames	

def record_to_file(file_path):
	wf = wave.open(file_path, 'wb')
	wf.setnchannels(CHANNELS)
	sample_width, frames = record()
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
    


def transcribe(audio):
    record_to_file(audio)
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(modelwhisper.device)

    # detect the spoken language
    _, probs = modelwhisper.detect_language(mel)
    print(f"langage: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(modelwhisper, mel, options)
   
    print(f"Moi: {result.text}")
    
    return result.text

from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer


from gtts import gTTS
from io import BytesIO
import pygame

tokenizeren = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
modelen = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to('cuda')

tokenizerfr=AutoTokenizer.from_pretrained("emil2000/dialogpt-for-french-language")
modelfr = AutoModelForCausalLM.from_pretrained("emil2000/dialogpt-for-french-language").to('cuda')
"""
ne marche pas bien
tokenizerja = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizerja.do_lower_case = True  # due to some bug of tokenizer config loading

modelja = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").to('cuda')"""

def speak(text):
	mp3_fo = BytesIO()
	tts = gTTS(text, lang=result.language,slow=False)
	tts.write_to_fp(mp3_fo)
	return mp3_fo
 

for step in range(10):
    
    record_to_file('output.wav')
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio('output.wav')
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(modelwhisper.device)

    # detect the spoken language
    _, probs = modelwhisper.detect_language(mel)
    print(f"langage: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(modelwhisper, mel, options)
   
    print(f"Moi: {result.text}")
    if result.language=='fr':
        tokenizer=tokenizerfr
        model=modelfr
    else:
        tokenizer=tokenizeren
        model=modelen
    """elif result.language=='ja':
        tokenizer=T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium") 
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        model=AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").to('cuda')"""
   

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode( result.text + tokenizer.eos_token, return_tensors='pt').to('cuda')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generation d'une reponse, pour eviter des repetitions ajouts d'aleatoire, 
    chat_history_ids = model.generate(bot_input_ids, do_sample=True, max_length=1000, top_k=50, 
        top_p=0.9, pad_token_id=tokenizer.eos_token_id, temperature =0.5 ).to('cuda')

    textefeyn=format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    # pretty print last ouput tokens from bot
    pygame.init()
    pygame.mixer.init()
    # sound.seek(0)
    sound = speak(textefeyn)
    pygame.mixer.music.load(sound, 'mp3')
    pygame.mixer.music.play()
    print("Feynbot: {}".format(textefeyn))


	
	
    
