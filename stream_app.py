import streamlit as st
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline
import torchaudio
import torch

import whisper
import stable_whisper

import time
import copy
import os
from io import BytesIO

from langchain_func import summary,word_extraction

import gc
gc.collect()
torch.cuda.empty_cache()
from st_audiorec import st_audiorec
import numpy as np

import pandas as pd

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Speaker = [':zero:', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5',
#            'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9', 'speaker 10',]

speaker_imo = [':zero:',':one:',':two:',':three:',':four:',
               ':five:',':six:',':seven:',':eight:',':nine:']

share = [0] * 10

df = False

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
class Meeting_Recorder():
    def __init__(self,PATH):
        self.path = PATH
        self.audio, self.sample_rate = self.load_audio(self.path)

        self.options = whisper.DecodingOptions(language='ko')
        self.conversation = []

        model_size = "large-v2"
        model_path = "models"

        CHUNK_LENGTH = 30
        self.N_SAMPLES = CHUNK_LENGTH * self.sample_rate

        self.pipeline = Pipeline.from_pretrained("config_dialog.yaml").to(device)
        self.whisper_model = stable_whisper.load_model(model_size, device, model_path)
        self.checker = False
        self.output = []
        self.speakers = 0

    def load_audio(self,audio):
        try:
            waveform, sample_rate = torchaudio.load(audio)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                sample_rate = 16000
            if waveform.shape[0] != 1:
                waveform = waveform[0]
            return waveform, sample_rate
        except:
            # raise Exception(f"audio's type is not string")
            raise Exception(f"audio's type is {type(audio)}")

    def transcribe(self):
        result = self.whisper_model.transcribe(self.audio.to(device))
        for i in result.segments:
            self.output.append((i.start,i.end,i.text))
        gc.collect()
        torch.cuda.empty_cache()

        return self.output
    def dialog_process(self):
        ans = []
        out = copy.deepcopy(self.output)
        speakers = [0] * (self.speakers + 1)

        for i in self.conversation:
            s,e,speaker = i
            sentence = ''
            speakers[int(speaker)] += (e - s)
            for j in out[:]:
                s1,e1,dialog = j
                if e < e1:
                    break
                if sentence:
                    sentence = sentence + ' ' + dialog
                else:
                    sentence = dialog
                out.remove(j)
            if sentence:
                ans.append((s,e,sentence,speaker))
        return ans, speakers
    def speaker_diarization(self):
        audio_in_memory = {"waveform": self.audio[None].to(device), "sample_rate": self.sample_rate}
        result = self.pipeline(audio_in_memory)

        tmp = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            tmp.append((turn.start,turn.end,speaker.split('_')[-1]))
        if tmp:
            tmp.append((-1,-1,-1))
            speaker = ''
            for i in range(len(tmp)-1):
                if not speaker:
                    s = tmp[i][0]
                    e = tmp[i][1]
                    speaker = tmp[i][2]
                    self.speakers = max(self.speakers, int(speaker))
                    if tmp[i][2] != tmp[i + 1][2] or tmp[i + 1][1] - tmp[i][0] > 3:
                        self.conversation.append((s, e, speaker))
                        speaker = ''

                    continue
                if tmp[i][2] != tmp[i+1][2] or tmp[i + 1][1] - tmp[i][0] > 3: # parameter 가능 해 보임 delay time
                    e = tmp[i][1]
                    self.conversation.append((s, e, speaker))
                    speaker = ''
        gc.collect()
        torch.cuda.empty_cache()

st.title(" Meeting Recorder Feat. KH :small_airplane: ")
st.subheader('STT :black_circle_for_record:')

empty1,load_audio,empty2 = st.columns([0.1,1.0,0.1])
empty1,recorde_audio,empty2 = st.columns([0.1,1.0,0.1])
empty1,stt_,select,empty2 = st.columns([0.1,0.5,0.5,0.1])
empty1,share_print,summary_print, word_extraction_print,empty2 = st.columns([0.1,0.3,0.3,0.3,0.1])
empty1,text_box,empty2 = st.columns([0.1,1.0,0.1])

with load_audio:
    file = st.file_uploader("select file(wav or mp3)", type = ['wav','mp3'])
    time.sleep(3)
    if file is not None:
        ext = file.name.split('.')[-1]
        if ext in ['wav', 'mp3']:
            st.write('successfully load')
        else:
            st.write('please upload again')

with recorde_audio:
    wav_audio_data = st_audiorec()
    time.sleep(3)
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')

with stt_:
    genre = st.radio(
        "select audio file",
        ["upload File", "Record File"])

with select:
    st.text(genre)
    STT = st.button("STT", type="primary")

if STT:
    if genre == "upload File":
        audio_file = file.getvalue()
    else:
        audio_file = wav_audio_data
    try:
        audio_bytesIO = BytesIO(audio_file)
        Recorder = Meeting_Recorder(audio_bytesIO)
        Recorder.speaker_diarization()
        Recorder.transcribe()
        dialog, share = Recorder.dialog_process()
        df = pd.DataFrame(dialog)
        df.columns = ['start_t', 'end_t', 'sentence', 'speaker']
        df.to_csv('ans.csv')
        st.write('complete')
    except:
        st.write('please select file')


if os.path.exists('ans.csv'):
    with share_print:
        df = pd.read_csv('ans.csv')
        labels = list(df['speaker'].unique())
        labels.sort()
        options = st.multiselect(
            'How many talk this meet',
            labels,
            labels)

        if options:
            ratios = []
            label = []
            if sum(share) == 0:
                for i in range(len(df)):
                    dt = df.loc[i]
                    s = float(dt['start_t'])
                    e = float(dt['end_t'])
                    sentence = dt['sentence']
                    speaker = dt['speaker']
                    share[int(speaker)] += (e-s)
            for li in options:
                ratios += [share[labels.index(li)]]
                label += [li]
            plt.pie(ratios, labels=label, autopct='%.1f%%')
            st.pyplot(plt)

    with text_box:
        for i in range(len(df)):
            dt = df.loc[i]
            if dt['speaker'] in options:
                s = float(dt['start_t'])
                e = float(dt['end_t'])
                sentence = dt['sentence']
                speaker = dt['speaker']
                st.text(f'{speaker} : {s:.2f} ~ {e:.2f} : {sentence}')
        q = ''
        for i in range(len(df)):
            dt = df.loc[i]
            if dt['speaker'] in options:
                sentence = dt['sentence']
                q += ' ' + sentence
        st.download_button(label = 'Download data as csv',
                           data = convert_df(df),
                           file_name='MeetingRecord.csv',
                           mime='text/csv')
    with summary_print:
        st.text(summary(q))
        gc.collect()
        torch.cuda.empty_cache()
    with word_extraction_print:
        st.text(word_extraction(q))
        gc.collect()
        torch.cuda.empty_cache()
