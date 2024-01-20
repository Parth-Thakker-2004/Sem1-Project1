from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()
# importing libraries 
import speech_recognition as sr 
import os
import cv2
import fitz
import tabula
import pytesseract
from textblob import TextBlob
from pydub import AudioSegment
from pydub.silence import split_on_silence
from textblob.classifiers import NaiveBayesClassifier

train=[
        ('Sustainable energy','pos'),
        ('green energy','pos'),
        ('fossil fuels','neg'),
        ('green','pos'),
        ('grass','pos'),
        ('elms','pos'),
        ('treasure','pos'),
        ('blossoms','pos'),
        ('air','pos'),
        ('bright','pos'),
        ('himalay','pos'),
        ('communion','pos'),
        ('path','pos'),
        ('pollution','neg'),
        ('mountains','pos'),
        ('rivers','pos'),
        ('birds','pos'),
        ('deer','pos'),
        ('elephant','pos'),
        ('swan','pos'),
        ('harmony','pos'),
        ('natural food','pos'),
        ('gurukula','pos'),
        ('poor resource management','neg'),
        ('serials and entertainment','neg'),
        ('poor understanding','neg'),
        ('lack of knowledge','neg'),
        ('harmful side effects','neg'),
        ('artificial products','neg')
]
cl=NaiveBayesClassifier(train)
    
# create a speech recognition object
r = sr.Recognizer()

# a function to recognize speech in the audio file
# so that we don't repeat ourselves in in other functions
def transcribe_audio(path):
    # use the audio file as the audio source
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        # try converting it to text
        text = r.recognize_google(audio_listened)
    return text

# a function that splits the audio file into chunks on silence
# and applies speech recognition
def get_large_audio_transcription_on_silence(path):
    """Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks"""
    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    # split audio sound where silence is 500 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 650,
        # adjust this per requirement
        silence_thresh = sound.dBFS-12,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
            continue
        else:
            text = f"{text.capitalize()}. "
            #print(chunk_filename, ":", text)
            blob=TextBlob(text,classifier=cl)
            k=blob.classify()
            print(k)
            whole_text += text
    # return the text for all chunks detected
    return whole_text
print("1.Audio\n2.Image\n3.PDF\nEnter your choice:",end='')
choice=int(input("Enter your choice:"))
if(choice==1):
    path=input("Enter the path of the audio file:")
    whole_text=get_large_audio_transcription_on_silence(path)
    #sendic=sentiment.polarity_scores(whole_text)
    #print(sendic['compound'])
    blob=TextBlob(whole_text,classifier=cl)
    k=blob.classify()
    #print(k)
    #cl.show_informative_features(20)
elif(choice==2):
    img_path=input("Enter the path of the image:")
    img=cv2.imread(img_path)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_text=pytesseract.image_to_string(img_rgb)
    blob=TextBlob(img_text,classifier=cl)
    k=blob.classify()
    print(k)
else:
    path=input("Enter the path of the pdf file:")
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text+=page.get_text()
    blob=TextBlob(text,classifier=cl)
    k=blob.classify()
    print(k)
