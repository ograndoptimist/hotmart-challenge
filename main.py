from moviepy.editor import *

import torch

import whisper

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import VitsModel, AutoTokenizer


def save_txt(file: str,
             destine: str):
  """
    Save txt files.
  """
  try:
    with open(destine, "w") as file:
      file.write(file)
  except:
    return "Error"


def transcribe_audio(model: whisper.Whisper,
                     audio_path: str) -> str:
  """
      Transcribe audio into text.
  """
  result = model.transcribe(audio_path)
  text = result["text"]
  return text


def translate_text(model: MBartForConditionalGeneration,
                   tokenizer: MBart50TokenizerFast,
                   targeted_language: str,
                   portuguese_text: str) -> str:
  """
      Translate text from portuguese to english.
  """
  encoded = tokenizer(portuguese_text, return_tensors="pt")
  encoded = encoded.to(device)
  generated_tokens = (
      model
      .generate(**encoded,
                max_length=1200,
                forced_bos_token_id=tokenizer.lang_code_to_id[targeted_language]
      )
  )
  decoded_text = (
      tokenizer
      .batch_decode(generated_tokens, skip_special_tokens=True)
  )
  return decoded_text


def sintetize_text(model: VitsModel,
                   tokenizer: AutoTokenizer,
                   text_input: str):
  """
    Sintetize english text to english audio.
  """
  inputs = tokenizer(text_input,
                     return_tensors="pt",
                     max_length=5909).to(device)

  with torch.no_grad():
   output = model_tts(**inputs).waveform

  return output


def convert_to_audio(read_video_path: str,
                     write_audio_path: str) -> None:
  """
      Extract audio from a video.
  """
  videoclip = VideoFileClip(read_video_path)
  audioclip = videoclip.audio
  audioclip.write_audiofile(write_audio_path)


def cut_audio(read_audio_path: str,
              save_sample_audio_path: str,
              start: int,
              end: int,
              ) -> None:
  """
    Define a start and end of an audio and cut it.
  """
  audioclip = AudioFileClip(read_audio_path)
  segment = audioclip.subclip(start, end)
  segment.write_audiofile(save_sample_audio_path)


def main(start_video,
         end_video
         device,
         video_raw_path,
         audio_raw_path,
         audio_sample_raw_path):
  ## Preprocess the data
  convert_to_audio(
      read_video_path=video_raw_path,
      write_audio_path= audio_raw_path
  )

  cut_audio(
      read_audio_path= audio_raw_path,
      save_sample_audio_path=audio_sample_raw_path,
      start=start_video,
      end=end_video
  )


  ## Execute the transcription model
  transcription_model = whisper.load_model("medium")

  transcripted_pt_text = transcribe_audio(
    model=transcription_model,
    audio_path=audio_sample_raw_path
  )


  ## Execute the translation model
  translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
  translation_tokenizer.src_lang = "pt_XX"

  translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
  translation_model.to(device)

  translated_en_text = translate_text(
    model=translation_model,
    tokenizer=translation_tokenizer,
    targeted_language="en_XX",
    portuguese_text=transcripted_pt_text
  )

  ## Execute the TTS model
  model_tts = VitsModel.from_pretrained("facebook/mms-tts-eng")
  model_tts.to(device)

  translated_en_text = translate_text(
    model=translation_model,
    tokenizer=translation_tokenizer,
    targeted_language="en_XX",
    portuguese_text=transcripted_pt_text
  )

  tokenizer_tts = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

  en_audio = sintetize_text(
      model=model_tts,
      tokenizer=tokenizer_tts,
      text_input=translated_en_text
  )


  ## Save outputs
  save_txt(file=transcripted_pt_text,
           destine="outputs/transcripted_pt_text.txt")

  save_txt(file=translated_en_text,
           destine="outputs/translated_en_text.txt")

  save_txt(file=en_audio,
           destine="outputs/en_audio.wav")


if __name__ == "__main__":
   START_VIDEO = 0
   END_VIDEO = 3 * 60

   device = "cuda:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu"

   video_raw_path = "data/raw_data/case_ai.mp4"
   audio_raw_path = "data/raw_data/case_ai_full_audio.mp3"
   audio_sample_raw_path = "data/raw_data/case_ai_3min_sample_audio.mp3"

   main(START_VIDEO,
        END_VIDEO,
        device,
        video_raw_path,
        audio_raw_path,
        audio_sample_raw_path)

