---
title: Audio understanding
description: Starting with Gemma 3n, you can use audio directly into your prompts and workflows.
models: [gemma-4, gemma-3n]
requires: audio
frameworks: [transformers]
colab: https://colab.research.google.com/github/google-gemma/docs/blob/main/docs/notebooks/audio.ipynb
---

# Audio understanding

Starting with [Gemma 3n](../../models/gemma-3n/index.md), you can use audio directly into your prompts and workflows. Audio and spoken language are rich sources of data for capturing user intents, recording information about the world around us, and understanding
specific problems to be solved.

This guide provides an overview of the audio processing capabilities of [Gemma 4](../../models/gemma-4/index.md), including automatic speech recognition (ASR), translation, and general speech understanding.


## Install Python packages

Install the Hugging Face libraries required for running the Gemma model and making requests.

```bash { .notebook-cell }
# Install PyTorch & other libraries
pip install torch accelerate

# Install the transformers library
pip install "transformers>=5.10.1"
```

## Load Model

Use the `transformers` libraries to create an instance of a `processor` and `model` using the `AutoProcessor` and `AutoModelForImageTextToText` classes as shown in the following code example:

```python { .notebook-cell .model-swap }
MODEL_ID = "google/gemma-4-E2B-it"

from transformers import pipeline

pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)
```

``` { .no-copy .output }
config.json:   0%|          | 0.00/4.95k [00:00<?, ?B/s]
model.safetensors:   0%|          | 0.00/10.2G [00:00<?, ?B/s]
Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]
generation_config.json:   0%|          | 0.00/208 [00:00<?, ?B/s]
processor_config.json:   0%|          | 0.00/1.69k [00:00<?, ?B/s]
chat_template.jinja:   0%|          | 0.00/17.3k [00:00<?, ?B/s]
tokenizer_config.json:   0%|          | 0.00/2.10k [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/32.2M [00:00<?, ?B/s]
```

## Audio data

Digital audio data can come in many formats and levels of resolution. The actual audio formats you can use with Gemma, such as MP3 and WAV formats, are determined by the framework you choose to convert sound data into tensors. Here are some specific considerations for preparing audio data for processing with Gemma:

- **Token cost:** Each second of audio is 25 tokens for Gemma 4. (6.25 tokens for Gemma 3n).
- **Clip length:** Audio supports a maximum length of 30 seconds.
- **Audio channels:** Audio data is processed as a single audio channel. If you are using multi-channel audio, such as left and right channels, consider reducing the data to a single channel by removing channels or combining the sound data into a single channel.
- **Technical Encoding:**
  - **Sample Rate:** 16kHz
  - **Bit Depth:** 32-bit float format, with samples normalized within the range of \[-1, 1\].

If the audio data you plan to process is significantly different from the input
processing, particularly in terms of channels, sample rate and bit depth,
consider resampling or trimming your audio data to match the data
resolution handled by the model.

## Audio encoding

While high-level libraries (such as Hugging Face `AutoProcessor`) often handle audio preprocessing automatically, you may sometimes need to implement custom encoding.

When encoding audio data with your own code implementation for use with Gemma, you should follow the recommended conversion process. If you are working
with audio files encoded in a specific format, such as MP3 or WAV encoded data,
you must first decode these to samples using a library such as `ffmpeg`. Once
the data is decoded, convert the audio into mono-channel, 16 kHz float32
waveforms in the range \[-1, 1\]. For example, if you are working with stereo
signed 16-bit PCM integer WAV files at 44.1 kHz, follow these steps:

- Resample the audio data to 16 kHz
- Downmix from stereo to mono by averaging the 2 channels
- Convert from int16 to float32, and divide by 32768.0 to scale to the range \[-1, 1\]

!!! Note
    When resampling audio to 16 kHz, you should use a Fourier method for best results, such as `scipy.signal.resample` or `librosa.sample(res_type ='scipy')`.

## Speech to text

Gemma 4 E2B, E4B, and 12B Unified are trained for multilingual speech recognition, allowing you to transcribe audio input in various languages into text.

Use the following prompt structure for **Audio Speech Recognition (ASR)**.

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
*   Only output the transcription, with no newlines.
*   When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

The following code examples show how to prompt the model to transcribe text from audio files using Hugging Face Transformers:

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

RESOURCE_URL_PREFIX = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            #{"type": "text", "text": "Transcribe the following speech segment in English into English text. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal1.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
I woke up early today feeling really fresh the morning light was beautiful and I enjoyed a nice cup of coffee<turn|>
```

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 1024
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Give me a concise overview of these audio files."},
            {"type": "text", "text": "journal1:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal1.wav"},
            {"type": "text", "text": "journal2:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal2.wav"},
            {"type": "text", "text": "journal3:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal3.wav"},
            {"type": "text", "text": "journal4:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal4.wav"},
            {"type": "text", "text": "journal5:"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal5.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
Here is a concise overview of each audio file:

**journal1:** The speaker describes a fresh and peaceful day, enjoying a cup of coffee.
**journal2:** The speaker had a perfect day at the park, including a walk and watching cherry blossoms.
**journal3:** The speaker finished the day with a good book, feeling grateful for simple moments.
**journal4:** The speaker returned from work and noted the beautiful night sky and a clear view from the train.
**journal5:** The speaker had a great lunch with an old friend, which was a pleasant way to catch up and made their day.
<turn|>
```

## Automated speech translation

Gemma 4 E2B, E4B and 12B Unified are trained for multilingual speech translation tasks, allowing you to translate spoken audio directly into another language.

Use the following prompt structure for **Automatic Speech Translation (AST)**.

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

The following code examples show how to prompt the model to translate spoken audio into text using Hugging Face Transformers:

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English, then translate it into Korean. When formatting the answer, first output the transcription in English, then one newline, then output the string 'Korean: ', then the translation in Korean."},
            {"type": "audio", "audio": "https://ai.google.dev/gemma/docs/audio/roses-are.wav"},
        ]
    }
]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
Roses are red, violets are blue.
Korean: 장미는 빨갛고, 제비꽃은 파랗다.<turn|>
```

## Automatic Speech Translation / Automatic Speech Recognition

Try this by yourself

```bash { .notebook-cell }
pip install ipywebrtc
```

Press the circle button and start speaking. Click the circle button again when you are finished. The widget will immediately begin to play back what it captured.

```python { .notebook-cell }
from google.colab import output
output.enable_custom_widget_manager()

from ipywebrtc import AudioRecorder, CameraStream

camera = CameraStream(constraints={'audio': True,'video':False})
recorder = AudioRecorder(stream=camera)
recorder
```

``` { .no-copy .output }
AudioRecorder(audio=Audio(value=b'', format='webm'), stream=CameraStream(constraints={'audio': True, 'video': …
```

Convert webm file to wav format that PyTorch can understand.

```python { .notebook-cell }
with open('/content/recording.webm', 'wb') as f:
    f.write(recorder.audio.value)
!ffmpeg -i /content/recording.webm /content/recording.wav -y
```

``` { .no-copy .output }
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, matroska,webm, from '/content/recording.webm':
  Metadata:
    encoder         : Chrome
  Duration: 00:00:03.00, start: 0.000000, bitrate: 132 kb/s
  Stream #0:0(eng): Audio: opus, 48000 Hz, mono, fltp (default)
Stream mapping:
  Stream #0:0 -> #0:0 (opus (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '/content/recording.wav':
  Metadata:
    ISFT            : Lavf58.76.100
  Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, mono, s16, 768 kb/s (default)
    Metadata:
      encoder         : Lavc58.134.100 pcm_s16le
size=     287kB time=00:00:02.99 bitrate= 783.7kbits/s speed=79.4x    
video:0kB audio:287kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.026552%
```

### ASR

```python { .notebook-cell }
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64
gen_kwargs = dict(generation_config=config)

messages = [{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
    {"type": "audio", "audio": "/content/recording.wav"},
  ]
}]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
How can I get to the station?<turn|>
```

### AST

```python { .notebook-cell }
messages = [{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe the following speech segment in English, then translate it into Korean. When formatting the answer, first output the transcription in English, then one newline, then output the string 'Korean: ', then the translation in Korean."},
    {"type": "audio", "audio": "/content/recording.wav"},
  ]
}]

outputs = pipe(messages, return_full_text=False, generate_kwargs=gen_kwargs)
print(outputs[0]['generated_text'])
```

``` { .no-copy .output }
How can I get to the station?
Korean: 역에 어떻게 가나요?<turn|>
```

## Summary

In this guide, you learned how to process audio using Gemma 4 models. The examples demonstrated how to perform Speech-to-Text (ASR) to transcribe spoken language, as well as Automated Speech Translation (AST) to translate spoken audio directly into another language. You also saw how to capture audio from a microphone in a notebook environment for processing.
## What's next

<div class="card-grid">
<a class="card" href="../prompt-formatting.md">
  <span class="card-icon">{{ icon('lucide/braces') }}</span>
  <div class="card-title">Prompt formatting</div>
  <p class="card-desc">The control tokens that structure a Gemma prompt.</p>
</a>
<a class="card" href="mtp.md">
  <span class="card-icon">{{ icon('lucide/zap') }}</span>
  <div class="card-title">Multi-Token Prediction</div>
  <p class="card-desc">Speculative decoding for faster inference at identical quality.</p>
</a>
<a class="card" href="../../deployment/index.md">
  <span class="card-icon">{{ icon('lucide/play') }}</span>
  <div class="card-title">Run Gemma</div>
  <p class="card-desc">Every engine that runs Gemma, from a laptop to a cluster.</p>
</a>
</div>
