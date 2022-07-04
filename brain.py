import asyncio
import sys
import time

import colorama
import requests
import json
import sounddevice
import pygame
import random

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.exceptions import BadRequestException
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError, ValidationError
from contextlib import closing
from tempfile import gettempdir

import os
import sys
import subprocess

# Create a client using the credentials and region defined in the [adminuser]
# section of the AWS credentials file (~/.aws/credentials).
colorama.init()
session = Session(profile_name="default")
polly = session.client("polly")
VOICES = ["Nicole", "Russell", "Amy", "Emma", "Brian", "Aditi", "Raveena", "Ivy",
          "Joanna", "Kendra", "Kimberly", "Salli", "Joey", "Justin", "Matthew", "Geraint"]
COLOR_CHOICE = [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
AGENTS_COLOR = {f"{name}": COLOR_CHOICE[ix] for ix, name in enumerate(VOICES)}

AGENT1_COLOR = '\033[96m'  # cyan
AGENT2_COLOR = '\033[95m'  # magenta
END_COLOR = '\033[0m'

LAST_HEARD = ""
LAST_SAID = ""
AGENT = ""
NUMBER_OF_LINES = 100
CHUNK_TIME_SIZE = 8
HEARING_STREAM = None
CLIENT_STREAM = None
SPEAKING = False


async def read_outloud(text: str):
    global HEARING_STREAM, AGENT
    # print("closing stream")
    HEARING_STREAM.close()

    start_color = AGENT1_COLOR if AGENTS_COLOR[AGENT] == 0 else AGENT2_COLOR

    try:
        # Request speech synthesis
        print(f"------------- said by: {start_color + AGENT + END_COLOR}", end="")
        # BRIAN
        response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=AGENT)
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)
        sys.exit(-1)

    # Access the audio stream from the response
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            output = os.path.join(gettempdir(), "speech.mp3")

            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
                # sys.exit(-1)
                # pass

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        sys.exit(-1)

    # Play the audio using the platform's default player
    if sys.platform == "win32":
        # os.startfile(output)  # start with default player
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(output)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # check if the file is playing
            # print("talking...")
            pass
        # pygame.event.wait()
        # TODO:: add one sound file to delete the other and so on or just remove the file after something is said
        pygame.mixer.quit()
        os.remove(output)
    else:
        # The following works on macOS and Linux. (Darwin = mac, xdg-open = linux).
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, output])


"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""


async def print_transcript(result):
    if result[-1] == ".":
        print(f"\r[hearing]... {result}", end="", flush=True)


async def new_line():  # new_line and speak_flag that is checked in speak_out_loud
    for _ in range(NUMBER_OF_LINES):
        await asyncio.sleep(CHUNK_TIME_SIZE)
        global SPEAKING
        SPEAKING = True
        print()


async def reply_async():
    for _ in range(NUMBER_OF_LINES):
        global SPEAKING, LAST_SAID, AGENT
        await asyncio.sleep(CHUNK_TIME_SIZE)
        SPEAKING = True
        print(f"[processing]... {LAST_HEARD}")

        if LAST_HEARD:
            try:
                AGENT = await random.choice(VOICES)
                start_color = AGENT1_COLOR if AGENTS_COLOR[AGENT] == 0 else AGENT2_COLOR
                speak = asyncio.create_task(text_assisting(LAST_HEARD))  # this writes in LAST_SAID
                await speak
                if 'error' not in LAST_SAID[0].keys():
                    print(f"[speaking]... {start_color + LAST_SAID[0]['generated_text'] + END_COLOR}")

                LAST_SAID = LAST_SAID[0]['generated_text']
            except KeyError:  # repeats the query in case model was not ready
                speak = asyncio.create_task(text_assisting(LAST_HEARD, model="gpt2"))  # this writes in LAST_SAID
                await speak
                if 'error' not in LAST_SAID[0].keys():
                    print(f"[speaking]... {start_color + LAST_SAID[0]['generated_text'] + END_COLOR}")
                LAST_SAID = LAST_SAID[0]['generated_text']
            finally:
                await read_outloud(LAST_SAID)
                print("--end of speech--")
                SPEAKING = False
                LAST_SAID = ""


async def new_agent(li):
    global AGENT
    AGENT = random.choice(li)


async def reply_async_single():
    global SPEAKING, LAST_SAID, AGENT
    await asyncio.sleep(CHUNK_TIME_SIZE)
    SPEAKING = True
    print(f"[processing]... {LAST_HEARD}")

    if LAST_HEARD:
        try:
            await new_agent(VOICES)
            # print("agent: ", AGENT)
            print("[bloom]...")
            start_color = AGENT1_COLOR if AGENTS_COLOR[AGENT] == 0 else AGENT2_COLOR
            speak = asyncio.create_task(text_assisting(LAST_HEARD, model="bloom"))  # this writes in LAST_SAID
            await speak
            # print("spoken...")
            if 'error' not in LAST_SAID[0].keys():
                LAST_SAID = LAST_SAID[0]['generated_text']
                print(f"[speaking]... {start_color + LAST_SAID + END_COLOR}")
            # else:
            #     print("error...")

        except KeyError:
            print("[gpt2]...")
            speak = asyncio.create_task(text_assisting(LAST_HEARD, model="gpt2"))  # this writes in LAST_SAID
            await speak
            if 'error' not in LAST_SAID[0].keys():
                LAST_SAID = LAST_SAID[0]['generated_text']
                print(f"[speaking]... {start_color + LAST_SAID + END_COLOR}")

        finally:
            await read_outloud(LAST_SAID)
            # await asyncio.sleep(3)
            SPEAKING = False
            LAST_SAID = ""
            # LAST_HEARD == LAST_SAID   # TODO: use this as mode 3 where it just continues the story alone


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results

        for result in results:
            # task = asyncio.create_task(new_line())
            for alt in result.alternatives:
                global LAST_HEARD
                LAST_HEARD = alt.transcript
                task_1 = asyncio.create_task(print_transcript(LAST_HEARD))
                await task_1


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    global SPEAKING, HEARING_STREAM

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )

    HEARING_STREAM = stream

    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.

    # while True:
    #     while SPEAKING:
    #         await asyncio.sleep(0.1)
    #         print("speaking...")
    #     print("(hearing...)")
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    # global SPEAKING
    # if not SPEAKING:
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region="us-west-2")

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )
    # Instantiate our handler and start processing events

    # await asyncio.gather(new_line(), reply_async())
    N = 5
    while N > 0:
        try:
            handler = MyEventHandler(stream.output_stream)
            await asyncio.gather(write_chunks(stream), handler.handle_events(), new_line(), reply_async())
        except BadRequestException:
            await asyncio.sleep(0.1)
            print(f"all_tasks: {asyncio.all_tasks}")


async def start_client():
    global CLIENT_STREAM
    client = TranscribeStreamingClient(region="us-west-2")
    # Start transcription to generate our async stream
    CLIENT_STREAM = await client.start_stream_transcription(language_code="en-US", media_sample_rate_hz=16000,
                                                            media_encoding="pcm")


async def cancel_all_tasks(tasks_list=[]):
    for task in tasks_list:
        task.cancel()


async def basic_transcribe2():
    N = 100
    print("\nWELCOME TO THE CONVERSATION ARENA!\n")
    while N > 0:
        task0 = asyncio.create_task(start_client())  # writes CLIENT STREAM
        await task0
        handler = MyEventHandler(CLIENT_STREAM.output_stream)
        task1 = asyncio.create_task(handler.handle_events())  # writes on LAST_HEARD
        task2 = asyncio.create_task(write_chunks(CLIENT_STREAM))  # prints LAST_HEARD
        task3 = asyncio.create_task(new_line())
        task4 = asyncio.create_task(reply_async_single())
        await task4
        await cancel_all_tasks([task0, task1, task2, task3, task4])
        print("")


async def text_assisting(context, model="bloom"):
    if model == "gpt2":
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
    elif model == "bloom":
        API_URL = f"https://api-inference.huggingface.co/models/bigscience/{model}"

    with open("webis_token.json", "r") as f:
        authorization_token = json.load(f)

    headers = {"Authorization": f"Bearer api_org_{authorization_token['authorization']}"}
    global LAST_SAID
    LAST_SAID = requests.post(API_URL, headers=headers, json={"inputs": f"{context}"}).json()


def run_conscious_state():
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(basic_transcribe())
    loop.run_until_complete(basic_transcribe2())
    # basic_transcribe is my main transciption co-routine where I await write_chunks & handle_events())

    loop.close()


if __name__ == "__main__":
    run_conscious_state()
