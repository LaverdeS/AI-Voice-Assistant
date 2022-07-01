import asyncio
import sys

import requests
import json
import sounddevice

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""

LAST_HEARD = ""


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results

        global LAST_HEARD
        for result in results:
            for alt in result.alternatives:
                print(f"\r{alt.transcript}", end="", flush=True)
                LAST_HEARD = alt.transcript


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
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
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
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())


def query(payload, API_URL, headers):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def text_assisting(context):
    model = "gpt2"
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    with open("webis_token.json", "r") as f:
        authorization_token = json.load(f)

    headers = {"Authorization": f"Bearer api_org_{authorization_token['authorization']}"}
    return query({"inputs": f"{context}"}, API_URL, headers,)


def run_conscious_state():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(basic_transcribe())
    print()
    loop.close()


if __name__ == "__main__":
    model_output = text_assisting("that is so ")
    generated_text = model_output[0]['generated_text'].replace("\n", "")
    print(generated_text)
    run_conscious_state()
