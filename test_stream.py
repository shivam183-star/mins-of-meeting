import websocket
import wave
import json
import time
import threading

def on_message(ws, message):
    msg = json.loads(message)
    print(f"\n[{time.strftime('%H:%M:%S')}] Received from server: {msg['type']}")
    data = msg.get("data")
    if data:
        print(f" Chunk index: {data.get('chunk_index')}")
        print(f" Audio processed: {data.get('audio_duration'):.2f}s")
        print(f" Speech latency: {data.get('speech_latency'):.2f}s")
        print(f" LLM latency: {data.get('llm_latency'):.2f}s")
        print(f" Total latency: {data.get('total_latency'):.2f}s")
        mom = data.get("mom")
        if mom:
            print("\nIncremental MoM Update:")
            print(json.dumps(mom, indent=2))
        else:
            print("No MoM data.")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")

def on_open(ws):
    def run(*args):
        try:
            # We will read sample3.wav and stream it in chunks of 1s (16000 * 2 bytes)
            with wave.open("C:\\MoM Fresh\\sample3.wav", 'rb') as wf:
                framerate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                print(f"Audio params: {framerate}Hz, {channels}ch, {sampwidth}B")
                
                total_frames = wf.getnframes()
                processed = 0
                
                print("Starting to stream audio (simulating fast-forwarded real-time)...")
                while processed < total_frames:
                    data = wf.readframes(framerate) # read 1 second
                    if not data:
                        break
                    
                    ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                    processed += framerate
                    
                    # Fast-forward 10x for testing speed (wait 0.1s instead of 1.0s)
                    time.sleep(0.1)
                
                print("Finished streaming audio. Sending end_meeting event.")
                ws.send(json.dumps({"type": "end_meeting"}))
                
        except Exception as e:
            print(f"Stream error: {e}")
            ws.close()
    
    threading.Thread(target=run).start()

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8000/meetings/stream",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_forever()
