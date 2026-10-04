import websocket
import json
import time
import threading
import pyaudio
import os
import sys

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000  # 1 second of audio per packet

audio = pyaudio.PyAudio()
is_recording = True

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def on_message(ws, message):
    msg = json.loads(message)
    data = msg.get("data")
    if not data:
        return
        
    mom = data.get("mom")
    if mom:
        clear_screen()
        print("="*60)
        print("                 🔴 LIVE MINUTES OF MEETING")
        print("="*60)
        print(f"📌 TITLE: {mom.get('meeting_title', 'Meeting')}")
        print("-" * 60)
        
        print("\n📝 SUMMARY:")
        print(mom.get('summary', ''))
        
        print("\n🗣 PARTICIPANTS:")
        print(", ".join(mom.get('participants', [])))
        
        print("\n💬 KEY DISCUSSION POINTS:")
        for pt in mom.get('key_discussion_points', []):
            print(f"   • {pt}")
            
        print("\n✅ DECISIONS:")
        for dec in mom.get('decisions', []):
            print(f"   • {dec}")
            
        print("\n🚀 ACTION ITEMS:")
        for action in mom.get('action_items', []):
            owner = action.get('owner') or "Unassigned"
            deadline = action.get('deadline') or "None"
            print(f"   [ ] {action.get('task')} (Owner: {owner}, Due: {deadline})")
            
        print("\n❓ OPEN QUESTIONS:")
        for q in mom.get('open_questions', []):
            print(f"   • {q}")
            
        print("\n" + "="*60)
        print(f"Last updated: Chunk {data.get('chunk_index')} | Audio processed: {data.get('audio_duration'):.1f}s")
        print("Listening for next 60s of audio... (Press Ctrl+C to stop)")

def on_error(ws, error):
    print("\n[WebSocket Error]:", error)

def on_close(ws, close_status_code, close_msg):
    global is_recording
    is_recording = False
    print("\n### Disconnected from server ###")

def on_open(ws):
    def record_and_stream():
        try:
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
            print("\n🎙️ Microphone active. Listening and streaming to server...")
            print("Speak normally. The MoM will update approximately every 60 seconds.")
            print("(Press Ctrl+C to stop recording and generate final MoM)\n")
            
            while is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                
        except Exception as e:
            print(f"\n[Recording Error]: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            
    # Start the recording thread
    threading.Thread(target=record_and_stream, daemon=True).start()

if __name__ == "__main__":
    clear_screen()
    print("="*60)
    print("           🎤 STARTING LIVE MEETING CAPTURE")
    print("="*60)
    print("Waiting for connection to ws://localhost:8000/meetings/stream ...")
    
    # Disable tracing to keep console clean
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8000/meetings/stream",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping meeting... Sending final audio to server.")
        is_recording = False
        try:
            ws.send(json.dumps({"type": "end_meeting"}))
            # Give it a couple of seconds to receive the final update
            time.sleep(3)
        except:
            pass
        ws.close()
        audio.terminate()
        print("Exited.")
        sys.exit(0)
