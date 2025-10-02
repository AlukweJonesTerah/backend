from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech, texttospeech
from langdetect import detect, DetectorFactory
from pydantic import BaseModel
import tempfile, os, requests, base64, logging, asyncio, uvicorn, socket
from typing import Optional
from pydub import AudioSegment
from pydub.utils import mediainfo
from datetime import datetime, timedelta
import json

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Pydantic models
class ChatbotRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

app = FastAPI(title="AgriWatt Voice Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost/CompanyWeb1/chatbot-api.php")
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB limit

# Initialize Google Cloud clients
speech_client = None
tts_client = None

def initialize_google_clients():
    global speech_client, tts_client
    try:
        # Method 1: Environment variable with file path
        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        if creds_path and os.path.exists(creds_path):
            speech_client = speech.SpeechClient.from_service_account_file(creds_path)
            tts_client = texttospeech.TextToSpeechClient.from_service_account_file(creds_path)
            logger.info("Google Cloud clients initialized from file")
            return
        
        # Method 2: Base64 encoded credentials in environment variable
        creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if creds_base64:
            import base64
            from google.oauth2 import service_account
            
            creds_json = base64.b64decode(creds_base64).decode('utf-8')
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(creds_json)
            )
            speech_client = speech.SpeechClient(credentials=credentials)
            tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Google Cloud clients initialized from environment variable")
            return
        
        # Method 3: Default location (for development)
        default_path = "./google-credentials.json"
        if os.path.exists(default_path):
            speech_client = speech.SpeechClient.from_service_account_file(default_path)
            tts_client = texttospeech.TextToSpeechClient.from_service_account_file(default_path)
            logger.info("Google Cloud clients initialized from default path")
            return
            
        raise Exception("No Google Cloud credentials found")
        
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud clients: {e}")

# Initialize on startup
initialize_google_clients()

def enhanced_language_detection(text: str) -> str:
    """Enhanced multilingual detection with agricultural focus"""
    try:
        detected_lang = detect(text.lower())
        
        # Boost confidence for agricultural terms
        agricultural_terms = {
            'en': ['farm', 'crop', 'irrigation', 'harvest', 'soil', 'weather'],
            'sw': ['shamba', 'mazao', 'umwagiliaji', 'mvua', 'udongo'],
            'fr': ['ferme', 'culture', 'irrigation', 'récolte', 'sol'],
            'es': ['granja', 'cultivo', 'riego', 'cosecha', 'suelo']
        }
        
        text_lower = text.lower()
        for lang, terms in agricultural_terms.items():
            for term in terms:
                if term in text_lower:
                    # If agricultural term found, boost this language
                    if lang == detected_lang:
                        return detected_lang
                    else:
                        # Consider switching if agricultural term is strong
                        logger.info(f"Agricultural term '{term}' suggests language: {lang}")
                        return lang
        
        return detected_lang
        
    except:
        # Fallback to keyword detection
        language_keywords = {
            'sw': ['jambo', 'asante', 'sana', 'habari', 'pole', 'shamba', 'mazao', 'mkulima'],
            'fr': ['bonjour', 'merci', 'agriculture', 'ferme', 'cultiver', 'plante'],
            'es': ['hola', 'gracias', 'agricultura', 'granja', 'cultivar', 'planta'],
            'pt': ['olá', 'obrigado', 'agricultura', 'fazenda', 'cultivar', 'planta'],
            'ar': ['مرحبا', 'شكرا', 'زراعة', 'مزرعة', 'يزرع', 'نبات'],
            'hi': ['नमस्ते', 'धन्यवाद', 'कृषि', 'खेत', 'उगाना', 'पौधा'],
            'en': ['hello', 'thanks', 'farming', 'agriculture', 'crop', 'plant']
        }
        
        text_lower = text.lower()
        scores = {lang: 0 for lang in language_keywords}
        
        for lang, keywords in language_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[lang] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0]
    

def convert_audio_format(audio_content: bytes, target_format: str = "wav") -> bytes:
    """Convert audio to compatible format for Google Speech-to-Text"""
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        # Convert using pydub
        audio = AudioSegment.from_file(temp_in_path)
        
        # Export to target format
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_format}") as temp_out:
            if target_format == "wav":
                audio.export(temp_out.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            else:
                audio.export(temp_out.name, format=target_format)
            
            # Read converted audio
            with open(temp_out.name, "rb") as f:
                converted_audio = f.read()
        
        # Cleanup
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return converted_audio
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return audio_content  # Return original if conversion fails

def get_speech_context():
    """Provide domain-specific phrases to improve STT accuracy"""
    return speech.SpeechContext(phrases=[
        "agriwatt", "agriwatt hub", "agriculture", "farming", "kenya",
        "crops", "irrigation", "solar", "technology", "farm",
        "precipitation", "weather", "soil", "crop", "livestock",
        "maize", "coffee", "tea", "dairy", "horticulture",
        "smart farming", "precision agriculture", "climate", "water", "fertilizer"
    ])

def correct_domain_terms(transcript: str) -> str:
    """Correct common misrecognitions for domain-specific terms"""
    corrections = {
        "agree what": "agriwatt",
        "agree watt": "agriwatt",
        "agriculture what": "agriwatt",
        "agri what": "agriwatt",
        "a agree": "agri",
        "precipitation": "precipitation",
        "irrigation": "irrigation",
        "horticulture": "horticulture",
        "maize": "maize",
        "fertilizer": "fertilizer"
    }
    
    transcript_lower = transcript.lower()
    for wrong, correct in corrections.items():
        if wrong in transcript_lower:
            transcript = transcript.replace(wrong, correct)
            logger.info(f"Corrected '{wrong}' to '{correct}'")
    
    return transcript


def get_voice_config(language_code: str) -> tuple:
    """
    Get appropriate voice configuration for multiple languages
    """
    voice_map = {
        'en': ('en-US-Wavenet-F', texttospeech.SsmlVoiceGender.FEMALE, 'en-US'),
        'sw': ('en-US-Wavenet-D', texttospeech.SsmlVoiceGender.MALE, 'en-US'),  # Better voice for Swahili
        'fr': ('fr-FR-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE, 'fr-FR'),
        'es': ('es-ES-Wavenet-B', texttospeech.SsmlVoiceGender.MALE, 'es-ES'),
        'pt': ('pt-PT-Wavenet-C', texttospeech.SsmlVoiceGender.FEMALE, 'pt-PT'),
        'ar': ('ar-XA-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE, 'ar-XA'),
        'hi': ('hi-IN-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE, 'hi-IN')
    }
    
    # Default to English if language not supported
    voice_config = voice_map.get(language_code, voice_map['en'])
    return voice_config[0], voice_config[1], voice_config[2]

def enhance_audio_quality(audio_content: bytes) -> bytes:
    """Enhance audio quality for better STT accuracy"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        # Use pydub to enhance audio
        audio = AudioSegment.from_file(temp_in_path)
        
        # Apply audio enhancements
        audio = audio.high_pass_filter(80)  # Remove low-frequency noise
        audio = audio.low_pass_filter(8000)  # Remove high-frequency noise
        audio = audio.normalize()  # Normalize volume
        audio = audio.compression_threshhold(threshold=-20.0, ratio=4.0)  # Compress dynamic range
        
        # Export enhanced audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            audio.export(temp_out.name, format="wav", parameters=[
                "-ac", "1", 
                "-ar", "16000",
                "-acodec", "pcm_s16le"
            ])
            
            with open(temp_out.name, "rb") as f:
                enhanced_audio = f.read()
        
        # Cleanup
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return enhanced_audio
        
    except Exception as e:
        logger.warning(f"Audio enhancement failed: {e}")
        return audio_content  # Return original if enhancement fails


async def process_audio_to_text(audio_content: bytes, content_type: str = "audio/webm") -> str:
    """Convert audio to text with better format handling"""
    try:
         # Enhance audio quality first
        enhanced_audio = enhance_audio_quality(audio_content)

        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Audio recording too short")
        
        # Determine audio format from content type
        format_map = {
            'audio/webm': 'webm',
            'audio/wav': 'wav', 
            'audio/mpeg': 'mp3',
            'audio/mp4': 'mp4',
            'audio/ogg': 'ogg'
        }
        
        file_ext = format_map.get(content_type, 'webm')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        try:
            # Convert to compatible format if needed
            if content_type not in ['audio/webm', 'audio/wav']:
                audio_content = convert_audio_format(audio_content, "wav")
                content_type = 'audio/wav'
                file_ext = 'wav'
            
            # Extract metadata
            metadata = extract_audio_metadata(temp_audio_path)
            sample_rate = metadata.get("sample_rate_hertz", 16000)
            channel_count = metadata.get("audio_channel_count", 1)
            
            logger.info(f"Audio processing - Format: {file_ext}, Sample rate: {sample_rate}, Channels: {channel_count}")
            
            # Configure encoding based on format
            encoding_map = {
                'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
                'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            }
            
            encoding = encoding_map.get(file_ext, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
            
            audio = speech.RecognitionAudio(content=audio_content)
            
            config = speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate,
                audio_channel_count=channel_count,
                language_code="en-US",
                alternative_language_codes=["sw-KE", "fr-FR", "es-ES", "pt-PT", "ar-XA", "hi-IN"],
                enable_automatic_punctuation=True,
                model="default",
                use_enhanced=True,  # Enable enhanced model
                speech_contexts=[get_speech_context()] 
            )
            
            # Use async pattern for Google Cloud client
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: speech_client.recognize(config=config, audio=audio)
            )
            
            if not response.results:
                raise HTTPException(status_code=400, detail="Hakuna sauti iliyogundulika. Tafadhali rekodi tena.")
            
            transcript = response.results[0].alternatives[0].transcript
            transcript = correct_domain_terms(transcript)
            confidence = response.results[0].alternatives[0].confidence
            
            logger.info(f"Speech-to-Text: '{transcript}' (Confidence: {confidence:.2f})")
            
            if confidence < 0.5:  # Low confidence threshold
                logger.warning(f"Low confidence transcription: {confidence}")
                raise HTTPException(status_code=400, detail="Sauti haikusikika vizuri. Tafadhali ongea wazi zaidi na karibu na kifaa.")
            
            return transcript.strip()
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        raise HTTPException(status_code=400, detail="Shida katika kusikiliza sauti. Tafadhali jaribu tena baada ya muda.")

def extract_audio_metadata(audio_path: str) -> dict:
    """
    Extract audio metadata such as sample rate and channel count.
    """
    try:
        info = mediainfo(audio_path)
        return {
            "sample_rate_hertz": int(info.get("sample_rate", 48000)),
            "audio_channel_count": int(info.get("channels", 2)),
        }
    except Exception as e:
        logger.error(f"Failed to extract audio metadata: {e}")
        return {"sample_rate_hertz": 48000, "audio_channel_count": 2}
    
async def call_php_chatbot(message: str, session_id: Optional[str] = None) -> dict:
    """
    Call the PHP chatbot backend and return the complete response
    """
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AgriWatt-Voice-Bot/1.0'
        }

        logger.info(f"Calling PHP chatbot: {CHATBOT_URL}")
        logger.info(f"Payload: {payload}")

        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: requests.post(CHATBOT_URL, json=payload, headers=headers, timeout=30)
            ),
            timeout=35
        )

        logger.info(f"PHP chatbot response status: {response.status_code}")
        response.raise_for_status()
        
        return response.json()
        
    except asyncio.TimeoutError:
        logger.error("PHP chatbot request timed out")
        return {"success": False, "reply": "Samahani, mfumo umechelewa. Jaribu tena baadaye."}
    except requests.RequestException as e:
        logger.error(f"PHP chatbot request failed: {e}")
        return {"success": False, "reply": "Samahani, kuna tatizo la kimtandao. Jaribu tena."}
    except Exception as e:
        logger.error(f"Unexpected error calling PHP chatbot: {e}")
        return {"success": False, "reply": "Samahani, kuna hitilafu. Jaribu tena."}

def text_to_speech(text: str, language_code: str) -> Optional[str]:
    """
    Convert text to speech with multilingual support
    """
    try:
        if not text or not text.strip():
            return None
            
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_name, voice_gender, tts_language = get_voice_config(language_code)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=tts_language,  # Use the specific language code for TTS
            name=voice_name,
            ssml_gender=voice_gender
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,
            pitch=0.0
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        logger.info(f"TTS successful for language {language_code}, audio size: {len(response.audio_content)} bytes")
        return audio_base64
        
    except Exception as e:
        logger.error(f"Text-to-speech conversion failed for {language_code}: {e}")
        return None
    
@app.post("/api/voice")
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Main voice processing endpoint:
    1. Convert speech to text
    2. Send text to PHP chatbot backend
    3. Convert response to speech
    4. Return text and audio response
    """
    try:
         # Check if Google Cloud clients are initialized
        if speech_client is None or tts_client is None:
            raise HTTPException(
                status_code=503, 
                detail="Voice service temporarily unavailable. Please try again later."
            )
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Faili la sauti pekee linakubalika")
        
        audio_content = await audio.read()
        
        if len(audio_content) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail="Faili la sauti ni kubwa sana (max 10MB)")
        
        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Rekodi ni fupi sana. Rekodi kwa muda mrefu zaidi.")

        logger.info(f"Processing voice request: {len(audio_content)} bytes, session: {session_id}")

        # Step 1: Speech to Text
        try:
            transcript = await process_audio_to_text(audio_content)
            if not transcript or len(transcript.strip()) < 2:
                raise HTTPException(status_code=400, detail="Mazungumzo yamekubalika lakini hayaeleweki. Tafadhali ongea wazi zaidi.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise HTTPException(status_code=400, detail="Shida katika kusikiliza sauti. Tafadhali jaribu tena.")

        # Detect language from transcript
        detected_lang = enhanced_language_detection(transcript)
        logger.info(f"Detected language: {detected_lang}")

        context_data = {
            "message": transcript,
            "session_id": session_id,
            "is_voice": True  # Flag for voice conversations
        }

        # Step 2: Call PHP Chatbot Backend
        chatbot_response = await call_php_chatbot(transcript, session_id)
        
        reply_text = chatbot_response.get("reply", "Samahani, sijapata jibu kwa swali lako.")
        new_session_id = chatbot_response.get("session_id", session_id)
        success = chatbot_response.get("success", False)

        # Step 3: Text to Speech
        reply_language = enhanced_language_detection(reply_text)
        audio_base64 = text_to_speech(reply_text, reply_language)

        # Prepare response
        response_data = {
            "success": success,
            "transcript": transcript,
            "reply": reply_text,
            "session_id": new_session_id,
            "language": reply_language,
        }
        
        if audio_base64:
            response_data["audio"] = audio_base64
        else:
            response_data["warning"] = "Sauti haijapatikana, lakini jibu la maandishi lipo"

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice processing: {e}")
        raise HTTPException(status_code=500, detail="Kuna hitilafu ya ndani. Tafadhali jaribu tena.")

@app.post("/api/chatbot")
async def chatbot_endpoint(request_data: ChatbotRequest):
    """
    Text-only endpoint that forwards to PHP chatbot
    Useful for testing and mixed voice/text applications
    """
    try:
        user_message = request_data.message.strip()
        
        if not user_message:
            return {
                "success": False,
                "reply": "Samahani, hakuna ujumbe uliowasilishwa.",
                "session_id": request_data.session_id
            }
        
        # Call PHP chatbot backend
        chatbot_response = await call_php_chatbot(user_message, request_data.session_id)
        
        return {
            "success": chatbot_response.get("success", False),
            "reply": chatbot_response.get("reply", "Samahani, kuna tatizo."),
            "session_id": chatbot_response.get("session_id", request_data.session_id)
        }
        
    except Exception as e:
        logger.error(f"Chatbot endpoint error: {e}")
        return {
            "success": False,
            "reply": "Samahani, kuna hitilafu ya ndani.",
            "session_id": request_data.session_id
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test PHP chatbot connectivity
        test_response = await call_php_chatbot("test")
        php_status = "connected" if test_response.get("success") else "error"
        
        return {
            "status": "healthy",
            "service": "AgriWatt Voice Bot",
            "php_chatbot": php_status,
            "php_chatbot_url": CHATBOT_URL
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "AgriWatt Voice Bot", 
            "php_chatbot": "disconnected",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AgriWatt Voice Bot API",
        "version": "1.0.0",
        "description": "Voice interface for AgriWatt Hub chatbot",
        "endpoints": {
            "voice": "/api/voice (POST)",
            "chatbot": "/api/chatbot (POST)", 
            "health": "/health (GET)"
        },
        "backend": CHATBOT_URL
    }

@app.post("/api/voice/feedback")
async def voice_feedback(
    original_transcript: str = Form(...),
    corrected_text: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Collect user feedback to improve STT accuracy"""
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Log the correction for analysis
        logger.info(f"STT Correction - Original: '{original_transcript}' -> Corrected: '{corrected_text}'")
        
        # Store in a file or database for future model improvements
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'original': original_transcript,
            'corrected': corrected_text,
            'difference': len([i for i in range(min(len(original_transcript), len(corrected_text))) 
                            if original_transcript[i] != corrected_text[i]])
        }
        
        # Append to feedback file
        feedback_file = 'logs/stt_feedback.json'
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        return JSONResponse({"success": True, "message": "Feedback recorded"})
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return JSONResponse({"success": False, "message": "Feedback failed"})

if __name__ == "__main__":
    port=8000
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is already in use!")
        print("💡 Try these solutions:")
        print("   1. Kill existing process: sudo lsof -t -i:8000 | xargs kill -9")
        print("   2. Use different port: python main.py --port 8001")
        exit(1)
    
    print(f"🚀 Starting AgriWatt Voice Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)