from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech, texttospeech
from langdetect import detect, DetectorFactory
from pydantic import BaseModel
from contextlib import asynccontextmanager
import tempfile, os, base64, logging, asyncio, uvicorn
from typing import Optional
from pydub import AudioSegment
from datetime import datetime
import json
import psutil
import httpx

# Safe import for Linux-only resource module
try:
    import resource
except ImportError:
    resource = None

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Railway-specific optimizations - MUST BE SET BEFORE IMPORTING GOOGLE CLOUD
if os.getenv("RAILWAY_ENVIRONMENT"):
    logger.info("🚆 Running on Railway - applying optimizations")
    
    # CRITICAL: Set these BEFORE importing Google Cloud libraries
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["GRPC_POLL_STRATEGY"] = "poll"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GRPC_THREADS"] = "1"
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    
    if resource:
        try:
            # Thread limit for Railway free tier
            resource.setrlimit(resource.RLIMIT_NPROC, (40, 80))
            # Memory limit
            memory_limit = 450 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            logger.info("Memory and thread limits set for Railway free tier")
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")

# Global variables
http_session: Optional[httpx.AsyncClient] = None
speech_client = None
tts_client = None

# Health check cache
_health_cache = {
    "last_check": None,
    "php_status": "unknown",
    "php_error": None
}
HEALTH_CACHE_SECONDS = 60

# Pydantic models
class ChatbotRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Environment variables
CHATBOT_URL = os.getenv("CHATBOT_URL", "https://agriwatthub.com/chatbot-api.php")
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 5 * 1024 * 1024))
PORT = int(os.getenv("PORT", 8000))

def initialize_google_clients():
    """Lazy initialization of Google Cloud clients"""
    global speech_client, tts_client
    
    if speech_client is not None and tts_client is not None:
        return  # Already initialized
    
    try:
        creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if not creds_base64:
            logger.error("GOOGLE_CREDENTIALS_BASE64 not set")
            return
            
        from google.oauth2 import service_account
        creds_json = base64.b64decode(creds_base64).decode('utf-8')
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json)
        )
        
        # Initialize with minimal gRPC channels
        speech_client = speech.SpeechClient(credentials=credentials)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        logger.info("✅ Google Cloud clients initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler"""
    global http_session
    
    logger.info("🚆 Starting AgriWatt Voice Bot on Railway")
    
    # Lazy initialization - only create clients when needed
    # This avoids creating gRPC threads during startup
    logger.info("⏭️ Deferring Google Cloud client initialization")
    
    # Create httpx session with strict limits
    http_session = httpx.AsyncClient(
        timeout=30,
        limits=httpx.Limits(max_connections=3, max_keepalive_connections=1)
    )
    logger.info("✅ Global httpx session created")
    logger.info(f"✅ AgriWatt Voice Bot ready on port {PORT}")
    
    yield  # Application runs here
    
    # Cleanup
    if http_session:
        await http_session.aclose()
        logger.info("🛑 httpx session closed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AgriWatt Voice Bot API", 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def call_php_chatbot(message: str, session_id: Optional[str] = None) -> dict:
    """Call PHP chatbot API"""
    global http_session
    
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AgriWatt-Voice-Bot/1.0'
        }

        if not http_session:
            http_session = httpx.AsyncClient(
                timeout=30,
                limits=httpx.Limits(max_connections=3, max_keepalive_connections=1)
            )

        response = await http_session.post(CHATBOT_URL, json=payload, headers=headers)

        if response.status_code != 200:
            return {"success": False, "reply": "Samahani, mfumo wa mazungumzo unakabiliwa na shida."}

        response_data = response.json()
        if not isinstance(response_data, dict):
            return {"success": False, "reply": "Samahani, jibu lisilotarajiwa."}

        response_data.setdefault("success", True)
        response_data.setdefault("reply", "Samahani, hakuna jibu.")
        return response_data

    except httpx.TimeoutException:
        return {"success": False, "reply": "Samahani, mfumo umechelewa."}
    except Exception as e:
        logger.error(f"PHP chatbot error: {e}")
        return {"success": False, "reply": "Samahani, kuna hitilafu isiyotarajiwa."}

def check_memory_usage():
    """Check memory usage"""
    try:
        return psutil.virtual_memory().percent
    except:
        return 0

def reduce_audio_quality(audio_content: bytes) -> bytes:
    """Reduce audio quality for memory optimization"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        audio = AudioSegment.from_file(temp_in_path)
        audio = audio.set_frame_rate(8000).set_channels(1).low_pass_filter(4000)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            audio.export(temp_out.name, format="wav", 
                        parameters=["-ac", "1", "-ar", "8000", "-acodec", "pcm_s16le"])
            with open(temp_out.name, "rb") as f:
                reduced_audio = f.read()
        
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return reduced_audio
    except Exception as e:
        logger.warning(f"Audio reduction failed: {e}")
        return audio_content

def enhanced_language_detection(text: str) -> str:
    """Enhanced language detection"""
    try:
        return detect(text.lower())
    except:
        return 'en'

def convert_audio_format(audio_content: bytes, target_format: str = "wav") -> bytes:
    """Convert audio format"""
    try:
        if check_memory_usage() > 70:
            audio_content = reduce_audio_quality(audio_content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        audio = AudioSegment.from_file(temp_in_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_format}") as temp_out:
            if target_format == "wav":
                audio.export(temp_out.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            else:
                audio.export(temp_out.name, format=target_format)
            
            with open(temp_out.name, "rb") as f:
                converted_audio = f.read()
        
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return converted_audio
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return audio_content

def get_speech_context():
    """Domain-specific speech context"""
    return speech.SpeechContext(phrases=[
        "agriwatt", "agriculture", "farming", "kenya", "crops", "irrigation", 
        "solar", "maize", "coffee", "tea", "smart farming"
    ])

def correct_domain_terms(transcript: str) -> str:
    """Correct common misrecognitions"""
    corrections = {
        "agree what": "agriwatt",
        "agree watt": "agriwatt",
        "agriculture what": "agriwatt"
    }
    
    for wrong, correct in corrections.items():
        if wrong in transcript.lower():
            transcript = transcript.replace(wrong, correct)
    
    return transcript

def get_voice_config(language_code: str) -> tuple:
    """Get voice configuration"""
    voice_map = {
        'en': ('en-US-Wavenet-F', texttospeech.SsmlVoiceGender.FEMALE, 'en-US'),
        'sw': ('en-US-Wavenet-D', texttospeech.SsmlVoiceGender.MALE, 'en-US'),
        'fr': ('fr-FR-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE, 'fr-FR'),
        'es': ('es-ES-Wavenet-B', texttospeech.SsmlVoiceGender.MALE, 'es-ES'),
    }
    return voice_map.get(language_code, voice_map['en'])

def enhance_audio_quality(audio_content: bytes) -> bytes:
    """Enhance audio quality"""
    try:
        if check_memory_usage() > 75:
            return audio_content

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        audio = AudioSegment.from_file(temp_in_path)
        audio = audio.high_pass_filter(80).normalize()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            audio.export(temp_out.name, format="wav", 
                        parameters=["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"])
            with open(temp_out.name, "rb") as f:
                enhanced_audio = f.read()
        
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return enhanced_audio
    except:
        return audio_content

async def process_audio_to_text(audio_content: bytes, content_type: str = "audio/webm") -> str:
    """Convert audio to text"""
    # Initialize Google clients on first use (lazy initialization)
    if speech_client is None:
        initialize_google_clients()
    
    if speech_client is None:
        raise HTTPException(status_code=503, detail="Speech service unavailable")
    
    try:
        memory_usage = check_memory_usage()
        if memory_usage > 80:
            audio_content = reduce_audio_quality(audio_content)
        else:
            audio_content = enhance_audio_quality(audio_content)

        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short")
        
        format_map = {
            'audio/webm': 'webm',
            'audio/wav': 'wav', 
            'audio/mpeg': 'mp3',
        }
        
        file_ext = format_map.get(content_type, 'webm')
        
        if content_type not in ['audio/webm', 'audio/wav']:
            audio_content = convert_audio_format(audio_content, "wav")
            file_ext = 'wav'
        
        encoding_map = {
            'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
        }
        
        encoding = encoding_map.get(file_ext, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
        
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=16000,
            audio_channel_count=1,
            language_code="en-US",
            alternative_language_codes=["sw-KE", "fr-FR", "es-ES"],
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
            speech_contexts=[get_speech_context()] 
        )
        
        # Use asyncio.to_thread for async gRPC call
        response = await asyncio.to_thread(speech_client.recognize, config=config, audio=audio)
        
        if not response.results:
            raise HTTPException(status_code=400, detail="No speech detected")
        
        transcript = response.results[0].alternatives[0].transcript
        transcript = correct_domain_terms(transcript)
        confidence = response.results[0].alternatives[0].confidence
        
        logger.info(f"STT: '{transcript}' (Confidence: {confidence:.2f})")
        
        if confidence < 0.5:
            raise HTTPException(status_code=400, detail="Low confidence")
        
        return transcript.strip()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        raise HTTPException(status_code=400, detail="Speech recognition failed")

def text_to_speech(text: str, language_code: str) -> Optional[str]:
    """Convert text to speech"""
    # Initialize Google clients on first use
    if tts_client is None:
        initialize_google_clients()
    
    if tts_client is None:
        return None
    
    try:
        if not text or not text.strip():
            return None
            
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_name, voice_gender, tts_language = get_voice_config(language_code)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=tts_language,
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

        return base64.b64encode(response.audio_content).decode('utf-8')
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return None

@app.post("/api/voice")
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Main voice processing endpoint"""
    try:
        memory_usage = check_memory_usage()
        if memory_usage > 85:
            raise HTTPException(status_code=503, detail="System overloaded")
        
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        audio_content = await audio.read()
        
        if len(audio_content) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail="Audio file too large")
        
        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short")

        # Speech to Text
        transcript = await process_audio_to_text(audio_content, audio.content_type)

        # Call PHP Chatbot
        chatbot_response = await call_php_chatbot(transcript, session_id)
        
        reply_text = chatbot_response.get("reply", "Samahani, sijapata jibu.")
        new_session_id = chatbot_response.get("session_id", session_id)
        success = chatbot_response.get("success", False)

        # Text to Speech
        reply_language = enhanced_language_detection(reply_text)
        audio_base64 = text_to_speech(reply_text, reply_language)

        response_data = {
            "success": success,
            "transcript": transcript,
            "reply": reply_text,
            "session_id": new_session_id,
            "language": reply_language,
        }
        
        if audio_base64:
            response_data["audio"] = audio_base64

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.post("/api/chatbot")
async def chatbot_endpoint(request_data: ChatbotRequest):
    """Text-only chatbot endpoint"""
    try:
        user_message = request_data.message.strip()
        
        if not user_message:
            return {
                "success": False,
                "reply": "No message provided",
                "session_id": request_data.session_id
            }
        
        chatbot_response = await call_php_chatbot(user_message, request_data.session_id)
        
        return {
            "success": chatbot_response.get("success", False),
            "reply": chatbot_response.get("reply", "Error"),
            "session_id": chatbot_response.get("session_id", request_data.session_id)
        }
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return {
            "success": False,
            "reply": "Internal error",
            "session_id": request_data.session_id
        }

@app.get("/health")
async def health_check():
    """Optimized health check with caching"""
    global _health_cache
    
    try:
        memory = psutil.virtual_memory()
        google_status = "connected" if speech_client and tts_client else "lazy_init"
        
        # Cache PHP chatbot checks
        now = datetime.now()
        if (_health_cache["last_check"] is None or 
            (now - _health_cache["last_check"]).total_seconds() > HEALTH_CACHE_SECONDS):
            
            try:
                test_response = await call_php_chatbot("health")
                _health_cache["php_status"] = "connected" if test_response.get("success") else "error"
                _health_cache["php_error"] = None if test_response.get("success") else test_response.get("reply")
                _health_cache["last_check"] = now
            except Exception as e:
                _health_cache["php_status"] = "error"
                _health_cache["php_error"] = str(e)
                _health_cache["last_check"] = now
        
        status = "healthy" if memory.percent < 90 else "unhealthy"
        
        return {
            "status": status,
            "service": "AgriWatt Voice Bot",
            "memory_usage": f"{memory.percent}%",
            "google_cloud": google_status,
            "php_chatbot": _health_cache["php_status"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AgriWatt Voice Bot API",
        "version": "1.0.1",
        "platform": "Railway",
        "memory_usage": f"{check_memory_usage()}%",
        "endpoints": {
            "voice": "/api/voice (POST)",
            "chatbot": "/api/chatbot (POST)", 
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    print(f"🚀 Starting AgriWatt Voice Server on port {PORT}...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,
        loop="asyncio",
        http="h11"
    )