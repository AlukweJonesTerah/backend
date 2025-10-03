from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Railway-specific optimizations
if os.getenv("RAILWAY_ENVIRONMENT"):
    logger.info("Railway environment detected - applying optimizations")
    
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["GRPC_POLL_STRATEGY"] = "poll"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    
    if resource:
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (40, 80))
            memory_limit = 450 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            logger.info("Memory and thread limits set")
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")

# Global variables
http_session: Optional[httpx.AsyncClient] = None
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
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 10 * 1024 * 1024))
MIN_AUDIO_SIZE = int(os.getenv("MIN_AUDIO_SIZE", 5000))
PORT = int(os.getenv("PORT", 8000))

logger.info(f"MAX_AUDIO_SIZE: {MAX_AUDIO_SIZE/1024/1024:.1f}MB")
logger.info(f"MIN_AUDIO_SIZE: {MIN_AUDIO_SIZE/1024:.1f}KB")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_session
    
    logger.info("Starting AgriWatt Voice Bot")
    
    http_session = httpx.AsyncClient(
        timeout=30,
        limits=httpx.Limits(max_connections=3, max_keepalive_connections=1)
    )
    logger.info(f"Service ready on port {PORT}")
    
    yield
    
    if http_session:
        await http_session.aclose()
        logger.info("httpx session closed")

app = FastAPI(
    title="AgriWatt Voice Bot API", 
    version="1.0.3",
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
            http_session = httpx.AsyncClient(timeout=30)

        response = await http_session.post(CHATBOT_URL, json=payload, headers=headers)

        if response.status_code != 200:
            return {"success": False, "reply": "Service temporarily unavailable."}

        response_data = response.json()
        if not isinstance(response_data, dict):
            return {"success": False, "reply": "Invalid response."}

        response_data.setdefault("success", True)
        response_data.setdefault("reply", "No response available.")
        return response_data

    except Exception as e:
        logger.error(f"PHP chatbot error: {e}")
        return {"success": False, "reply": "Service error."}

def check_memory_usage():
    try:
        return psutil.virtual_memory().percent
    except:
        return 0

def reduce_audio_quality(audio_content: bytes) -> bytes:
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
    try:
        return detect(text.lower())
    except:
        return 'en'

def convert_audio_format(audio_content: bytes, target_format: str = "wav") -> bytes:
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

def correct_domain_terms(transcript: str) -> str:
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
    voice_map = {
        'en': ('en-US-Wavenet-F', 'FEMALE', 'en-US'),
        'sw': ('en-US-Wavenet-D', 'MALE', 'en-US'),
        'fr': ('fr-FR-Wavenet-A', 'FEMALE', 'fr-FR'),
        'es': ('es-ES-Wavenet-B', 'MALE', 'es-ES'),
    }
    return voice_map.get(language_code, voice_map['en'])

def enhance_audio_quality(audio_content: bytes) -> bytes:
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
    """Convert audio to text using Google Cloud Speech REST API"""
    global http_session
    
    try:
        memory_usage = check_memory_usage()
        if memory_usage > 80:
            audio_content = reduce_audio_quality(audio_content)
        else:
            audio_content = enhance_audio_quality(audio_content)

        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short")
        
        # Get credentials
        creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if not creds_base64:
            raise HTTPException(status_code=503, detail="Speech service not configured")
        
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        
        creds_json = base64.b64decode(creds_base64).decode('utf-8')
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Get access token
        credentials.refresh(Request())
        access_token = credentials.token
        
        # Prepare audio
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # Map content type to encoding
        encoding_map = {
            'audio/webm': 'WEBM_OPUS',
            'audio/wav': 'LINEAR16',
            'audio/mpeg': 'MP3',
        }
        encoding = encoding_map.get(content_type, 'WEBM_OPUS')
        
        # Build request
        request_payload = {
            "config": {
                "encoding": encoding,
                "sampleRateHertz": 16000,
                "audioChannelCount": 1,
                "languageCode": "en-US",
                "alternativeLanguageCodes": ["sw-KE", "fr-FR", "es-ES"],
                "enableAutomaticPunctuation": True,
                "model": "default",
                "useEnhanced": True,
                "speechContexts": [{
                    "phrases": [
                        "agriwatt", "agriculture", "farming", "kenya", "crops", 
                        "irrigation", "solar", "maize", "coffee", "tea"
                    ]
                }]
            },
            "audio": {
                "content": audio_base64
            }
        }
        
        # Make REST API call
        api_url = "https://speech.googleapis.com/v1/speech:recognize"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        if not http_session:
            http_session = httpx.AsyncClient(timeout=30)
        
        logger.info("Calling Google Speech API (REST)...")
        response = await http_session.post(api_url, json=request_payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Google API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=400, detail="Speech recognition failed")
        
        result = response.json()
        
        if not result.get("results"):
            raise HTTPException(status_code=400, detail="No speech detected")
        
        transcript = result["results"][0]["alternatives"][0]["transcript"]
        transcript = correct_domain_terms(transcript)
        confidence = result["results"][0]["alternatives"][0].get("confidence", 0.0)
        
        logger.info(f"Transcript: '{transcript}' (Confidence: {confidence:.2f})")
        
        return transcript.strip()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Speech recognition failed")

async def text_to_speech(text: str, language_code: str) -> Optional[str]:
    """Convert text to speech using Google Cloud TTS REST API"""
    global http_session
    
    try:
        if not text or not text.strip():
            return None
        
        # Get credentials
        creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if not creds_base64:
            logger.error("GOOGLE_CREDENTIALS_BASE64 not set")
            return None
            
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        
        creds_json = base64.b64decode(creds_base64).decode('utf-8')
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Get access token
        credentials.refresh(Request())
        access_token = credentials.token
        
        voice_name, voice_gender, tts_language = get_voice_config(language_code)
        
        # Build request
        request_payload = {
            "input": {
                "text": text
            },
            "voice": {
                "languageCode": tts_language,
                "name": voice_name,
                "ssmlGender": voice_gender
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 0.9,
                "pitch": 0.0
            }
        }
        
        # Make REST API call
        api_url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        if not http_session:
            http_session = httpx.AsyncClient(timeout=30)
        
        logger.info("Calling Google TTS API (REST)...")
        response = await http_session.post(api_url, json=request_payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Google TTS error: {response.status_code}")
            return None
        
        result = response.json()
        
        if "audioContent" in result:
            return result["audioContent"]
        
        return None
        
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
        logger.info(f"Audio limits - MIN: {MIN_AUDIO_SIZE}, MAX: {MAX_AUDIO_SIZE}")
        
        memory_usage = check_memory_usage()
        if memory_usage > 85:
            logger.error(f"Memory overload: {memory_usage}%")
            raise HTTPException(status_code=503, detail="System overloaded")
        
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            logger.error(f"Invalid content type: {audio.content_type}")
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        audio_content = await audio.read()
        audio_size_kb = len(audio_content) / 1024
        
        logger.info(f"Received: {len(audio_content)} bytes ({audio_size_kb:.1f}KB)")
        logger.info(f"Type: {audio.content_type}")
        
        if len(audio_content) < MIN_AUDIO_SIZE:
            logger.error(f"Audio too short: {len(audio_content)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Audio too short - speak for at least 3 seconds"
            )
        
        if len(audio_content) > MAX_AUDIO_SIZE:
            logger.error(f"Audio too large: {len(audio_content)}")
            raise HTTPException(status_code=400, detail="Audio too large")
        
        logger.info("Audio validation passed")

        # Speech to Text
        logger.info("Starting speech-to-text...")
        transcript = await process_audio_to_text(audio_content, audio.content_type)
        logger.info(f"Transcript: '{transcript}'")

        # Call PHP Chatbot
        logger.info("Calling chatbot...")
        chatbot_response = await call_php_chatbot(transcript, session_id)
        
        reply_text = chatbot_response.get("reply", "No response available.")
        new_session_id = chatbot_response.get("session_id", session_id)
        success = chatbot_response.get("success", False)
        
        logger.info(f"Chatbot response received (success: {success})")

        # Text to Speech
        logger.info("Generating audio response...")
        reply_language = enhanced_language_detection(reply_text)
        audio_base64 = await text_to_speech(reply_text, reply_language)

        response_data = {
            "success": success,
            "transcript": transcript,
            "reply": reply_text,
            "session_id": new_session_id,
            "language": reply_language,
        }
        
        if audio_base64:
            response_data["audio"] = audio_base64
            logger.info("Audio response generated")

        logger.info("Voice processing completed")
        return JSONResponse(response_data)

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

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
    """Health check endpoint"""
    global _health_cache
    
    try:
        memory = psutil.virtual_memory()
        
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
            "google_cloud": "REST API",
            "php_chatbot": _health_cache["php_status"],
            "max_audio_size": f"{MAX_AUDIO_SIZE/1024/1024:.1f}MB",
            "min_audio_size": f"{MIN_AUDIO_SIZE/1024:.1f}KB",
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
        "version": "1.0.3",
        "platform": "Railway",
        "memory_usage": f"{check_memory_usage()}%",
        "max_audio_size": f"{MAX_AUDIO_SIZE/1024/1024:.1f}MB",
        "min_audio_size": f"{MIN_AUDIO_SIZE/1024:.1f}KB",
        "endpoints": {
            "voice": "/api/voice (POST)",
            "chatbot": "/api/chatbot (POST)", 
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    print(f"Starting AgriWatt Voice Server on port {PORT}...")
    print(f"Audio limits - MIN: {MIN_AUDIO_SIZE/1024:.1f}KB, MAX: {MAX_AUDIO_SIZE/1024/1024:.1f}MB")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,
        loop="asyncio",
        http="h11"
    )