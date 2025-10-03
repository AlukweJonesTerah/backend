from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, os, requests, base64, logging, asyncio, uvicorn
from typing import Optional
from pydub import AudioSegment
from datetime import datetime
import json
import resource

# Set seed for consistent language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Configure GRPC for low-resource environments BEFORE importing Google clients
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "False"
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["GRPC_MAX_IO_THREADS"] = "1"
os.environ["GRPC_MINIMUM_CHANNELS"] = "1"

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, memory monitoring limited")

# Railway-specific optimizations
if os.getenv("RAILWAY_ENVIRONMENT"):
    logger.info("🚆 Running on Railway - applying optimizations")
    
    # Set memory limits
    try:
        memory_limit = 450 * 1024 * 1024  # 450MB limit
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        logger.info("Memory limits set for Railway free tier")
    except:
        logger.warning("Could not set memory limits")

def optimize_thread_pool():
    """Optimize thread pool for Railway free tier"""
    import concurrent.futures
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Initialize optimized executor
executor = optimize_thread_pool()

def check_memory_usage():
    """Check current memory usage"""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
    
    # Fallback using resource module
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024
        estimated_percent = (memory_mb / 512) * 100
        return min(estimated_percent, 100)
    except:
        return 0

# Pydantic models
class ChatbotRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    original_transcript: str
    corrected_text: str
    session_id: Optional[str] = None

app = FastAPI(
    title="AgriWatt Voice Bot API", 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
CHATBOT_URL = os.getenv("CHATBOT_URL", "https://agriwatthub.com/chatbot-api.php")
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 2 * 1024 * 1024))  # 2MB limit
PORT = int(os.getenv("PORT", 8000))

# Global variables for Google clients
speech_client = None
tts_client = None
clients_initialized = False
clients_lock = asyncio.Lock()

async def initialize_google_clients():
    """Initialize Google Cloud clients on first use with thread safety"""
    global speech_client, tts_client, clients_initialized
    
    async with clients_lock:
        if clients_initialized:
            return True
            
        try:
            # Method 1: Base64 encoded credentials
            creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
            if creds_base64:
                from google.oauth2 import service_account
                from google.cloud import speech, texttospeech
                
                creds_json = base64.b64decode(creds_base64).decode('utf-8')
                credentials_dict = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                
                # Initialize clients
                speech_client = speech.SpeechClient(credentials=credentials)
                tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
                
                clients_initialized = True
                logger.info("✅ Google Cloud clients initialized from environment variable")
                return True
            
            logger.error("❌ No Google Cloud credentials found")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")
            return False

async def ensure_google_clients():
    """Ensure Google clients are initialized before use"""
    if not clients_initialized:
        return await initialize_google_clients()
    return speech_client is not None and tts_client is not None

def reduce_audio_quality(audio_content: bytes) -> bytes:
    """Reduce audio quality to save memory"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        audio = AudioSegment.from_file(temp_in_path)
        
        # Reduce quality for memory conservation
        audio = audio.set_frame_rate(8000)
        audio = audio.set_channels(1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            audio.export(temp_out.name, format="wav", 
                        parameters=["-ac", "1", "-ar", "8000", "-acodec", "pcm_s16le"])
            
            with open(temp_out.name, "rb") as f:
                reduced_audio = f.read()
        
        # Cleanup
        for path in [temp_in_path, temp_out.name]:
            if os.path.exists(path):
                os.unlink(path)
                
        return reduced_audio
        
    except Exception as e:
        logger.warning(f"Audio quality reduction failed: {e}")
        return audio_content

@app.on_event("startup")
async def startup_event():
    """Startup optimization for Railway"""
    logger.info("🚀 Starting AgriWatt Voice Bot")
    logger.info("Google Cloud clients will be initialized on first use")
    logger.info(f"✅ Service ready on port {PORT}")

def enhanced_language_detection(text: str) -> str:
    """Enhanced multilingual detection with agricultural focus"""
    try:
        detected_lang = detect(text.lower())
        
        # Agricultural terms for language detection
        agricultural_terms = {
            'en': ['farm', 'crop', 'irrigation', 'harvest', 'soil', 'weather', 'agriculture'],
            'sw': ['shamba', 'mazao', 'umwagiliaji', 'mvua', 'udongo', 'kilimo'],
            'fr': ['ferme', 'culture', 'irrigation', 'récolte', 'sol', 'agriculture'],
            'es': ['granja', 'cultivo', 'riego', 'cosecha', 'suelo', 'agricultura'],
        }
        
        text_lower = text.lower()
        for lang, terms in agricultural_terms.items():
            for term in terms:
                if term in text_lower:
                    return lang
        
        return detected_lang
        
    except:
        # Fallback to keyword detection
        language_keywords = {
            'sw': ['jambo', 'asante', 'sana', 'habari', 'pole', 'shamba', 'mazao', 'mkulima'],
            'fr': ['bonjour', 'merci', 'agriculture', 'ferme', 'cultiver', 'plante'],
            'es': ['hola', 'gracias', 'agricultura', 'granja', 'cultivar', 'planta'],
            'en': ['hello', 'thanks', 'farming', 'agriculture', 'crop', 'plant']
        }
        
        text_lower = text.lower()
        scores = {lang: 0 for lang in language_keywords}
        
        for lang, keywords in language_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[lang] += 1
        
        best_lang = max(scores.items(), key=lambda x: x[1])[0]
        return best_lang if scores[best_lang] > 0 else 'en'

def convert_audio_format(audio_content: bytes, target_format: str = "wav") -> bytes:
    """Convert audio to compatible format for Google Speech-to-Text"""
    try:
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
    """Provide domain-specific phrases to improve STT accuracy"""
    from google.cloud import speech
    return speech.SpeechContext(phrases=[
        "agriwatt", "agriculture", "farming", "kenya",
        "crops", "irrigation", "solar", "farm",
        "weather", "soil", "crop", "livestock",
        "maize", "coffee", "tea", "dairy",
        "smart farming", "climate", "water", "fertilizer"
    ])

def correct_domain_terms(transcript: str) -> str:
    """Correct common misrecognitions for domain-specific terms"""
    corrections = {
        "agree what": "agriwatt",
        "agree watt": "agriwatt",
        "agriculture what": "agriwatt",
        "agri what": "agriwatt",
    }
    
    transcript_lower = transcript.lower()
    for wrong, correct in corrections.items():
        if wrong in transcript_lower:
            transcript = transcript.replace(wrong, correct)
            logger.info(f"Corrected '{wrong}' to '{correct}'")
    
    return transcript

async def process_audio_to_text(audio_content: bytes, content_type: str = "audio/webm") -> str:
    """Convert audio to text with optimized settings"""
    try:
        if not await ensure_google_clients():
            raise HTTPException(status_code=503, detail="Speech service temporarily unavailable")
        
        # Memory check
        memory_usage = check_memory_usage()
        if memory_usage > 80:
            logger.warning(f"High memory usage ({memory_usage}%), reducing audio quality")
            audio_content = reduce_audio_quality(audio_content)

        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Audio recording too short")
        
        format_map = {
            'audio/webm': 'webm',
            'audio/wav': 'wav', 
            'audio/mpeg': 'mp3',
        }
        
        file_ext = format_map.get(content_type, 'webm')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name

        try:
            if content_type not in ['audio/webm', 'audio/wav']:
                audio_content = convert_audio_format(audio_content, "wav")
                content_type = 'audio/wav'
                file_ext = 'wav'
            
            from google.cloud import speech
            
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
                model="command_and_search",
                use_enhanced=False,
                speech_contexts=[get_speech_context()] 
            )
            
            # Use optimized executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor, 
                lambda: speech_client.recognize(config=config, audio=audio)
            )
            
            if not response.results:
                raise HTTPException(status_code=400, detail="Hakuna sauti iliyogundulika. Tafadhali rekodi tena.")
            
            transcript = response.results[0].alternatives[0].transcript
            transcript = correct_domain_terms(transcript)
            confidence = response.results[0].alternatives[0].confidence
            
            logger.info(f"Speech-to-Text: '{transcript}' (Confidence: {confidence:.2f})")
            
            if confidence < 0.3:
                logger.warning(f"Low confidence transcription: {confidence}")
            
            return transcript.strip()
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        raise HTTPException(status_code=400, detail="Shida katika kusikiliza sauti. Tafadhali jaribu tena baadae.")

async def call_php_chatbot(message: str, session_id: Optional[str] = None) -> dict:
    """Call the PHP chatbot backend"""
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

        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                executor, 
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

def get_voice_config(language_code: str) -> tuple:
    """Get appropriate voice configuration for multiple languages"""
    from google.cloud import texttospeech
    
    voice_map = {
        'en': ('en-US-Wavenet-F', texttospeech.SsmlVoiceGender.FEMALE, 'en-US'),
        'sw': ('en-US-Wavenet-D', texttospeech.SsmlVoiceGender.MALE, 'en-US'),
        'fr': ('fr-FR-Wavenet-A', texttospeech.SsmlVoiceGender.FEMALE, 'fr-FR'),
        'es': ('es-ES-Wavenet-B', texttospeech.SsmlVoiceGender.MALE, 'es-ES'),
    }
    
    voice_config = voice_map.get(language_code, voice_map['en'])
    return voice_config[0], voice_config[1], voice_config[2]

async def text_to_speech(text: str, language_code: str) -> Optional[str]:
    """Convert text to speech with multilingual support"""
    try:
        if not await ensure_google_clients():
            logger.error("TTS client not initialized")
            return None
            
        if not text or not text.strip():
            return None
            
        from google.cloud import texttospeech
        
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

        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
        )

        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        logger.info(f"TTS successful for language {language_code}")
        return audio_base64
        
    except Exception as e:
        logger.error(f"Text-to-speech conversion failed: {e}")
        return None

@app.post("/api/voice")
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Main voice processing endpoint"""
    try:
        # Check memory before processing
        memory_usage = check_memory_usage()
        if memory_usage > 85:
            raise HTTPException(
                status_code=503, 
                detail="Mfumo umejaa kwa sasa. Tafadhali jaribu tena baada ya dakika chache."
            )
        
        if not await ensure_google_clients():
            raise HTTPException(
                status_code=503, 
                detail="Huduma ya sauti haipatikani kwa sasa. Tafadhali jaribu tena baadaye."
            )
        
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Faili la sauti pekee linakubalika")
        
        audio_content = await audio.read()
        
        if len(audio_content) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail=f"Faili la sauti ni kubwa sana (kikomo ni {MAX_AUDIO_SIZE//1024//1024}MB)")
        
        if len(audio_content) < 1000:
            raise HTTPException(status_code=400, detail="Rekodi ni fupi sana. Rekodi kwa muda mrefu zaidi.")

        logger.info(f"Processing voice request: {len(audio_content)} bytes, session: {session_id}")

        # Step 1: Speech to Text
        try:
            transcript = await process_audio_to_text(audio_content, audio.content_type)
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

        # Step 2: Call PHP Chatbot Backend
        chatbot_response = await call_php_chatbot(transcript, session_id)
        
        reply_text = chatbot_response.get("reply", "Samahani, sijapata jibu kwa swali lako.")
        new_session_id = chatbot_response.get("session_id", session_id)
        success = chatbot_response.get("success", False)

        # Step 3: Text to Speech (conditional)
        audio_base64 = None
        reply_language = enhanced_language_detection(reply_text)
        
        # Only generate TTS if memory usage is low and text is reasonable length
        if check_memory_usage() < 70 and len(reply_text) < 500:
            audio_base64 = await text_to_speech(reply_text, reply_language)
        else:
            logger.info("Skipping TTS due to resource constraints")

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

        logger.info("Request completed successfully")
        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice processing: {e}")
        raise HTTPException(status_code=500, detail="Kuna hitilafu ya ndani. Tafadhali jaribu tena.")

@app.post("/api/chatbot")
async def chatbot_endpoint(request_data: ChatbotRequest):
    """Text-only endpoint that forwards to PHP chatbot"""
    try:
        user_message = request_data.message.strip()
        
        if not user_message:
            return {
                "success": False,
                "reply": "Samahani, hakuna ujumbe uliowasilishwa.",
                "session_id": request_data.session_id
            }
        
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
        clients_ready = await ensure_google_clients()
        
        return {
            "status": "healthy" if clients_ready else "degraded",
            "service": "AgriWatt Voice Bot",
            "google_services": "connected" if clients_ready else "disconnected",
            "memory_usage": f"{check_memory_usage()}%",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "AgriWatt Voice Bot", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    clients_ready = await ensure_google_clients()
    memory_usage = check_memory_usage()
    
    return {
        "message": "AgriWatt Voice Bot API",
        "version": "1.0.0",
        "description": "Voice interface for AgriWatt Hub chatbot",
        "status": "connected" if clients_ready else "disconnected",
        "platform": "Railway",
        "memory_usage": f"{memory_usage}%",
        "endpoints": {
            "voice": "/api/voice (POST)",
            "chatbot": "/api/chatbot (POST)", 
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        },
        "backend": CHATBOT_URL,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/voice/feedback")
async def voice_feedback(request_data: FeedbackRequest):
    """Collect user feedback to improve STT accuracy"""
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Log the correction for analysis
        logger.info(f"STT Correction - Original: '{request_data.original_transcript}' -> Corrected: '{request_data.corrected_text}'")
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': request_data.session_id,
            'original': request_data.original_transcript,
            'corrected': request_data.corrected_text,
        }
        
        feedback_file = 'logs/stt_feedback.json'
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        return JSONResponse({"success": True, "message": "Feedback recorded"})
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return JSONResponse({"success": False, "message": "Feedback failed"})

@app.get("/metrics")
async def metrics():
    """System metrics endpoint for monitoring"""
    try:
        memory_usage = check_memory_usage()
        
        metrics_data = {
            "memory_usage_percent": memory_usage,
            "google_clients": "initialized" if clients_initialized else "not_initialized",
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics_data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)