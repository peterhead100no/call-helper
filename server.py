import os
import multiprocessing
from contextlib import asynccontextmanager
import asyncio
import xml.etree.ElementTree as ET
from typing import Optional

import aiohttp
import psycopg2
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from working_pipeline import (
            get_call_info,
            download_audio,
            transcribe_audio,
            analyze_transcript_with_openai,
            save_structured_analysis_to_db,
            insert_call_record,
            restructure_analysis,
            check_sid_exists
        )

load_dotenv(override=True)

# Request/Response Models
class CallProcessRequest(BaseModel):
    call_sid: str

class CallProcessDetailedResponse(BaseModel):
    status: str
    message: str
    call_sid: str
    data: Optional[dict] = None

# ----------------- HELPERS ----------------- #

async def get_call_info_from_exotel(call_sid: str) -> dict:
    """
    Fetch call information from Exotel API (returns XML).
    
    Args:
        call_sid (str): The call SID to fetch information for
        
    Returns:
        dict: Parsed call information with status and other details, or error dict if failed
    """
    def _fetch_from_exotel():
        try:
            api_key = os.getenv("EXOTEL_API_KEY")
            api_token = os.getenv("EXOTEL_API_TOKEN")
            exotel_sid = os.getenv("EXOTEL_SID")
            exotel_subdomain = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
            
            if not all([api_key, api_token, exotel_sid]):
                return {"status": "error", "message": "Missing Exotel credentials"}
            
            # Build the Exotel API URL with credentials embedded
            url = f"https://{api_key}:{api_token}@{exotel_subdomain}/v1/Accounts/{exotel_sid}/Calls/{call_sid}"
            
            # Make the API request
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                try:
                    # Parse XML response
                    root = ET.fromstring(response.text)
                    
                    # Extract call data from XML
                    call_data = {}
                    call_element = root.find('Call')
                    
                    if call_element is not None:
                        for child in call_element:
                            call_data[child.tag] = child.text
                        
                        # Extract status from the parsed data
                        status = call_data.get('Status', 'unknown')
                        print(f"✅ Call info fetched for SID {call_sid}: Status = {status}")
                        
                        return {"status": "success", "call_status": status, "call_data": call_data}
                    else:
                        return {"status": "error", "message": "Call element not found in XML response"}
                        
                except ET.ParseError as e:
                    print(f"❌ Error parsing XML response for SID {call_sid}: {e}")
                    print(f"Response text: {response.text}")
                    return {"status": "error", "message": f"Failed to parse XML: {str(e)}"}
            else:
                print(f"❌ Error fetching call info for SID {call_sid}: Status {response.status_code}")
                print(f"Response text: {response.text}")
                return {"status": "error", "message": f"Exotel API returned status {response.status_code}"}
        
        except requests.exceptions.RequestException as error:
            print(f"❌ Request error while fetching call info from Exotel API: {error}")
            return {"status": "error", "message": f"Request failed: {str(error)}"}
        except Exception as error:
            print(f"❌ Error while fetching call info from Exotel API: {error}")
            return {"status": "error", "message": str(error)}
    
    return await asyncio.to_thread(_fetch_from_exotel)

# Helper function to fetch call data from database
def get_call_data_from_db(conn, call_sid):
    """Fetch call data from database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                sid, call_status, transcript, transcript_status,
                summary, summary_completed,
                information_requested, information_requested_completed,
                threat, threat_completed,
                priority, priority_completed,
                human_intervention, human_intervention_completed,
                satisfaction, satisfaction_completed,
                frustration, frustration_completed,
                nuisance, nuisance_completed,
                repeated_complaint, repeated_complaint_completed,
                next_best_action, next_best_action_completed,
                open_questions, open_questions_completed,
                pii_details, pii_details_completed,
                "Completed", recording_url, call_duration
            FROM "crm-ai-db" 
            WHERE sid=%s
        ''', (call_sid,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            # Build response data from database
            import json
            call_data = {
                "sid": result[0],
                "call_status": result[1],
                "transcript": result[2],
                "transcript_status": result[3],
                "summary": result[4],
                "summary_completed": result[5],
                "information_requested": result[6],
                "information_requested_completed": result[7],
                "threat": json.loads(result[8]) if result[8] else {},
                "threat_completed": result[9],
                "priority": json.loads(result[10]) if result[10] else {},
                "priority_completed": result[11],
                "human_intervention": json.loads(result[12]) if result[12] else {},
                "human_intervention_completed": result[13],
                "satisfaction": json.loads(result[14]) if result[14] else {},
                "satisfaction_completed": result[15],
                "frustration": json.loads(result[16]) if result[16] else {},
                "frustration_completed": result[17],
                "nuisance": json.loads(result[18]) if result[18] else {},
                "nuisance_completed": result[19],
                "repeated_complaint": json.loads(result[20]) if result[20] else {},
                "repeated_complaint_completed": result[21],
                "next_best_action": result[22],
                "next_best_action_completed": result[23],
                "open_questions": json.loads(result[24]) if result[24] else [],
                "open_questions_completed": result[25],
                "pii_details": json.loads(result[26]) if result[26] else {},
                "pii_details_completed": result[27],
                "completed": result[28],
                "recording_url": result[29],
                "call_duration": result[30],
            }
            return call_data
        return None
    except Exception as e:
        print(f"DB fetch error: {e}")
        return None

# ----------------- FASTAPI SETUP ----------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck():
    required_envs = [
        "EXOTEL_API_KEY",
        "EXOTEL_API_TOKEN",
        "EXOTEL_SID",
        "EXOTEL_PHONE_NUMBER",
    ]

    missing = [env for env in required_envs if not os.getenv(env)]

    if missing:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "missing_env_vars": missing},
        )

    return {
        "status": "ok",
        "service": "exotel-outbound-server",
    }

@app.get("/check_call_status")
async def check_call_status(sid: str) -> JSONResponse:
    """Check the status of a call by SID using Exotel API."""
    if not sid:
        raise HTTPException(
            status_code=400,
            detail="sid parameter is required",
        )
    
    try:
        result = await get_call_info_from_exotel(sid)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch call status: {result['message']}",
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "sid": sid,
                "call_status": result["call_status"],
                "call_duration": result["call_data"].get("Duration"),
                "recording_url": result["call_data"].get("RecordingUrl"),
                }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background processing function (from api.py)
def process_call_sync(call_sid):
    """
    Fully synchronous call processing.
    NO database writes unless ALL steps succeed.
    """
    conn = None
    audio_path = f"/tmp/{call_sid}.mp3"

    try:
        print(f"[Sync] Starting processing for call: {call_sid}")

        # STEP 1: Fetch call info
        call_data = get_call_info(call_sid)
        if not call_data:
            return False, "Call not found on Exotel", None

        recording_url = call_data.get("RecordingUrl")
        call_duration = call_data.get("Duration")

        if not recording_url:
            return False, "Recording not yet available", None

        # STEP 2: Download audio
        if not download_audio(recording_url, audio_path):
            return False, "Failed to download recording", None

        # STEP 3: Transcribe
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return False, "Failed to transcribe audio", None

        # STEP 4: Analyze
        structured_analysis = analyze_transcript_with_openai(transcript)
        if not structured_analysis:
            return False, "Failed to analyze transcript", None

        structured_analysis = restructure_analysis(structured_analysis)

        # STEP 5: SAVE TO DB (ONLY NOW)
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        # IMPORTANT: single transaction
        conn.autocommit = False

        if not insert_call_record(conn, call_sid):
            raise Exception("Failed to insert call record")

        save_structured_analysis_to_db(
            conn,
            call_sid,
            transcript,
            structured_analysis,
            recording_url,
            call_duration
        )

        conn.commit()

        return True, "Call processed successfully", {
            "transcript": transcript,
            "analysis": structured_analysis,
            "recording_url": recording_url,
            "call_duration": call_duration
        }

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Sync] ❌ Error: {e}")
        return False, str(e), None

    finally:
        if conn:
            conn.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/process-call", response_model=CallProcessDetailedResponse)
def process_call(request: CallProcessRequest):
    call_sid = request.call_sid

    if not call_sid:
        raise HTTPException(status_code=400, detail="call_sid is required")

    conn = None

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        # If already processed → return cached result
        if check_sid_exists(conn, call_sid):
            call_data = get_call_data_from_db(conn, call_sid)
            conn.close()

            return CallProcessDetailedResponse(
                status=call_data.get("call_status", "completed"),
                message="Call already processed",
                call_sid=call_sid,
                data=call_data
            )

        conn.close()

        # FULLY BLOCKING PROCESS
        success, message, result = process_call_sync(call_sid)

        if not success:
            return CallProcessDetailedResponse(
                status="failed",
                message=message,
                call_sid=call_sid,
                data=None
            )

        return CallProcessDetailedResponse(
            status="completed",
            message="Call processed successfully",
            call_sid=call_sid,
            data=result
        )

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get the number of workers from the environment variable or calculate based on CPU cores
    workers = int(os.getenv("UVICORN_WORKERS", (2 * multiprocessing.cpu_count()) + 1))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7863,
        workers=1  # Specify the number of workers
    )

