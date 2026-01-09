import os
import psycopg2
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import time
import json

load_dotenv()

# Database
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Exotel
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_SUBDOMAIN = "api.exotel.in"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_URL = f"https://{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}@{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_SID}"

# ------------------ DATABASE & API FUNCTIONS ------------------

def check_sid_exists(conn, call_sid):
    """Check if sid exists in database"""
    try:
        cursor = conn.cursor()
        cursor.execute("""SELECT sid FROM "crm-ai-db" WHERE sid=%s""", (call_sid,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None
    except Exception as e:
        print(f"DB error checking sid: {e}")
        return False

def insert_call_record(conn, call_sid):
    """
    Insert a new call record into the database with initial status.
    This should only be called AFTER all processing steps have succeeded.
    """
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO "crm-ai-db" (sid, "Completed", call_status)
            VALUES (%s, FALSE, %s)
        ''', (call_sid, "in-progress"))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"DB insert error: {e}")
        conn.rollback()
        return False

def get_incomplete_sids(conn=None):
    try:
        if not conn:
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()
        cursor.execute("""SELECT sid FROM "crm-ai-db" WHERE "Completed" = FALSE and call_status='in-progress'""")
        results = cursor.fetchall()
        cursor.close()
        return [row[0] for row in results]
    except Exception as e:
        print(f"DB error: {e}")
        return []

def get_call_info(call_sid):
    try:
        url = f"{BASE_URL}/Calls/{call_sid}"
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            call_data = {}
            call_element = root.find('Call')
            if call_element is not None:
                for child in call_element:
                    call_data[child.tag] = child.text
            return call_data
        else:
            print(f"Exotel error {response.status_code}")
            return None
    except Exception as e:
        print(f"Exotel API error: {e}")
        return None

def download_audio(recording_url, output_path):
    try:
        auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN)
        response = requests.get(recording_url, auth=auth, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Audio download error: {e}")
        return False

def make_openai_client(timeout_sec=600):
    return OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=0,
        timeout=httpx.Timeout(connect=10, read=timeout_sec, write=60, pool=10)
    )

def analyze_transcript_with_openai(transcript_text):
    """
    Analyze transcript using OpenAI API with structured prompt.

    Args:
        transcript_text: The transcript text

    Returns:
        Dictionary with analysis results
    """
    if not transcript_text or transcript_text.strip() == "":
        return None

    prompt = f"""
You are a senior call QA, compliance, and triage analyst.

Your task:
Analyze the call transcript and return ONLY a valid JSON object that strictly follows the schema and rules below.
 - Do NOT use Unicode escape sequences (e.g. \\uXXXX)
========================
ABSOLUTE OUTPUT RULES
========================
1) Output MUST be valid JSON only.
   - Use double quotes only
   - No markdown
   - No comments
   - No trailing commas
   - No text before or after the JSON

2) Use ONLY the keys listed in the schema.
   - No extra keys
   - No missing keys

3) Enum fields MUST use one of the allowed values EXACTLY as written.
   - If evidence is insufficient, use "Unclear"
   - Do NOT guess or infer without evidence

4) Every classification field MUST include:
   - A short direct quote from the transcript (max 20 words) as evidence
   - OR "" ONLY if the value is "Unclear"

5) Evidence MUST be verbatim or lightly trimmed from the transcript.
   - Do NOT paraphrase evidence
   - Do NOT fabricate quotes

6) CRITICAL — HUMAN READABLE TEXT ONLY:
   - Output MUST be human-readable UTF-8 text
   - Do NOT use Unicode escape sequences (e.g. \\uXXXX)
   - Hindi or other non-English text MUST be rendered directly (e.g. "मैं शिकायत करूँगा")
   - Any use of \\uXXXX makes the output INVALID

7) If any rule above is violated, the output is invalid.

========================
SPEAKER INFERENCE
========================
- Infer speakers from context.
- "Customer" = person seeking help or raising an issue.
- "Call_Assistant" = agent, IVR, support rep, or automated system.

========================
THREAT CLASSIFICATION (CRITICAL)
========================
Set "threat_flag" = "Yes" ONLY if the customer explicitly mentions ANY of the following:
- Police complaint / FIR / calling police
- Legal action / lawsuit / court case / lawyer
- Reporting to regulators, government, or media
- Violence, self-harm, or harm to others
- Harassment or intimidation threats

Examples that MUST be "Yes":
- "I will file a police complaint"
- "I am going to court"
- "My lawyer will contact you"
- "मैं पुलिस में शिकायत करूँगा"

Indirect, emotional, or vague language → "Unclear"
No threat language → "No"

========================
PRIORITY RULES
========================
High:
- Safety risk
- Legal or police threats
- Account locked or fraud
- Payment loss
- Customer demands immediate resolution

Medium:
- Issue requires follow-up
- Bugs, confusion, service problems

Low:
- General inquiry
- Information request only

========================
NUISANCE RULES
========================
Set "nuisance" = "Yes" ONLY if transcript contains:
- Profanity
- Harassment
- Abusive language
- Discriminatory or personal attacks

Complaints alone ≠ nuisance.

========================
SATISFACTION RULES
========================
Yes:
- Explicit thanks
- Issue confirmed resolved
- Positive closing statement

No:
- Complaint or frustration at end
- Issue unresolved
- Negative or angry closing

Unclear:
- No clear closing signal

========================
FRUSTRATION RULES
========================
High:
- Anger, threats, repeated complaints

Medium:
- Repeated concern, impatience

Low:
- Calm, neutral, cooperative

========================
PII RULES
========================
Detect ONLY if explicitly spoken in the transcript.
Do NOT assume or infer PII.

========================
SCHEMA (RETURN EXACTLY)
========================
{{
  "summary": "string (detailed line by line summary, factual, no assumptions)",
  "information_requested": "string",
  "threat_flag": "Yes|No|Unclear",
  "threat_reason": "string",
  "priority": "High|Medium|Low",
  "priority_reason": "string",
  "human_intervention_required": "Yes|No|Unclear",
  "human_intervention_reason": "string",
  "satisfied": "Yes|No|Unclear",
  "satisfied_reason": "string",
  "nuisance": "Yes|No|Unclear",
  "nuisance_reason": "string",
  "frustration_level": "Low|Medium|High|Unclear",
  "frustration_reason": "string",
  "repeated_complaint": "Yes|No|Unclear",
  "repeated_complaint_reason": "string",
  "next_best_action": "string (single clear next step)",
  "open_questions": ["string", "string"],
  "pii_detected": "Yes|No|Unclear",
  "pii_types": ["Email", "Phone", "Address", "Card", "Other", "None"]
}}

========================
CALL TRANSCRIPT
========================
{transcript_text}

Return JSON ONLY.
"""



    try:
        client = make_openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        response_text = response.choices[0].message.content.strip()

        # Remove markdown code block formatting if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()
        print(f"✓ Analysis response received")

        analysis = json.loads(response_text)
        return analysis

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def restructure_analysis(analysis):
    """
    Restructure the OpenAI response into the database format.

    Args:
        analysis: Raw analysis dictionary from OpenAI

    Returns:
        Restructured dictionary matching database schema
    """
    structured = {
        "summary": analysis.get("summary", ""),
        "information_requested": analysis.get("information_requested", ""),

        "threat": {
            "flag": analysis.get("threat_flag", "Unclear"),
            "reason": analysis.get("threat_reason", "")
        },

        "priority": {
            "level": analysis.get("priority", "Low"),
            "reason": analysis.get("priority_reason", "")
        },

        "human_intervention": {
            "required": analysis.get("human_intervention_required", "Unclear"),
            "reason": analysis.get("human_intervention_reason", "")
        },

        "satisfaction": {
            "value": analysis.get("satisfied", "Unclear"),
            "reason": analysis.get("satisfied_reason", "")
        },

        "frustration": {
            "level": analysis.get("frustration_level", "Unclear"),
            "reason": analysis.get("frustration_reason", "")
        },

        "nuisance": {
            "value": analysis.get("nuisance", "No"),
            "reason": analysis.get("nuisance_reason", "")
        },

        "repeated_complaint": {
            "value": analysis.get("repeated_complaint", "No"),
            "reason": analysis.get("repeated_complaint_reason", "")
        },

        "pii_details": {
            "detected": analysis.get("pii_detected", "No"),
            "types": analysis.get("pii_types", ["None"])
        },

        "next_best_action": analysis.get("next_best_action", ""),
        "open_questions": analysis.get("open_questions", [])
    }

    return structured

def transcribe_audio(audio_path):
    try:
        client = make_openai_client()
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="gpt-4o-transcribe",
                response_format="text"
            )
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


# ------------------ DATABASE SAVING ------------------

def save_call_status_to_db(conn, call_sid, call_status):
    try:
        cursor = conn.cursor()
        cursor.execute('UPDATE "crm-ai-db" SET call_status=%s WHERE sid=%s', (call_status, call_sid))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"DB save error: {e}")
        return False

def save_structured_analysis_to_db(conn, call_sid, transcript_text, structured_analysis):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE "crm-ai-db" SET transcript=%s, transcript_status=%s,
            summary=%s, summary_completed=%s,
            information_requested=%s, information_requested_completed=%s,
            threat=%s, threat_completed=%s,
            priority=%s, priority_completed=%s,
            human_intervention=%s, human_intervention_completed=%s,
            satisfaction=%s, satisfaction_completed=%s,
            frustration=%s, frustration_completed=%s,
            nuisance=%s, nuisance_completed=%s,
            repeated_complaint=%s, repeated_complaint_completed=%s,
            next_best_action=%s, next_best_action_completed=%s,
            open_questions=%s, open_questions_completed=%s,
            pii_details=%s, pii_details_completed=%s,
            "Completed"=TRUE,
            call_status=%s
            WHERE sid=%s
        ''', (
            transcript_text, "completed",
            structured_analysis.get("summary",""), True,
            structured_analysis.get("information_requested",""), True,
            json.dumps(structured_analysis.get("threat",{})), True,
            json.dumps(structured_analysis.get("priority",{})), True,
            json.dumps(structured_analysis.get("human_intervention",{})), True,
            json.dumps(structured_analysis.get("satisfaction",{})), True,
            json.dumps(structured_analysis.get("frustration",{})), True,
            json.dumps(structured_analysis.get("nuisance",{})), True,
            json.dumps(structured_analysis.get("repeated_complaint",{})), True,
            structured_analysis.get("next_best_action",""), True,
            json.dumps(structured_analysis.get("open_questions",[])), True,
            json.dumps(structured_analysis.get("pii_details",{})), True,
            "completed",
            call_sid
        ))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"DB save error: {e}")
        conn.rollback()
        return False
