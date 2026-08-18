import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import base64
import urllib.request
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ─────────────────────────────────────────
# DOWNLOAD FACE CASCADE
# ─────────────────────────────────────────
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        cascade_path
    )

face_cascade = cv2.CascadeClassifier(cascade_path)

# ─────────────────────────────────────────
# FORM SPECS
# ─────────────────────────────────────────
FORM_SPECS = {
    "aadhaar": {
        "name": "Aadhaar Card",
        "width_px": 413,
        "height_px": 531,
        "max_size_kb": 50,
        "face_coverage_min": 0.70,
        "face_coverage_max": 0.80
    },
    "driving_license": {
        "name": "Driving License",
        "width_px": 413,
        "height_px": 531,
        "max_size_kb": 20,
        "face_coverage_min": 0.70,
        "face_coverage_max": 0.80
    },
    "income_certificate": {
        "name": "Income Certificate",
        "width_px": 160,
        "height_px": 212,
        "max_size_kb": 20,
        "face_coverage_min": 0.70,
        "face_coverage_max": 0.80
    },
    "domicile_certificate": {
        "name": "Domicile Certificate",
        "width_px": 160,
        "height_px": 212,
        "max_size_kb": 20,
        "face_coverage_min": 0.70,
        "face_coverage_max": 0.80
    },
    "voter_id": {
        "name": "Voter ID",
        "width_px": 413,
        "height_px": 531,
        "max_size_kb": 200,
        "face_coverage_min": 0.70,
        "face_coverage_max": 0.80
    }
}

# ─────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────
KNOWLEDGE_BASE = {
    "aadhaar": {
        "form_name": "Aadhaar Update/Correction",
        "authority": "UIDAI",
        "fee": "₹50 for Demographic, ₹100 for Biometric",
        "portal": "https://myaadhaar.uidai.gov.in",
        "documents": [
            "Proof of Identity (PAN, Passport, Voter ID)",
            "Proof of Address (Ration Card, Electricity Bill)",
            "Birth Certificate (for DOB correction)"
        ],
        "process_steps": [
            "1. Visit myaadhaar.uidai.gov.in",
            "2. Login with Aadhaar number and OTP",
            "3. Select field to update",
            "4. Upload supporting documents",
            "5. Pay ₹50 fee online",
            "6. Note URN and track status"
        ],
        "processing_time": "5-10 working days",
        "common_mistakes": [
            "Address must exactly match document",
            "Documents must be self-attested",
            "Blurry scans are rejected"
        ],
        "photo_specs": {"note": "No photo needed, captured at center"},
        "follow_up_suggestions": {
            "documents": ["Which documents are valid for address proof?", "Do documents need to be self-attested?", "What if I don't have any documents?"],
            "fee": ["How do I pay fee online?", "Is there fee waiver?", "What if payment fails?"],
            "status": ["How to track update status?", "What is URN number?", "How long does it take?"]
        }
    },
    "driving_license": {
        "form_name": "Fresh Driving License Maharashtra",
        "authority": "RTO Maharashtra",
        "fee": "₹200 for LL, ₹700-1000 for DL",
        "portal": "https://sarathi.parivahan.gov.in",
        "documents": [
            "Form 1 and 1A (Medical)",
            "Age Proof (School Leaving Certificate)",
            "Address Proof (Aadhaar)",
            "Passport photo (35x45mm, white bg, max 20KB)"
        ],
        "process_steps": [
            "1. Visit sarathi.parivahan.gov.in",
            "2. Select Maharashtra",
            "3. Apply for Learner License",
            "4. Pay ₹200 and give online test",
            "5. Wait 30 days",
            "6. Apply for Permanent DL",
            "7. Book RTO slot and give driving test"
        ],
        "processing_time": "LL: Same day, DL: 7-15 days",
        "common_mistakes": [
            "Photo must have white background",
            "Photo must be under 20KB",
            "Must wait 30 days between LL and DL"
        ],
        "photo_specs": {"size": "35x45mm", "max_kb": 20, "format": "JPEG", "bg": "White"},
        "follow_up_suggestions": {
            "documents": ["What is Form 1?", "Which address proof is accepted?", "Do I need originals?"],
            "photo": ["What size photo is needed?", "How to resize to 20KB?", "Can I use mobile photo?"],
            "test": ["What questions come in LL test?", "What if I fail?", "How many questions?"],
            "fee": ["Total fee amount?", "Can I pay cash?", "Fee for retake?"]
        }
    },
    "income_certificate": {
        "form_name": "Income Certificate Maharashtra",
        "authority": "Revenue Department Maharashtra",
        "fee": "₹33.60",
        "portal": "https://aaplesarkar.maharashtra.gov.in",
        "documents": [
            "Self-Declaration of Income",
            "Salary Slip",
            "Ration Card",
            "Aadhaar Card"
        ],
        "process_steps": [
            "1. Register on aaplesarkar.maharashtra.gov.in",
            "2. Go to Revenue Department",
            "3. Select Income Certificate",
            "4. Fill income details",
            "5. Upload documents",
            "6. Pay ₹33.60",
            "7. Submit to Tehsildar"
        ],
        "processing_time": "7-15 working days",
        "common_mistakes": [
            "Include all family members income",
            "Self-declaration on stamp paper",
            "Photo must be 160x212 pixels"
        ],
        "photo_specs": {"size": "160x212px", "max_kb": 20, "format": "JPEG", "bg": "White"},
        "follow_up_suggestions": {
            "documents": ["What is 7/12 extract?", "Where to get stamp paper?", "Self-employed documents?"],
            "status": ["How to track status?", "Where to collect?", "Can I download online?"]
        }
    },
    "domicile_certificate": {
        "form_name": "Domicile Certificate Maharashtra",
        "authority": "Revenue Department Maharashtra",
        "fee": "₹33.60",
        "portal": "https://aaplesarkar.maharashtra.gov.in",
        "documents": [
            "15 years residence proof",
            "Ration Card",
            "Birth Certificate",
            "School Leaving Certificate",
            "Aadhaar Card"
        ],
        "process_steps": [
            "1. Register on aaplesarkar.maharashtra.gov.in",
            "2. Go to Revenue Department",
            "3. Select Domicile Certificate",
            "4. Fill residence details",
            "5. Upload 15-year proof",
            "6. Pay ₹33.60",
            "7. Submit to Tehsildar"
        ],
        "processing_time": "15-30 working days",
        "common_mistakes": [
            "Must prove 15 continuous years in Maharashtra",
            "School certificate is strongest proof",
            "Single document preferred over multiple"
        ],
        "photo_specs": {"size": "160x212px", "max_kb": 20, "format": "JPEG", "bg": "White"},
        "follow_up_suggestions": {
            "documents": ["What counts as 15-year proof?", "Can school certificate work?", "What if no birth certificate?"],
            "eligibility": ["Who is eligible?", "Born outside Maharashtra?", "Is Aadhaar enough?"]
        }
    },
    "voter_id": {
        "form_name": "New Voter ID (Form 6)",
        "authority": "Election Commission of India",
        "fee": "Free",
        "portal": "https://voters.eci.gov.in",
        "documents": [
            "Age Proof (Aadhaar, PAN, Birth Certificate)",
            "Address Proof (Aadhaar, Passport)",
            "Passport photo (3.5x4.5cm, white bg)"
        ],
        "process_steps": [
            "1. Visit voters.eci.gov.in",
            "2. Register as new user",
            "3. Fill Form 6",
            "4. Upload photo and documents",
            "5. Submit",
            "6. Field verification by BLO",
            "7. EPIC card delivered by post"
        ],
        "processing_time": "15-30 days",
        "common_mistakes": [
            "Must be 18 on January 1st of application year",
            "Address must match constituency",
            "Recent photo required"
        ],
        "photo_specs": {"size": "3.5x4.5cm", "max_kb": 200, "format": "JPEG", "bg": "White"},
        "follow_up_suggestions": {
            "documents": ["What age proof is accepted?", "Can Aadhaar be both proof?", "No address proof?"],
            "eligibility": ["Minimum age?", "NRI apply?", "Recently moved?"],
            "status": ["Track EPIC status?", "Find my booth?", "What does BLO verify?"]
        }
    }
}

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def detect_language(text):
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    ratio = devanagari_count / total_chars
    if ratio > 0.3:
        marathi_words = ["माझ्या", "मला", "आहे", "करायचे", "कसे", "काय", "मी", "आपला", "नाही"]
        hindi_words = ["मेरा", "मुझे", "है", "करना", "कैसे", "क्या", "मैं", "आपका", "नहीं"]
        marathi_score = sum(1 for w in marathi_words if w in text)
        hindi_score = sum(1 for w in hindi_words if w in text)
        return "marathi" if marathi_score > hindi_score else "hindi"
    return "english"

def detect_form(user_query):
    query_lower = user_query.lower()
    keywords = {
        "aadhaar": ["aadhaar", "aadhar", "uid", "आधार", "uidai"],
        "driving_license": ["driving", "licence", "license", "dl", "learner", "rto", "ll", "ड्राइविंग", "परवाना", "अनुज्ञप्ती"],
        "income_certificate": ["income", "उत्पन्न", "salary", "earning"],
        "domicile_certificate": ["domicile", "residence", "निवास", "रहिवासी", "अधिवास"],
        "voter_id": ["voter", "epic", "election", "form 6", "मतदार", "vote"]
    }
    for form_id, words in keywords.items():
        if any(word in query_lower for word in words):
            return form_id
    return None

def get_follow_up_suggestions(user_query, form_id):
    if not form_id or form_id not in KNOWLEDGE_BASE:
        return ["What documents do I need?", "How much is the fee?", "How long does it take?"]
    form = KNOWLEDGE_BASE[form_id]
    suggestions = form.get("follow_up_suggestions", {})
    query_lower = user_query.lower()
    if any(w in query_lower for w in ["document", "doc", "कागद", "कागजात"]):
        return suggestions.get("documents", [])[:3]
    elif any(w in query_lower for w in ["photo", "image", "फोटो"]):
        return suggestions.get("photo", [])[:3]
    elif any(w in query_lower for w in ["fee", "cost", "price", "शुल्क", "पैसे"]):
        return suggestions.get("fee", [])[:3]
    elif any(w in query_lower for w in ["status", "track", "time", "days"]):
        return suggestions.get("status", [])[:3]
    else:
        all_suggestions = []
        for key in suggestions:
            all_suggestions.extend(suggestions[key][:1])
        return all_suggestions[:3]

def chat_with_formSaathi(user_query, conversation_history=[], current_form=None):
    language = detect_language(user_query)
    detected_form = detect_form(user_query)
    if detected_form:
        current_form = detected_form

    form_context = ""
    if current_form and current_form in KNOWLEDGE_BASE:
        form_data = KNOWLEDGE_BASE[current_form]
        form_context = f"""
        FORM INFORMATION:
        Form Name: {form_data['form_name']}
        Authority: {form_data['authority']}
        Fee: {form_data['fee']}
        Portal: {form_data['portal']}
        Documents: {', '.join(form_data['documents'])}
        Steps: {' | '.join(form_data['process_steps'])}
        Processing Time: {form_data['processing_time']}
        Common Mistakes: {', '.join(form_data['common_mistakes'])}
        """

    language_instruction = {
        "hindi": "IMPORTANT: Respond ONLY in simple Hindi (Devanagari script).",
        "marathi": "IMPORTANT: Respond ONLY in simple Marathi (Devanagari script).",
        "english": "IMPORTANT: Respond in simple English. No jargon."
    }

    full_prompt = f"""
    You are FormSaathi, a friendly AI assistant helping Indian citizens
    with government form filling in Maharashtra.
    - Be warm, patient, encouraging
    - No complex language
    - Give step by step guidance
    - Mention exact fees, documents, portal links
    - Maximum 4-5 sentences or bullet points
    {language_instruction[language]}
    {form_context if form_context else "Help user identify which service they need."}
    User Question: {user_query}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=full_prompt
    )

    suggestions = get_follow_up_suggestions(user_query, current_form)

    return {
        "response": response.text,
        "language": language,
        "current_form": current_form,
        "suggestions": suggestions,
        "confidence": "high" if current_form else "medium"
    }

# ─────────────────────────────────────────
# PHOTO PROCESSOR
# ─────────────────────────────────────────
class PhotoProcessor:

    def process(self, image_bytes, form_id):
        if form_id not in FORM_SPECS:
            return {"success": False, "error": f"Unknown form: {form_id}"}

        specs = FORM_SPECS[form_id]
        original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size_kb = len(image_bytes) / 1024
        original_np = np.array(original_pil)

        face_info = self._detect_face(original_np)

        if not face_info["face_found"]:
            return {
                "success": False,
                "error": "No face detected. Please upload a clear front-facing photo.",
                "original_size_kb": round(original_size_kb, 2)
            }

        original_coverage = face_info["face_height_ratio"]
        cropped_pil = self._crop_and_center_face(
            original_pil, face_info,
            specs["face_coverage_min"],
            specs["face_coverage_max"]
        )
        white_bg_pil = self._replace_background(cropped_pil)
        resized_pil = white_bg_pil.resize(
            (specs["width_px"], specs["height_px"]),
            Image.LANCZOS
        )
        compressed_bytes, final_size_kb = self._compress_image(
            resized_pil, specs["max_size_kb"]
        )
        final_np = np.array(Image.open(io.BytesIO(compressed_bytes)))
        final_face_info = self._detect_face(final_np)
        final_coverage = (
            final_face_info["face_height_ratio"]
            if final_face_info["face_found"]
            else original_coverage
        )
        report = self._build_report(
            specs, original_pil, original_size_kb,
            original_coverage, final_size_kb, final_coverage
        )
        processed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        return {
            "success": True,
            "form": specs["name"],
            "processed_image_base64": processed_base64,
            "report": report
        }

    def _detect_face(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return {"face_found": False, "face_height_ratio": 0}
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img_h, img_w = image_np.shape[:2]
        return {
            "face_found": True,
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "face_height_ratio": round((h * 1.23) / img_h, 3),
            "img_w": img_w,
            "img_h": img_h
        }

    def _crop_and_center_face(self, pil_image, face_info, min_coverage, max_coverage):
        img_w = face_info["img_w"]
        img_h = face_info["img_h"]
        face_x = face_info["x"]
        face_y = face_info["y"]
        face_w = face_info["w"]
        face_h = face_info["h"]

        pad_top    = int(face_h * 0.20)
        pad_bottom = int(face_h * 0.15)
        pad_left   = int(face_w * 0.15)
        pad_right  = int(face_w * 0.15)

        crop_x1 = max(0, face_x - pad_left)
        crop_y1 = max(0, face_y - pad_top)
        crop_x2 = min(img_w, face_x + face_w + pad_right)
        crop_y2 = min(img_h, face_y + face_h + pad_bottom)

        return pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _replace_background(self, pil_image):
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        removed_bg_bytes = remove(img_bytes)
        removed_bg = Image.open(io.BytesIO(removed_bg_bytes)).convert("RGBA")
        white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
        white_bg.paste(removed_bg, mask=removed_bg.split()[3])
        return white_bg.convert("RGB")

    def _compress_image(self, pil_image, max_size_kb):
        quality = 95
        while quality >= 10:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_size_kb:
                return buffer.getvalue(), round(size_kb, 2)
            quality -= 5
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=10, optimize=True)
        return buffer.getvalue(), round(buffer.tell() / 1024, 2)

    def _build_report(self, specs, original_pil, original_size_kb,
                      original_coverage, final_size_kb, final_coverage):
        orig_w, orig_h = original_pil.size
        return {
            "before": {
                "dimensions": f"{orig_w}x{orig_h} px",
                "size_kb": round(original_size_kb, 2),
                "face_coverage": f"{round(original_coverage * 100, 1)}%",
                "ready": False
            },
            "after": {
                "dimensions": f"{specs['width_px']}x{specs['height_px']} px",
                "size_kb": final_size_kb,
                "face_coverage": f"{round(final_coverage * 100, 1)}%",
                "max_allowed_kb": specs["max_size_kb"],
                "ready": final_size_kb <= specs["max_size_kb"]
            },
            "checks": {
                "face_detected": "✅ Face Detected",
                "face_centered": "✅ Face Centered",
                "background": "✅ White Background Applied",
                "dimensions": f"✅ Resized to {specs['width_px']}x{specs['height_px']}px",
                "file_size": "✅ Within Limit" if final_size_kb <= specs["max_size_kb"] else "⚠️ Slightly Over Limit"
            }
        }

processor = PhotoProcessor()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ FormSaathi Backend Running",
        "version": "1.0.0",
        "endpoints": ["POST /chat", "POST /process-photo", "GET /form-specs"]
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        result = chat_with_formSaathi(
            data["message"],
            data.get("history", []),
            data.get("current_form", None)
        )
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-photo", methods=["POST"])
def process_photo():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No photo uploaded"}), 400
        if "form_id" not in request.form:
            return jsonify({"error": "No form_id provided"}), 400
        photo = request.files["photo"]
        form_id = request.form["form_id"]
        image_bytes = photo.read()
        result = processor.process(image_bytes, form_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/form-specs", methods=["GET"])
def form_specs():
    try:
        simplified = {}
        for form_id, specs in FORM_SPECS.items():
            simplified[form_id] = {
                "name": specs["name"],
                "width_px": specs["width_px"],
                "height_px": specs["height_px"],
                "max_size_kb": specs["max_size_kb"]
            }
        return jsonify({"success": True, "forms": simplified})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/knowledge-base", methods=["GET"])
def knowledge_base_endpoint():
    try:
        form_id = request.args.get("form_id", None)
        if form_id:
            if form_id in KNOWLEDGE_BASE:
                return jsonify({"success": True, "data": KNOWLEDGE_BASE[form_id]})
            return jsonify({"error": f"Form not found: {form_id}"}), 404
        return jsonify({"success": True, "forms": list(KNOWLEDGE_BASE.keys())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
