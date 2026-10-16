"""
image_validator.py
CivicServe Computer Vision Validation Engine

Performs mandatory server-side validation for complaint image uploads:
1. Image Relevance Verification: Pretrained CLIP model checks whether the image
   matches the selected category (Roads, Street Lights, Waste Management, etc.),
   title, and description against negative/irrelevant classes.
2. AI-Generated Image Detection: Pretrained ViT classifier identifies potential
   AI-generated images without claiming 100% certainty.
3. Security & File Integrity: Detects corrupted files, checks byte integrity,
   enforces size limits and decompression bomb protections.
"""

import os
import io
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union

# Suppress Hugging Face symlinks warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline

from config import Config

logger = logging.getLogger(__name__)

# Enforce decompression bomb protection
Image.MAX_IMAGE_PIXELS = 25_000_000

# Allowed formats
ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

# Category definitions and descriptive prompts for CLIP zero-shot classification
CATEGORY_PROMPTS = {
    "Roads": [
        "a photo of road damage, potholes, cracked asphalt, or damaged road surface",
        "a photograph of a damaged city street, broken pavement, or pothole hazard"
    ],
    "Street Lights": [
        "a photo of a street light, broken streetlamp, dark light pole, or outdoor street lighting fixture",
        "a photo of a streetlight pole, damaged public light post, or exposed electrical wires on street lamp"
    ],
    "Water Supply": [
        "a photo of a broken water supply pipe, bursting pipeline, leaking public water tap, or water gusher",
        "a photo of clean water leaking from municipal water pipe on the street or road"
    ],
    "Drainage": [
        "a photo of an overflowing sewage drain, clogged stormwater drainage ditch, flooded street water, or missing manhole cover",
        "a photo of dirty sewage water puddle, clogged municipal storm drain, or sewer overflow"
    ],
    "Waste Management": [
        "a photo of an overflowing garbage pile, roadside trash dump, scattered plastic waste, or municipal refuse",
        "a photo of discarded rubbish, waste heap on street, or overflowing trash bin"
    ],
    "Parks": [
        "a photo of a municipal public park, broken park bench, damaged playground swing, or fallen tree branch in park",
        "a photo of damaged park recreational facilities, broken walking pathway, or overgrown public garden"
    ]
}

# Negative and irrelevant contrastive classes
NEGATIVE_PROMPTS = {
    "Selfie / Person": "a selfie photo or close-up portrait of a human face or person",
    "Indoor Interior": "a photo of indoor home interior, bedroom, living room sofa, or dining table",
    "Food / Meal": "a photo of a plate of food, meal, snacks, cooking dish, or beverage",
    "Pet Animal": "a photo of a pet animal, domestic dog, or cat",
    "Cartoon / Meme": "a digital cartoon, anime illustration, graphic artwork, or internet meme",
    "Screen Screenshot": "a screenshot of a mobile phone screen, computer monitor, or software application",
    "Document / Text": "a photo of a printed paper document, book page, certificate, or receipt text"
}


class ModelRegistry:
    """
    Thread-safe lazy-loaded singleton container for computer vision models.
    Loads models into CPU/memory once on first use and caches them.
    """
    _clip_model: Optional[CLIPModel] = None
    _clip_processor: Optional[CLIPProcessor] = None
    _ai_detector_pipe = None
    _is_loading: bool = False

    @classmethod
    def get_clip(cls) -> Tuple[CLIPModel, CLIPProcessor]:
        if cls._clip_model is None or cls._clip_processor is None:
            logger.info("Initializing CLIP model (openai/clip-vit-base-patch32)...")
            cls._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            cls._clip_model.eval()
        return cls._clip_model, cls._clip_processor

    @classmethod
    def get_ai_detector(cls):
        if cls._ai_detector_pipe is None:
            logger.info("Initializing AI image detector (umm-maybe/AI-image-detector)...")
            cls._ai_detector_pipe = pipeline(
                "image-classification",
                model="umm-maybe/AI-image-detector"
            )
        return cls._ai_detector_pipe


# ----------------------------------------------------------------------
# 1. Image Content & Security Validation
# ----------------------------------------------------------------------
def validate_image_security(file_or_path: Union[str, bytes, io.BytesIO, Any]) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Validates physical image content integrity:
    - Checks file size limits (<= 5MB)
    - Verifies valid image stream using Pillow
    - Rejects corrupted or truncated files
    - Prevents decompression bomb attacks
    - Returns opened RGB PIL Image object if valid
    """
    try:
        # Determine bytes or file handle
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                return False, None, f"Image file not found on server: {file_or_path}"
            file_size = os.path.getsize(file_or_path)
            if file_size > Config.MAX_IMAGE_SIZE_BYTES:
                return False, None, f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds maximum allowed limit (5MB)."
            with open(file_or_path, "rb") as f:
                img_bytes = f.read()
        elif isinstance(file_or_path, bytes):
            img_bytes = file_or_path
            if len(img_bytes) > Config.MAX_IMAGE_SIZE_BYTES:
                return False, None, f"File size exceeds maximum allowed limit (5MB)."
        elif hasattr(file_or_path, "read"):
            # Werkzeug FileStorage or BytesIO
            file_or_path.seek(0, os.SEEK_END)
            file_size = file_or_path.tell()
            file_or_path.seek(0)
            if file_size > Config.MAX_IMAGE_SIZE_BYTES:
                return False, None, f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds maximum allowed limit (5MB)."
            img_bytes = file_or_path.read()
            file_or_path.seek(0)
        else:
            return False, None, "Invalid image data format provided."

        if not img_bytes:
            return False, None, "Uploaded file is empty (0 bytes)."

        # Step 1: Verify raw integrity
        stream1 = io.BytesIO(img_bytes)
        try:
            with Image.open(stream1) as raw_img:
                raw_format = raw_img.format
                if raw_format not in ALLOWED_IMAGE_FORMATS:
                    return False, None, f"Unsupported image format ({raw_format}). Only JPG, JPEG, PNG, and WEBP are accepted."
                raw_img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            return False, None, f"Corrupted or invalid image file: {str(e)}"

        # Step 2: Open and convert to RGB
        stream2 = io.BytesIO(img_bytes)
        img = Image.open(stream2)
        img.load()  # Read full pixel data to catch truncated streams

        # Check dimensions
        width, height = img.size
        if width < 50 or height < 50:
            return False, None, f"Image resolution ({width}x{height}) is too small. Please upload a clear photo (min 50x50)."

        if img.mode != "RGB":
            img = img.convert("RGB")

        return True, img, None

    except Exception as e:
        logger.error(f"Image security validation exception: {repr(e)}")
        return False, None, f"Image security validation failed: {str(e)}"


# ----------------------------------------------------------------------
# 2. Computer Vision: Image Relevance Verification (CLIP)
# ----------------------------------------------------------------------
def check_image_relevance(
    image: Image.Image,
    target_category: str,
    title: str = "",
    description: str = ""
) -> Dict[str, Any]:
    """
    Evaluates whether the image matches the selected complaint category,
    title, and description using zero-shot CLIP classification and text-image alignment.
    """
    clip_model, clip_processor = ModelRegistry.get_clip()

    # Normalize category name
    clean_category = target_category.strip()
    # Fallback to Roads if category not in dict
    matched_cat_key = None
    for k in CATEGORY_PROMPTS:
        if k.lower() == clean_category.lower():
            matched_cat_key = k
            break
    if not matched_cat_key:
        matched_cat_key = "Roads"

    category_keys = list(CATEGORY_PROMPTS.keys())
    # Representative prompt per civic category
    candidate_prompts = [CATEGORY_PROMPTS[c][0] for c in category_keys]

    # Negative / Irrelevant contrastive prompts
    neg_keys = list(NEGATIVE_PROMPTS.keys())
    candidate_prompts += [NEGATIVE_PROMPTS[k] for k in neg_keys]

    # Zero-shot softmax classification
    inputs = clip_processor(text=candidate_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    cat_probs = {c: probs[i].item() for i, c in enumerate(category_keys)}
    neg_probs = {k: probs[len(category_keys) + i].item() for i, k in enumerate(neg_keys)}

    # Direct title & description cosine similarity
    clean_desc = (description[:120] if description else title)[:120]
    context_text = f"a photograph of public municipal issue: {title}. {clean_desc}"
    text_inputs = clip_processor(text=[context_text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        direct_outputs = clip_model(**text_inputs)
        if hasattr(direct_outputs, "image_embeds") and direct_outputs.image_embeds is not None:
            img_emb = direct_outputs.image_embeds
            txt_emb = direct_outputs.text_embeds
        else:
            img_feat = clip_model.get_image_features(text_inputs["pixel_values"])
            txt_feat = clip_model.get_text_features(text_inputs["input_ids"], text_inputs["attention_mask"])
            img_emb = img_feat.pooler_output if hasattr(img_feat, "pooler_output") else img_feat
            txt_emb = txt_feat.pooler_output if hasattr(txt_feat, "pooler_output") else txt_feat

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cosine_sim = (img_emb @ txt_emb.T)[0][0].item()

    target_prob = cat_probs.get(matched_cat_key, 0.0)
    top_cat = max(cat_probs, key=cat_probs.get)
    top_cat_prob = cat_probs[top_cat]
    top_neg = max(neg_probs, key=neg_probs.get)
    top_neg_prob = neg_probs[top_neg]

    civic_prob_sum = sum(cat_probs.values())
    neg_prob_sum = sum(neg_probs.values())

    # Composite Relevance Score calculation (0 - 100)
    target_civic_ratio = (target_prob / civic_prob_sum) if civic_prob_sum > 0 else 0
    civic_dominance = max(0.0, (civic_prob_sum - neg_prob_sum + 1.0) / 2.0)
    text_align = max(0.0, min(1.0, (cosine_sim - 0.10) / 0.25))

    raw_score = (target_civic_ratio * 40.0) + (civic_dominance * 35.0) + (text_align * 25.0)
    score = round(max(0.0, min(100.0, raw_score)), 2)

    # Decision Logic
    # 1. Clearly Unrelated cases:
    if top_neg_prob > 0.40 and top_neg_prob > target_prob * 1.5:
        status = "Unrelated"
        message = f"The uploaded photo appears to depict a {top_neg.lower()} rather than a public infrastructure issue."
    elif matched_cat_key != top_cat and top_cat_prob > 0.40 and top_cat_prob > target_prob * 2.0:
        status = "Unrelated"
        message = f"The uploaded photo appears to show {top_cat} instead of the selected category ({matched_cat_key})."
    # 2. Relevant cases:
    elif score >= Config.RELEVANCE_ACCEPT_THRESHOLD and target_prob >= 0.12 and top_neg_prob < 0.45:
        status = "Relevant"
        message = f"Photo successfully verified as relevant to {matched_cat_key}."
    # 3. Uncertain cases:
    elif score >= Config.RELEVANCE_UNCERTAIN_THRESHOLD and target_prob >= 0.06:
        status = "Uncertain"
        message = f"Image relevance could not be definitively confirmed (Score: {score}%). Please ensure the photo clearly depicts {matched_cat_key}."
    else:
        status = "Unrelated"
        message = f"The uploaded photo does not clearly show {matched_cat_key} (Score: {score}%). Please upload a clear photo of the issue."

    return {
        "status": status,
        "score": score,
        "target_category": matched_cat_key,
        "target_prob": round(target_prob * 100, 2),
        "top_category": top_cat,
        "top_cat_prob": round(top_cat_prob * 100, 2),
        "top_negative": top_neg if top_neg_prob > 0.15 else None,
        "top_neg_prob": round(top_neg_prob * 100, 2),
        "cosine_similarity": round(cosine_sim, 4),
        "message": message
    }


# ----------------------------------------------------------------------
# 3. Computer Vision: AI-Generated Image Detection
# ----------------------------------------------------------------------
def check_ai_generation(image: Image.Image) -> Dict[str, Any]:
    """
    Runs pretrained ViT AI-image detector to identify potential AI generation.
    Labels: 'artificial' (AI generated) vs 'human' (camera photograph).
    Includes appropriate scientific disclaimers regarding statistical nature of detection.
    """
    detector_pipe = ModelRegistry.get_ai_detector()
    raw_results = detector_pipe(image)

    # Parse scores
    artificial_score = 0.0
    human_score = 0.0
    for item in raw_results:
        label = item.get("label", "").lower()
        score_val = float(item.get("score", 0.0))
        if label == "artificial":
            artificial_score = score_val
        elif label == "human":
            human_score = score_val

    ai_prob_pct = round(artificial_score * 100, 2)
    human_prob_pct = round(human_score * 100, 2)

    # Determine status
    if artificial_score >= Config.AI_DETECTION_THRESHOLD:
        status = "Suspected AI"
        is_suspected = True
        message = f"Automated detector flagged image as potentially AI-generated (AI Probability: {ai_prob_pct}%)."
    elif artificial_score >= 0.40:
        status = "Uncertain"
        is_suspected = False
        message = f"AI detection analysis is inconclusive (AI Probability: {ai_prob_pct}%)."
    else:
        status = "Authentic"
        is_suspected = False
        message = f"Image appears to be authentic camera-captured photography (Human Probability: {human_prob_pct}%)."

    return {
        "status": status,
        "is_ai_suspected": is_suspected,
        "ai_score": ai_prob_pct,
        "human_score": human_prob_pct,
        "primary_label": "artificial" if artificial_score > human_score else "human",
        "confidence": ai_prob_pct if artificial_score > human_score else human_prob_pct,
        "disclaimer": "Automated computer vision detection. Predictive estimate, not legal proof.",
        "message": message
    }


# ----------------------------------------------------------------------
# 4. Master Image Validation Engine
# ----------------------------------------------------------------------
def validate_complaint_image(
    file_or_path: Union[str, bytes, io.BytesIO, Any],
    category: str,
    title: str = "",
    description: str = "",
    allow_uncertain: bool = True
) -> Dict[str, Any]:
    """
    Full mandatory validation pipeline combining security, relevance, and AI detection.
    Enforces decision matrix:
    - Relevant: allow submission if other checks pass.
    - Clearly Unrelated: block submission, prompt citizen to upload relevant photo.
    - Suspected AI: flag for admin review OR reject if policy is 'reject'.
    - Uncertain: show warning, allow retry or review option.
    - Model/API failure: mandatory validation NEVER silently bypassed.
    """
    start_time = time.time()

    # Step 1: Security & File Integrity Check
    sec_ok, pil_image, sec_error = validate_image_security(file_or_path)
    if not sec_ok or pil_image is None:
        return {
            "success": False,
            "can_submit": False,
            "decision": "error",
            "error": sec_error or "Image file could not be verified.",
            "relevance": None,
            "ai_detection": None,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }

    # Step 2: Computer Vision Models Execution (Relevance & AI Detector)
    try:
        relevance_res = check_image_relevance(
            image=pil_image,
            target_category=category,
            title=title,
            description=description
        )
    except Exception as e:
        logger.error(f"Image relevance model failure: {repr(e)}")
        # Do not silently bypass mandatory validation
        return {
            "success": False,
            "can_submit": False,
            "decision": "error",
            "error": "Mandatory image relevance verification service encountered an internal error. Please try again.",
            "relevance": None,
            "ai_detection": None,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }

    try:
        ai_res = check_ai_generation(image=pil_image)
    except Exception as e:
        logger.error(f"AI image detector failure: {repr(e)}")
        # Do not silently bypass mandatory validation
        return {
            "success": False,
            "can_submit": False,
            "decision": "error",
            "error": "Mandatory AI image detection service encountered an internal error. Please try again.",
            "relevance": relevance_res,
            "ai_detection": None,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1)
        }

    # Step 3: Decision Synthesis
    relevance_status = relevance_res["status"]
    is_ai_suspected = ai_res["is_ai_suspected"]
    policy = Config.AI_IMAGE_POLICY  # 'flag' or 'reject'

    can_submit = True
    decision = "allow"
    warning = None

    # Check 1: Clearly Unrelated Image -> BLOCK SUBMISSION
    if relevance_status == "Unrelated":
        can_submit = False
        decision = "block_unrelated"

    # Check 2: Suspected AI Image
    elif is_ai_suspected:
        if policy == "reject":
            can_submit = False
            decision = "block_ai"
        else:
            can_submit = True
            decision = "flag_ai"
            warning = (
                f"Notice: Automated analysis indicates this photo may be AI-generated "
                f"({ai_res['ai_score']}% probability). It will be flagged for administrator review."
            )

    # Check 3: Uncertain Relevance
    elif relevance_status == "Uncertain":
        if allow_uncertain:
            can_submit = True
            decision = "uncertain"
            warning = (
                f"Warning: The relevance of this photo could not be definitively confirmed "
                f"(Score: {relevance_res['score']}%). You may submit for manual verification, or choose a clearer photo."
            )
        else:
            can_submit = False
            decision = "block_unrelated"

    # Check 4: Relevant and Authentic
    else:
        can_submit = True
        decision = "allow"

    return {
        "success": True,
        "can_submit": can_submit,
        "decision": decision,
        "error": None,
        "warning": warning,
        "relevance": relevance_res,
        "ai_detection": ai_res,
        "processing_time_ms": round((time.time() - start_time) * 1000, 1)
    }


def calculate_image_similarity(image1_path: str, image2_path: str) -> float:
    """
    Computes visual similarity between two complaint images using:
    1. Exact file hash/byte matching (100.0% if identical bytes)
    2. CLIP Vision Transformer embeddings and cosine similarity

    Returns a score between 0.0 and 100.0 (percentage).
    Returns 0.0 if either image path is missing, corrupted, or unreadable.
    """
    if not image1_path or not image2_path:
        return 0.0

    def _resolve(p: str) -> Optional[str]:
        if not p:
            return None
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if os.path.exists(p):
            return os.path.abspath(p)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        c1 = os.path.join(base_dir, p)
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(base_dir, "uploads", os.path.basename(p))
        if os.path.exists(c2):
            return c2
        c3 = os.path.join(base_dir, "static", p)
        if os.path.exists(c3):
            return c3
        return None

    path1 = _resolve(image1_path)
    path2 = _resolve(image2_path)

    if not path1 or not path2:
        return 0.0

    # 1. Exact byte / hash match
    try:
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            if f1.read() == f2.read():
                return 100.0
    except Exception as e:
        logger.warning(f"Error reading image files for byte comparison: {e}")

    # 2. CLIP visual embedding cosine similarity
    try:
        clip_model, clip_processor = ModelRegistry.get_clip()
        im1 = Image.open(path1).convert("RGB")
        im2 = Image.open(path2).convert("RGB")

        inputs = clip_processor(images=[im1, im2], return_tensors="pt")
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                embeds = feats.pooler_output
            elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                embeds = feats.image_embeds
            elif isinstance(feats, torch.Tensor):
                embeds = feats
            else:
                embeds = feats[0]

            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            cos_sim = (embeds[0] @ embeds[1]).item()
            score = max(0.0, min(100.0, cos_sim * 100.0))
            return round(score, 2)
    except Exception as e:
        logger.warning(f"Error computing CLIP image similarity between {path1} and {path2}: {e}")
        return 0.0
