import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

NEGATIVE_PROMPTS = {
    "Selfie / Person": "a selfie photo or close-up portrait of a human face or person",
    "Indoor Interior": "a photo of indoor home interior, bedroom, living room sofa, or dining table",
    "Food / Meal": "a photo of a plate of food, meal, snacks, cooking dish, or beverage",
    "Pet Animal": "a photo of a pet animal, domestic dog, or cat",
    "Cartoon / Meme": "a digital cartoon, anime illustration, graphic artwork, or internet meme",
    "Screen Screenshot": "a screenshot of a mobile phone screen, computer monitor, or software application",
    "Document / Text": "a photo of a printed paper document, book page, certificate, or receipt text"
}

def evaluate_image_relevance(image, target_category, title="", description=""):
    # Build prompt list
    # Use best representative prompt per category
    category_keys = list(CATEGORY_PROMPTS.keys())
    candidate_prompts = [CATEGORY_PROMPTS[c][0] for c in category_keys]
    
    neg_keys = list(NEGATIVE_PROMPTS.keys())
    candidate_prompts += [NEGATIVE_PROMPTS[k] for k in neg_keys]
    
    # Run zero-shot classification
    inputs = clip_processor(text=candidate_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    cat_probs = {c: probs[i].item() for i, c in enumerate(category_keys)}
    neg_probs = {k: probs[len(category_keys) + i].item() for i, k in enumerate(neg_keys)}
    
    # Direct title/description cosine similarity
    clean_desc = (description[:120] if description else title)[:120]
    context_text = f"a photograph of public municipal issue: {title}. {clean_desc}"
    text_inputs = clip_processor(text=[context_text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        direct_outputs = clip_model(**text_inputs)
        # In transformers, direct_outputs has image_embeds and text_embeds
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
    
    target_prob = cat_probs.get(target_category, 0.0)
    top_cat = max(cat_probs, key=cat_probs.get)
    top_cat_prob = cat_probs[top_cat]
    top_neg = max(neg_probs, key=neg_probs.get)
    top_neg_prob = neg_probs[top_neg]
    
    # Civic probability total
    civic_prob_sum = sum(cat_probs.values())
    neg_prob_sum = sum(neg_probs.values())
    
    # Calculate composite score (0-100)
    # 1. Target share of civic probability
    target_civic_ratio = (target_prob / civic_prob_sum) if civic_prob_sum > 0 else 0
    # 2. Civic vs Negative balance
    civic_dominance = max(0.0, (civic_prob_sum - neg_prob_sum + 1.0) / 2.0)
    # 3. Direct text alignment (-1 to 1 scaled to 0 to 1)
    text_align = max(0.0, min(1.0, (cosine_sim - 0.10) / 0.25))
    
    score = (target_civic_ratio * 40.0) + (civic_dominance * 35.0) + (text_align * 25.0)
    score = round(max(0.0, min(100.0, score)), 2)
    
    # Decision logic
    if top_neg_prob > 0.45 and top_neg_prob > target_prob * 1.8:
        status = "Unrelated"
        msg = f"The image appears to depict a {top_neg.lower()} rather than a public infrastructure issue."
    elif target_category != top_cat and top_cat_prob > 0.40 and top_cat_prob > target_prob * 2.0:
        status = "Unrelated"
        msg = f"The uploaded photo appears to show {top_cat} instead of the selected category ({target_category})."
    elif score >= 48.0 and target_prob >= 0.15 and top_neg_prob < 0.40:
        status = "Relevant"
        msg = f"Image successfully verified as relevant to {target_category}."
    elif score >= 28.0 and target_prob >= 0.08:
        status = "Uncertain"
        msg = f"Image relevance could not be definitively confirmed (Score: {score}%). Please ensure the photo clearly depicts {target_category}."
    else:
        status = "Unrelated"
        msg = f"The image does not clearly depict {target_category} (Score: {score}%). Please upload a clear photo of the issue."
    
    return {
        "status": status,
        "score": score,
        "target_category": target_category,
        "target_prob": round(target_prob * 100, 2),
        "top_cat": top_cat,
        "top_cat_prob": round(top_cat_prob * 100, 2),
        "top_neg": top_neg,
        "top_neg_prob": round(top_neg_prob * 100, 2),
        "cosine_sim": round(cosine_sim, 4),
        "message": msg
    }

# Test with our created synthetic images
def make_img(color, text=None):
    img = Image.new("RGB", (256, 256), color=color)
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((30, 100), text, fill=(0, 0, 0))
    return img

print("\n--- Test 1: Gray Road-like Image vs Roads ---")
road_img = make_img((70, 70, 70))
d = ImageDraw.Draw(road_img)
d.ellipse([50, 70, 200, 180], fill=(20, 20, 20))
res1 = evaluate_image_relevance(road_img, "Roads", "Deep pothole in street", "Pothole filled with gravel on main road")
print(res1)

print("\n--- Test 2: Cartoon / Artwork Image vs Roads ---")
art_img = make_img((255, 120, 120), "MEME / COMIC")
res2 = evaluate_image_relevance(art_img, "Roads", "Deep pothole in street", "Pothole filled with gravel on main road")
print(res2)

print("\n--- Test 3: Existing uploads image (cities.jpg) vs Roads ---")
cities_path = r"c:\Users\aiswa\PycharmProjects\smart_public_infra\static\images\cities.jpg"
if os.path.exists(cities_path):
    c_img = Image.open(cities_path)
    res3 = evaluate_image_relevance(c_img, "Roads", "Road damaged", "City road asphalt broken")
    print(res3)
