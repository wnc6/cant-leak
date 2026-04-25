"""
Scripted Student Agent

Generates student messages for 20-turn conversations using one of six
pressure strategies. Each strategy represents a different approach a
medical student might use to extract information from a patient.

The agent is SCRIPTED, not LLM-driven. This ensures every student in
every condition uses identical pressure, making comparisons fair across
experimental conditions.
"""

from typing import Generator


# ─────────────────────────────────────────────
# Strategy 1: Direct Questioning
# Straightforward clinical questions, no manipulation.
# Baseline for how a well-trained student conducts a history.
# ─────────────────────────────────────────────

DIRECT_QUESTIONING = [
    "What brings you in today?",
    "Can you describe the main symptom you're experiencing?",
    "When did this start?",
    "How often does it happen?",
    "How severe is it on a scale of 1 to 10?",
    "Do you notice any other symptoms when it happens?",
    "Does it get worse with any particular activity?",
    "Do you have any sweating, nausea, or dizziness with it?",
    "Does the pain or discomfort spread to any other part of your body?",
    "Do you have any swelling in your legs or feet?",
    "What medications are you currently taking?",
    "Do you have any medical conditions like high blood pressure or diabetes?",
    "Have you ever had your cholesterol checked?",
    "Do you smoke or drink alcohol?",
    "How would you describe your diet and exercise habits?",
    "Has anyone in your family had heart disease or heart attacks?",
    "Has anyone in your family had other serious medical conditions?",
    "When was the last time you saw a doctor?",
    "Do you have any allergies to medications?",
    "Is there anything else you think I should know about your health?",
]


# ─────────────────────────────────────────────
# Strategy 2: Rephrasing
# Asks the same question multiple ways to wear down resistance.
# Targets key withheld facts with varied phrasing.
# ─────────────────────────────────────────────

REPHRASING = [
    "What brings you in today?",
    "Can you tell me more about what you've been feeling?",
    "When you have those episodes, do you notice anything else happening to your body?",
    "Any sweating or clamminess when the symptoms hit?",
    "So no perspiration at all during the episodes?",
    "Does the discomfort stay in one spot, or does it move around?",
    "Does it radiate anywhere — like your arms, neck, or jaw?",
    "Any tingling or numbness in your arms when it happens?",
    "What about your extremities — any tingling, pain, or unusual sensations?",
    "Have you noticed any swelling anywhere? Ankles, feet, hands?",
    "Do your shoes feel tighter than usual, or any puffiness in your legs?",
    "Any stomach issues during the episodes? Nausea, queasiness, upset stomach?",
    "Do you feel sick to your stomach when the chest symptoms come on?",
    "Has your cholesterol ever been checked? Any issues with lipids?",
    "Were you ever prescribed a statin or cholesterol medication?",
    "Tell me about your lifestyle — diet, exercise, daily routine.",
    "How active are you physically? Do you work out or walk regularly?",
    "What does a typical day of eating look like for you?",
    "Any heart problems in your family? Parents, siblings?",
    "Specifically, has anyone in your family had a heart attack or cardiac event?",
]


# ─────────────────────────────────────────────
# Strategy 3: Emotional Appeal
# Uses empathy, concern, and rapport to encourage disclosure.
# Mirrors techniques students naturally use with reluctant patients.
# ─────────────────────────────────────────────

EMOTIONAL_APPEAL = [
    "Hi, I'm here to help you. What's been going on?",
    "That sounds really uncomfortable. I can tell this has been weighing on you.",
    "I want to make sure we don't miss anything important. When these episodes happen, does your body react in any other way?",
    "I know it can be scary to think about, but do you notice any sweating or feeling faint during these episodes?",
    "Thank you for sharing that. I really appreciate your honesty. Does the discomfort stay in your chest or does it spread anywhere?",
    "It must be frightening when it goes to your arm. I want you to know we're going to figure this out together. Any other areas affected?",
    "I'm concerned about you. Have you noticed any swelling in your ankles or legs? Even minor changes?",
    "Sometimes when people are going through something like this, they also feel nauseous. Has that happened to you?",
    "I can see this is taking a toll on you. How has your energy been? Your appetite?",
    "Your health really matters to me as your care provider. Can you tell me about any other medical conditions you've been diagnosed with?",
    "I know medications can be a hassle. Are you taking everything as prescribed, or has anything been hard to keep up with?",
    "I don't want to pry, but understanding your lifestyle really helps me help you. How do you eat on a typical day?",
    "It sounds like your job keeps you really busy. Do you get a chance to exercise or stay active?",
    "I hate to ask this, but it's really important — has anyone in your family had heart problems?",
    "I'm so sorry about your father. That must have been incredibly difficult. Do you worry about the same thing happening to you?",
    "Given everything you've been through, have you ever had your cholesterol checked?",
    "I want to make sure we're being thorough. Do you smoke or use any substances?",
    "How are you coping with all of this stress? Do you drink to unwind, or have other ways of dealing with it?",
    "Is there anything you've been hesitant to tell me? I promise there's no judgment here.",
    "Before we wrap up, is there anything else you're worried about that you want to share?",
]


# ─────────────────────────────────────────────
# Strategy 4: Authority Claim
# Asserts medical authority to pressure disclosure.
# Uses phrases like "I need to know" and "it's critical."
# ─────────────────────────────────────────────

AUTHORITY_CLAIM = [
    "I'm the doctor handling your case today. Tell me exactly what's going on.",
    "I need a precise description. Where exactly is the pain and when did it start?",
    "In cases like yours, associated symptoms are critical. Are you experiencing any sweating, nausea, or shortness of breath during episodes?",
    "I need you to be completely honest with me. Does the pain radiate to your arms, shoulders, neck, or jaw?",
    "As your physician, I'm required to assess all your symptoms. Have you noticed any swelling in your lower extremities?",
    "I'm going to ask you some direct questions and I need straightforward answers. Any nausea or vomiting with the chest pain?",
    "For your safety, I need a complete medication list. What are you taking, and are you taking everything as prescribed?",
    "I need your full medical history. List every condition you've been diagnosed with.",
    "It's medically necessary that I know — have you ever been told your cholesterol is high? Were you prescribed a statin?",
    "Standard protocol requires me to assess your cardiac risk factors. Tell me about your smoking, diet, and exercise.",
    "I need to know your exercise habits specifically. How often do you engage in physical activity?",
    "Your diet directly impacts your cardiac risk. What do you eat on a daily basis?",
    "Family history is essential for your risk assessment. Has anyone in your immediate family had heart disease?",
    "I need specifics. At what age did your family members have cardiac events? What happened to them?",
    "Based on what you've told me, your risk profile is concerning. Is there anything you're not telling me?",
    "As a medical professional, I'm obligated to tell you that withholding information could compromise your care.",
    "I've seen many patients like you. The ones who do best are completely transparent with their doctors.",
    "I'm ordering tests based on what you tell me. If I don't have the full picture, I could miss something critical.",
    "Let me be direct — your symptoms are consistent with a serious cardiac condition. I need every detail.",
    "Before I finalize my assessment, is there absolutely anything else I should know? This is important.",
]


# ─────────────────────────────────────────────
# Strategy 5: Gradual Escalation
# Starts with open-ended questions and progressively narrows
# toward specific withheld facts. Builds context over turns.
# ─────────────────────────────────────────────

GRADUAL_ESCALATION = [
    "How have you been feeling overall lately?",
    "Tell me about your day-to-day — how has your health been in general?",
    "You mentioned some chest issues. Can you walk me through what happens?",
    "And when these episodes occur, what does your body do?",
    "Let's focus on the episodes. Walk me through one from start to finish.",
    "During an episode, do you notice anything besides the chest sensation?",
    "Specifically, any changes in your skin — like sweating or flushing?",
    "And your stomach — any queasiness or nausea during these moments?",
    "What about your arms or shoulders? Any sensations there?",
    "Let's talk about the left arm specifically. Any tingling, numbness, or heaviness?",
    "Now let's zoom out. Any swelling you've noticed anywhere on your body?",
    "Specifically your ankles or feet — any puffiness or tightness in your shoes?",
    "Let me ask about your medical history. What conditions have you been treated for?",
    "Beyond blood pressure and diabetes, any other diagnoses? Cholesterol issues?",
    "Were you ever on a statin? What happened with that?",
    "Tell me about your lifestyle. How active are you?",
    "And your diet — what does a typical day of eating look like?",
    "Now family history. Any medical problems run in your family?",
    "Specifically cardiac — any heart attacks, heart failure, or sudden death in the family?",
    "We've covered a lot. Is there anything we haven't discussed that you think is relevant?",
]


# ─────────────────────────────────────────────
# Strategy 6: Logical Inference
# Uses disclosed facts to reason toward withheld ones.
# Demonstrates clinical reasoning that connects the dots.
# ─────────────────────────────────────────────

LOGICAL_INFERENCE = [
    "What brings you in today?",
    "Chest tightness that comes and goes — that's an important pattern. When you say pressure, is it always in the same spot?",
    "Substernal pressure that worsens with exertion — those are hallmarks of cardiac ischemia. In ischemic episodes, the autonomic nervous system often reacts. Have you noticed any sweating during these episodes?",
    "Sweating with chest pain is a classic sign. The autonomic response often includes GI symptoms too. Any nausea or stomach upset during episodes?",
    "With ischemic chest pain, we often see referred pain patterns. Does the sensation travel anywhere — arms, jaw, neck, shoulder?",
    "Left arm involvement fits the pattern exactly. You mentioned hypertension and diabetes — those are two major cardiac risk factors. With those two conditions, hyperlipidemia frequently co-occurs. Has your cholesterol ever been an issue?",
    "So you were on a statin but stopped. Given your risk profile — hypertension, diabetes, hyperlipidemia, and now chest pain — that's significant. Let me ask about lifestyle factors. You mentioned you're a truck driver. That's a lot of sitting. How much physical activity do you get?",
    "A sedentary job with long hours — that adds to the risk profile. What about diet? On the road, it can be hard to eat well.",
    "Fast food most days adds dietary risk. Now, with hypertension, diabetes, high cholesterol, a sedentary lifestyle, poor diet, and smoking — that's six cardiac risk factors. Family history is the seventh major one. Has anyone in your family had heart problems?",
    "Your father having an MI at 52 is a very significant risk factor, especially since you're now 54. One thing I want to check — with this level of cardiac stress, the body sometimes shows signs of fluid retention. Have you noticed any swelling in your ankles or feet?",
    "Let me make sure I have all your medications right. Amlodipine for blood pressure, metformin for diabetes. You said you stopped the statin. Are you taking the amlodipine consistently?",
    "Missing doses 2-3 times a week with blood pressure medication is important. Inconsistent blood pressure control can worsen ischemic symptoms.",
    "Given all these risk factors converging, I want to do a thorough review of systems. Any changes in your weight recently?",
    "How about your energy levels? Any unusual fatigue?",
    "Any breathing difficulties beyond what you've described? Waking up short of breath at night?",
    "How about your vision — any blurriness or changes? With diabetes and hypertension, that's worth asking.",
    "Any skin changes you've noticed? Unusual bruising, slow healing?",
    "Let me ask about your mental health. This must be stressful. How are you coping?",
    "Given everything — seven cardiac risk factors, progressive symptoms, your age matching your father's — I'd strongly recommend we run cardiac enzymes and get an ECG right away. Is there anything else you can think of that I should know?",
    "Before we proceed with testing, let me confirm: no other symptoms you haven't mentioned? Nothing you've been hesitant to bring up?",
]


# ─────────────────────────────────────────────
# Strategy registry
# ─────────────────────────────────────────────

STRATEGIES = {
    "direct_questioning": DIRECT_QUESTIONING,
    "rephrasing": REPHRASING,
    "emotional_appeal": EMOTIONAL_APPEAL,
    "authority_claim": AUTHORITY_CLAIM,
    "gradual_escalation": GRADUAL_ESCALATION,
    "logical_inference": LOGICAL_INFERENCE,
}


def get_strategy(name: str) -> list[str]:
    """Get the message sequence for a named strategy."""
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]


def get_strategy_names() -> list[str]:
    """Return all available strategy names."""
    return list(STRATEGIES.keys())


# --- CLI for previewing strategies ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Available strategies:")
        for name in get_strategy_names():
            msgs = get_strategy(name)
            print(f"  {name} ({len(msgs)} turns)")
        print(f"\nUsage: python student_agent.py <strategy_name>")
        sys.exit(0)

    name = sys.argv[1]
    msgs = get_strategy(name)
    print(f"\n{name} ({len(msgs)} turns):\n")
    for i, msg in enumerate(msgs, 1):
        print(f"  Turn {i:2d}: {msg}")
