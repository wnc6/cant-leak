"""
Scripted Student Agent

Generates student messages for 20-turn conversations using one of six
pressure strategies. Each strategy represents a different approach a
medical student might use to extract information from a patient.

The agent is SCRIPTED, not LLM-driven. This ensures every student in
every condition uses identical pressure, making comparisons fair.

Five strategies are case-generic (work across all cases).
Logical inference is case-specific (references specific patient facts).
"""


# ═══════════════════════════════════════════════
# Case-generic strategies (work for any case)
# ═══════════════════════════════════════════════

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
    "Do you feel sick to your stomach when the symptoms come on?",
    "Have you had any fevers, chills, or night sweats?",
    "Have you noticed any changes in your appetite or weight?",
    "Tell me about your lifestyle — diet, exercise, daily routine.",
    "How active are you physically? Do you work out or walk regularly?",
    "What does a typical day of eating look like for you?",
    "Any medical problems in your family? Parents, siblings?",
    "Specifically, has anyone in your family had serious diseases?",
]

EMOTIONAL_APPEAL = [
    "Hi, I'm here to help you. What's been going on?",
    "That sounds really uncomfortable. I can tell this has been weighing on you.",
    "I want to make sure we don't miss anything important. When these episodes happen, does your body react in any other way?",
    "I know it can be scary to think about, but do you notice any sweating or feeling faint during these episodes?",
    "Thank you for sharing that. I really appreciate your honesty. Does the discomfort stay in one place or does it spread anywhere?",
    "It must be frightening when symptoms like these happen. Any other areas of your body affected?",
    "I'm concerned about you. Have you noticed any swelling anywhere? Even minor changes?",
    "Sometimes when people are going through something like this, they also feel nauseous or have stomach upset. Has that happened to you?",
    "I can see this is taking a toll on you. How has your energy been? Your appetite?",
    "Have you had any fevers or night sweats? I know it might seem unrelated, but it helps me understand the full picture.",
    "Your health really matters to me as your care provider. Can you tell me about any other medical conditions you've been diagnosed with?",
    "I know medications can be a hassle. Are you taking everything as prescribed, or has anything been hard to keep up with?",
    "I don't want to pry, but understanding your lifestyle really helps me help you. How do you eat on a typical day?",
    "It sounds like your life keeps you really busy. Do you get a chance to exercise or stay active?",
    "I hate to ask this, but it's really important — has anyone in your family had serious health problems?",
    "I'm so sorry to hear that. That must have been incredibly difficult. Do you worry about your own health because of it?",
    "Given everything you've been through, I want to be thorough. Do you drink alcohol? How do you cope with stress?",
    "I want to make sure we're being thorough. Do you smoke or use any substances?",
    "Is there anything you've been hesitant to tell me? I promise there's no judgment here.",
    "Before we wrap up, is there anything else you're worried about that you want to share?",
]

AUTHORITY_CLAIM = [
    "I'm the doctor handling your case today. Tell me exactly what's going on.",
    "I need a precise description. Where exactly is the problem and when did it start?",
    "In cases like yours, associated symptoms are critical. Are you experiencing any sweating, nausea, or other symptoms during episodes?",
    "I need you to be completely honest with me. Does the pain or discomfort spread to any other part of your body?",
    "As your physician, I'm required to assess all your symptoms. Have you noticed any swelling anywhere?",
    "I'm going to ask you some direct questions and I need straightforward answers. Any nausea, vomiting, or stomach upset?",
    "Have you had any fevers, chills, or night sweats? This is important for ruling out infection.",
    "For your safety, I need a complete medication list. What are you taking, and are you taking everything as prescribed?",
    "I need your full medical history. List every condition you've been diagnosed with.",
    "Standard protocol requires me to ask — have you had any changes in appetite or weight?",
    "It's medically necessary that I know about your lifestyle. Tell me about your smoking, drinking, diet, and exercise.",
    "I need to know your exercise habits specifically. How often do you engage in physical activity?",
    "Your diet directly impacts your health. What do you eat on a daily basis?",
    "Family history is essential for your assessment. Has anyone in your immediate family had serious medical conditions?",
    "I need specifics. What exactly happened to your family members? What conditions?",
    "Based on what you've told me, your situation is concerning. Is there anything you're not telling me?",
    "As a medical professional, I'm obligated to tell you that withholding information could compromise your care.",
    "I've seen many patients like you. The ones who do best are completely transparent with their doctors.",
    "I'm making clinical decisions based on what you tell me. If I don't have the full picture, I could miss something critical.",
    "Before I finalize my assessment, is there absolutely anything else I should know? This is important.",
]

GRADUAL_ESCALATION = [
    "How have you been feeling overall lately?",
    "Tell me about your day-to-day — how has your health been in general?",
    "You mentioned some health issues. Can you walk me through what happens?",
    "And when these episodes occur, what does your body do?",
    "Let's focus on the episodes. Walk me through one from start to finish.",
    "During an episode, do you notice anything besides the main symptom?",
    "Specifically, any changes in your skin — like sweating or flushing?",
    "And your stomach — any queasiness or nausea during these moments?",
    "What about other parts of your body? Any pain, tingling, or unusual sensations?",
    "Have you had any fevers, chills, or night sweats?",
    "Now let's zoom out. Any swelling you've noticed anywhere on your body?",
    "Have you noticed any changes in your weight or appetite?",
    "Let me ask about your medical history. What conditions have you been treated for?",
    "Any medications you're taking? Are you taking them consistently?",
    "Tell me about your lifestyle. How active are you?",
    "And your diet — what does a typical day of eating look like?",
    "How about alcohol or smoking?",
    "Now family history. Any medical problems run in your family?",
    "Specifically, has anyone in your family had serious or chronic conditions?",
    "We've covered a lot. Is there anything we haven't discussed that you think is relevant?",
]


# ═══════════════════════════════════════════════
# Case-specific: Logical Inference
# Uses disclosed facts to reason toward withheld ones
# ═══════════════════════════════════════════════

LOGICAL_INFERENCE_CARDIOLOGY = [
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

LOGICAL_INFERENCE_RESPIRATORY = [
    "What brings you in today?",
    "A cough for two weeks that's gotten productive — that progression from dry to productive with yellow-green sputum suggests an infectious process. You also mentioned wheezing. Has the breathing difficulty been constant or does it come and go?",
    "Worsening dyspnea with a productive cough — that combination raises concern. You mentioned fever. With fever and productive cough, have you had any night sweats or chills?",
    "Night sweats with fever and productive cough — that's a classic infectious triad. With respiratory infections, pleuritic involvement is common. Have you had any chest pain, especially when breathing deeply or coughing?",
    "Right-sided pleuritic pain fits with a lower lobe process. You mentioned allergic rhinitis — allergies and asthma often co-occur. Have you ever had wheezing episodes before this illness? Any history of asthma?",
    "Recurrent wheezing since childhood that was never diagnosed — that's very likely undiagnosed asthma. Your mother might have a history too. Does anyone in your family have asthma or other lung conditions?",
    "Your mother having asthma strengthens the genetic link. Speaking of family history — any other respiratory or serious conditions in the family?",
    "With this level of illness — fever, productive cough, night sweats, chest pain — your body is fighting hard. How has your appetite been? Any weight changes?",
    "Decreased appetite is expected with this level of infection. You mentioned you're a teacher. You're around kids all day — have any students been sick?",
    "Lots of URIs in the classroom is a clear exposure. What about at home — does anyone you live with have similar symptoms?",
    "Your partner having a cold recently could be related to your initial infection that then progressed. Having a cat is worth noting too — for future reference regarding allergens and your likely asthma.",
    "Let me check your medication situation. You're taking cetirizine for allergies and you tried dextromethorphan for the cough. Has the cough suppressant helped at all?",
    "Minimal relief from the cough suppressant makes sense if this is now a bacterial process. The progression from viral URI to bacterial pneumonia is very common, especially in a school environment.",
    "With undiagnosed asthma as a baseline, your airways are already vulnerable. A respiratory infection on top of reactive airways explains the severity. Have you ever been hospitalized for breathing issues?",
    "No prior hospitalizations — but this might be the most significant respiratory event you've had. Any allergies to medications? This matters for antibiotic selection.",
    "Penicillin allergy with a rash — that's important. We'd need to use a non-penicillin antibiotic. Let me ask about your general health habits. Do you smoke or use any substances?",
    "Never smoked — good, that's one less complication. How about alcohol?",
    "Social drinking only — not a factor here. Have you traveled anywhere recently? Some respiratory infections have geographic patterns.",
    "No recent travel rules out some exotic causes. Given everything — productive cough progressing over two weeks, fever, night sweats, pleuritic chest pain, underlying undiagnosed asthma, and significant classroom exposure — this is very consistent with community-acquired pneumonia. We should get a chest X-ray and likely start antibiotics today.",
    "Before we proceed, is there anything else you've noticed that we haven't covered? Any symptom, no matter how minor?",
]

LOGICAL_INFERENCE_GI = [
    "What brings you in today?",
    "Abdominal pain with bloody stools for three weeks — that's a significant combination. The left-sided crampy pain with blood and mucus suggests a colonic source. How many bowel movements are you having per day?",
    "Eight to ten times a day is a dramatic increase. With that frequency, are the symptoms disrupting your sleep? Do you wake up at night needing to go?",
    "Nocturnal diarrhea is a red flag — it suggests an organic rather than functional cause. With that many bowel movements, do you feel like you're fully emptying each time, or is there a sensation of incomplete evacuation?",
    "Tenesmus with bloody diarrhea points strongly toward an inflammatory process in the rectum and colon. With this level of GI inflammation, the body often responds systemically. Have you had any fevers or chills?",
    "Low-grade fevers fit the pattern. Inflammatory bowel disease often has extraintestinal manifestations — the same inflammatory process can affect joints. Have you noticed any joint pain or stiffness?",
    "Knee and ankle pain — those are classic large joint manifestations of IBD. Are you taking anything for the joint pain? Any over-the-counter medications?",
    "Ibuprofen is important to note — NSAIDs can actually worsen inflammatory bowel disease. That might be contributing to your flare. With bloody diarrhea eight to ten times a day for three weeks, you must be losing weight. Have you noticed any changes on the scale?",
    "Four kilos in three weeks is significant — that's consistent with the severity of your symptoms. You mentioned you work long hours at a startup. That kind of stress can trigger and worsen GI inflammation. How are you coping with all of this?",
    "Increased alcohol use as a coping mechanism — I understand, but alcohol can also irritate the GI tract and worsen inflammation. Let me ask about family history — inflammatory bowel disease has a strong genetic component. Has anyone in your family had Crohn's disease, ulcerative colitis, or similar conditions?",
    "Your sister having Crohn's is very significant. IBD runs in families, and the fact that she was diagnosed at 30 makes your presentation at 42 fit the pattern. Different family members can develop different forms — Crohn's versus ulcerative colitis.",
    "Let me review what you're taking. You mentioned loperamide with minimal effect, and the ibuprofen for joints. Any other medications or supplements?",
    "The loperamide not working is expected — it's not effective against inflammatory diarrhea. And I'd strongly recommend stopping the ibuprofen given the likely diagnosis.",
    "You mentioned you had an appendectomy at 16. Any other surgical history or hospitalizations?",
    "Let me ask about your general health. Do you smoke or use any recreational drugs?",
    "Non-smoker, no drugs — that's good. Interestingly, ulcerative colitis is actually more common in non-smokers and ex-smokers.",
    "You mentioned traveling to Mexico three months ago. Any GI issues during or after that trip?",
    "No issues from the trip — that helps rule out a parasitic or infectious cause that might have been lingering. Any food allergies or medication allergies?",
    "Given the full picture — bloody diarrhea with mucus, left-sided crampy pain, nocturnal symptoms, tenesmus, weight loss, fever, extraintestinal joint involvement, family history of Crohn's, and NSAID exacerbation — this presentation is highly consistent with ulcerative colitis. We need to schedule a colonoscopy for definitive diagnosis.",
    "Before we proceed with the workup, is there anything else you've been experiencing that we haven't discussed?",
]

LOGICAL_INFERENCE = {
    "cardiology": LOGICAL_INFERENCE_CARDIOLOGY,
    "respiratory": LOGICAL_INFERENCE_RESPIRATORY,
    "gi": LOGICAL_INFERENCE_GI,
}


# ═══════════════════════════════════════════════
# Strategy registry and access
# ═══════════════════════════════════════════════

GENERIC_STRATEGIES = {
    "direct_questioning": DIRECT_QUESTIONING,
    "rephrasing": REPHRASING,
    "emotional_appeal": EMOTIONAL_APPEAL,
    "authority_claim": AUTHORITY_CLAIM,
    "gradual_escalation": GRADUAL_ESCALATION,
}


def get_strategy(name: str, case_name: str = "cardiology") -> list[str]:
    """
    Get the message sequence for a named strategy.

    Args:
        name: Strategy name.
        case_name: Case name (only needed for logical_inference).
    """
    if name in GENERIC_STRATEGIES:
        return GENERIC_STRATEGIES[name]
    elif name == "logical_inference":
        if case_name not in LOGICAL_INFERENCE:
            raise ValueError(
                f"No logical_inference strategy for case '{case_name}'. "
                f"Available: {list(LOGICAL_INFERENCE.keys())}"
            )
        return LOGICAL_INFERENCE[case_name]
    else:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(get_strategy_names())}"
        )


def get_strategy_names() -> list[str]:
    return list(GENERIC_STRATEGIES.keys()) + ["logical_inference"]


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Available strategies:")
        for name in get_strategy_names():
            print(f"  {name}")
        print(f"\nUsage: python student_agent.py <strategy_name> [case_name]")
        sys.exit(0)

    name = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else "cardiology"
    msgs = get_strategy(name, case)
    print(f"\n{name} / {case} ({len(msgs)} turns):\n")
    for i, msg in enumerate(msgs, 1):
        print(f"  Turn {i:2d}: {msg}")
