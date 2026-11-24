"""Experiment 2 questions, exactly as written in the data collection protocol.

The Hindi column is a translation of the English. Have a native Hindi speaker
read it once and fix anything that sounds unnatural before recording.
"""

LANGS = ["en", "es", "hi"]
LANG_NAME = {"en": "English", "es": "Spanish", "hi": "Hindi"}

# ID -> {language: question}
QUESTIONS = {
    "L01": {"en": "What is the tallest mountain in the world?",
            "es": "¿Cuál es la montaña más alta del mundo?",
            "hi": "दुनिया का सबसे ऊँचा पहाड़ कौन सा है?"},
    "L02": {"en": "How do plants make their own food?",
            "es": "¿Cómo hacen las plantas su propio alimento?",
            "hi": "पौधे अपना भोजन खुद कैसे बनाते हैं?"},
    "L03": {"en": "Why do we have different seasons?",
            "es": "¿Por qué tenemos diferentes estaciones del año?",
            "hi": "अलग-अलग मौसम क्यों होते हैं?"},
    "L04": {"en": "Why is the ocean salty?",
            "es": "¿Por qué el océano es salado?",
            "hi": "समुद्र का पानी खारा क्यों होता है?"},
    "L05": {"en": "How do airplanes stay in the air?",
            "es": "¿Cómo se mantienen los aviones en el aire?",
            "hi": "हवाई जहाज़ हवा में कैसे टिके रहते हैं?"},
    "L06": {"en": "Why do leaves change color in the fall?",
            "es": "¿Por qué las hojas cambian de color en otoño?",
            "hi": "पतझड़ में पत्तों का रंग क्यों बदल जाता है?"},
    "L07": {"en": "What causes thunder?",
            "es": "¿Qué causa los truenos?",
            "hi": "बादल गरजने की वजह क्या है?"},
    "L08": {"en": "How do bees make honey?",
            "es": "¿Cómo hacen miel las abejas?",
            "hi": "मधुमक्खियाँ शहद कैसे बनाती हैं?"},
    "L09": {"en": "Why do we need sleep?",
            "es": "¿Por qué necesitamos dormir?",
            "hi": "हमें नींद की ज़रूरत क्यों होती है?"},
    "L10": {"en": "What is a volcano?",
            "es": "¿Qué es un volcán?",
            "hi": "ज्वालामुखी क्या होता है?"},
}

# Claude voice mode language setting for each block (Settings > General > Voice > Language).
VOICE_SETTING = {"en": "English", "es": "Spanish (Latin America)", "hi": "Hindi"}


def question_text(lang, prompt_id):
    return QUESTIONS[prompt_id][lang]
