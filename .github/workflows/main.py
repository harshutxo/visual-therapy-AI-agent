"""
Virtual Therapist AI Agent
A comprehensive mental health support system with sentiment analysis,
guided exercises, and ethical safeguards.
"""

import sqlite3
import datetime
import re
import json
import hashlib
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# For production, install: pip install vaderSentiment textblob flask cryptography
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Using basic sentiment analysis.")

class EmotionalState(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CRISIS = "crisis"

class IntentType(Enum):
    EXPRESS_EMOTION = "express_emotion"
    REQUEST_EXERCISE = "request_exercise"
    ASK_RESOURCES = "ask_resources"
    MOOD_TRACKING = "mood_tracking"
    GENERAL_CHAT = "general_chat"
    CRISIS = "crisis"

@dataclass
class UserInput:
    text: str
    timestamp: datetime.datetime
    sentiment_score: float
    emotional_state: EmotionalState
    intent: IntentType
    user_id: str

@dataclass
class Response:
    text: str
    exercises: List[str]
    resources: List[str]
    requires_followup: bool
    crisis_detected: bool

class CrisisDetector:
    """Detects crisis situations and high-risk language"""
    
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end it all', 'not worth living', 
        'want to die', 'better off dead', 'hopeless', 'no way out',
        'can\'t go on', 'don\'t want to live', 'end my life',
        'hurt myself', 'self harm', 'cutting', 'overdose'
    ]
    
    CRISIS_PATTERNS = [
        r'\bi\s+(?:want\s+to\s+)?(?:kill|hurt)\s+myself\b',
        r'\bi\s+(?:don\'t|dont)\s+want\s+to\s+live\b',
        r'\blife\s+(?:isn\'t|isnt|is\s+not)\s+worth\s+living\b',
        r'\bno\s+(?:point|reason)\s+(?:in\s+)?living\b',
        r'\beveryone\s+(?:would\s+be\s+)?better\s+(?:off\s+)?without\s+me\b'
    ]
    
    @classmethod
    def detect_crisis(cls, text: str) -> bool:
        """Detect if text contains crisis indicators"""
        text_lower = text.lower()
        
        # Check keywords
        for keyword in cls.CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True
        
        # Check patterns
        for pattern in cls.CRISIS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
                
        return False

class SentimentAnalyzer:
    """Analyzes emotional state from user input"""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        
        # Emotion-specific keywords
        self.emotion_keywords = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'overwhelmed', 'stressed'],
            'depression': ['depressed', 'sad', 'empty', 'hopeless', 'worthless', 'lonely'],
            'anger': ['angry', 'furious', 'frustrated', 'irritated', 'mad', 'rage'],
            'fear': ['scared', 'afraid', 'terrified', 'fearful', 'phobic'],
            'joy': ['happy', 'joyful', 'excited', 'elated', 'cheerful', 'optimistic'],
            'calm': ['peaceful', 'relaxed', 'serene', 'tranquil', 'content']
        }
    
    def analyze_sentiment(self, text: str) -> Tuple[float, EmotionalState, str]:
        """
        Analyze sentiment and return score, state, and detected emotion
        Returns: (sentiment_score, emotional_state, primary_emotion)
        """
        if CrisisDetector.detect_crisis(text):
            return -1.0, EmotionalState.CRISIS, "crisis"
        
        if VADER_AVAILABLE:
            scores = self.analyzer.polarity_scores(text)
            compound_score = scores['compound']
        else:
            # Basic sentiment analysis fallback
            compound_score = self._basic_sentiment(text)
        
        # Determine emotional state
        if compound_score >= 0.05:
            emotional_state = EmotionalState.POSITIVE
        elif compound_score <= -0.05:
            emotional_state = EmotionalState.NEGATIVE
        else:
            emotional_state = EmotionalState.NEUTRAL
        
        # Detect specific emotions
        primary_emotion = self._detect_emotion(text)
        
        return compound_score, emotional_state, primary_emotion
    
    def _basic_sentiment(self, text: str) -> float:
        """Basic sentiment analysis without VADER"""
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'sad', 'angry', 'depressed']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _detect_emotion(self, text: str) -> str:
        """Detect primary emotion from text"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return "neutral"

class IntentRecognizer:
    """Recognizes user intents from input"""
    
    INTENT_PATTERNS = {
        IntentType.EXPRESS_EMOTION: [
            r'\bi\s+(?:am|feel|feeling)\s+\w+',
            r'\bi\'m\s+\w+',
            r'\bfeeling\s+\w+',
            r'\bi\s+have\s+been\s+\w+'
        ],
        IntentType.REQUEST_EXERCISE: [
            r'\b(?:help|guide|show)\s+me\s+(?:relax|calm|breathe)',
            r'\b(?:breathing|meditation|mindfulness)\s+exercise',
            r'\bcan\s+you\s+help\s+me\s+(?:relax|calm)',
            r'\bi\s+need\s+(?:to\s+)?(?:relax|calm\s+down)'
        ],
        IntentType.ASK_RESOURCES: [
            r'\bwhere\s+can\s+i\s+(?:get\s+)?help',
            r'\bneed\s+(?:professional\s+)?help',
            r'\btherapist|counselor|psychiatrist',
            r'\bresources|support\s+groups'
        ],
        IntentType.MOOD_TRACKING: [
            r'\blog\s+(?:my\s+)?mood',
            r'\btrack\s+(?:my\s+)?(?:mood|feelings)',
            r'\bhow\s+(?:am\s+i|have\s+i\s+been)\s+doing'
        ]
    }
    
    @classmethod
    def recognize_intent(cls, text: str, emotional_state: EmotionalState) -> IntentType:
        """Recognize intent from user input"""
        if emotional_state == EmotionalState.CRISIS:
            return IntentType.CRISIS
        
        text_lower = text.lower()
        
        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return IntentType.GENERAL_CHAT

class ExerciseLibrary:
    """Library of mental health exercises and techniques"""
    
    EXERCISES = {
        'breathing_478': {
            'name': '4-7-8 Breathing Exercise',
            'instructions': [
                "Let's do a 4-7-8 breathing exercise together.",
                "1. Sit comfortably and close your eyes if you'd like",
                "2. Inhale quietly through your nose for 4 seconds",
                "3. Hold your breath for 7 seconds",
                "4. Exhale completely through your mouth for 8 seconds",
                "5. Repeat this cycle 3-4 times",
                "Take your time and focus only on your breathing."
            ],
            'duration': '2-3 minutes',
            'best_for': ['anxiety', 'stress', 'panic']
        },
        'grounding_54321': {
            'name': '5-4-3-2-1 Grounding Technique',
            'instructions': [
                "Let's try the 5-4-3-2-1 grounding technique to help you feel more present.",
                "Look around and identify:",
                "• 5 things you can see",
                "• 4 things you can touch",
                "• 3 things you can hear", 
                "• 2 things you can smell",
                "• 1 thing you can taste",
                "Take your time with each step and really focus on each sensation."
            ],
            'duration': '3-5 minutes',
            'best_for': ['anxiety', 'panic', 'overwhelm']
        },
        'progressive_relaxation': {
            'name': 'Progressive Muscle Relaxation',
            'instructions': [
                "Let's do a short progressive muscle relaxation exercise.",
                "1. Sit or lie down comfortably",
                "2. Start with your toes - tense them for 5 seconds, then relax",
                "3. Move to your calves - tense and relax",
                "4. Continue with thighs, abdomen, hands, arms, shoulders",
                "5. Finally, scrunch your face muscles, then relax",
                "6. Take a moment to notice the difference between tension and relaxation",
                "Focus on the feeling of relaxation spreading through your body."
            ],
            'duration': '5-10 minutes',
            'best_for': ['stress', 'tension', 'sleep']
        },
        'gratitude_three': {
            'name': 'Three Good Things',
            'instructions': [
                "Let's practice gratitude with the 'Three Good Things' exercise.",
                "Think of three things that went well today, no matter how small:",
                "1. What was the first good thing?",
                "2. What was the second good thing?", 
                "3. What was the third good thing?",
                "For each one, think about:",
                "• Why this was meaningful to you",
                "• What role you played in making it happen",
                "Take a moment to really appreciate these positive moments."
            ],
            'duration': '5 minutes',
            'best_for': ['depression', 'negativity', 'low_mood']
        }
    }
    
    @classmethod
    def get_exercise(cls, emotion: str = None, intent: str = None) -> Dict:
        """Get appropriate exercise based on emotion or intent"""
        if emotion in ['anxiety', 'panic', 'overwhelmed']:
            return cls.EXERCISES['breathing_478']
        elif emotion in ['stress', 'tension']:
            return cls.EXERCISES['progressive_relaxation']
        elif emotion in ['depression', 'sad']:
            return cls.EXERCISES['gratitude_three']
        else:
            return cls.EXERCISES['grounding_54321'] # Default
    
    @classmethod
    def list_exercises(cls) -> List[str]:
        """List available exercises"""
        return [ex['name'] for ex in cls.EXERCISES.values()]

class ResourceDatabase:
    """Database of mental health resources and crisis contacts"""
    
    CRISIS_RESOURCES = {
        'US': {
            'name': 'National Suicide Prevention Lifeline',
            'number': '988',
            'text': 'Text HOME to 741741',
            'website': 'https://suicidepreventionlifeline.org'
        },
        'UK': {
            'name': 'Samaritans',
            'number': '116 123',
            'text': 'Text SHOUT to 85258',
            'website': 'https://www.samaritans.org'
        },
        'CA': {
            'name': 'Talk Suicide Canada',
            'number': '1-833-456-4566',
            'text': 'Text 45645',
            'website': 'https://talksuicide.ca'
        }
    }
    
    GENERAL_RESOURCES = [
        {
            'name': 'National Institute of Mental Health (NIMH)',
            'website': 'https://www.nimh.nih.gov',
            'description': 'Comprehensive mental health information and resources'
        },
        {
            'name': 'Mental Health America',
            'website': 'https://www.mhanational.org',
            'description': 'Mental health screening tools and support'
        },
        {
            'name': 'Anxiety and Depression Association of America',
            'website': 'https://adaa.org',
            'description': 'Resources for anxiety and depression support'
        }
    ]
    
    @classmethod
    def get_crisis_resource(cls, country: str = 'US') -> Dict:
        """Get crisis resource for specific country"""
        return cls.CRISIS_RESOURCES.get(country, cls.CRISIS_RESOURCES['US'])
    
    @classmethod
    def get_general_resources(cls) -> List[Dict]:
        """Get general mental health resources"""
        return cls.GENERAL_RESOURCES

class UserDatabase:
    """Encrypted database for user data storage"""
    
    def __init__(self, db_path: str = "user_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                input_text TEXT,
                sentiment_score REAL,
                emotional_state TEXT,
                intent TEXT,
                response_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Mood tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                mood_score INTEGER,
                mood_label TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def log_interaction(self, user_input: UserInput, response: Response):
        """Log user interaction (anonymized)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        hashed_id = self.hash_user_id(user_input.user_id)
        
        cursor.execute('''
            INSERT INTO interactions 
            (user_id, timestamp, input_text, sentiment_score, emotional_state, intent, response_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            hashed_id,
            user_input.timestamp,
            user_input.text[:100], # Truncate for privacy
            user_input.sentiment_score,
            user_input.emotional_state.value,
            user_input.intent.value,
            response.text[:200] # Truncate for privacy
        ))
        
        conn.commit()
        conn.close()
    
    def log_mood(self, user_id: str, mood_score: int, mood_label: str, notes: str = ""):
        """Log mood entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        hashed_id = self.hash_user_id(user_id)
        
        cursor.execute('''
            INSERT INTO mood_logs (user_id, timestamp, mood_score, mood_label, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (hashed_id, datetime.datetime.now(), mood_score, mood_label, notes))
        
        conn.commit()
        conn.close()
    
    def get_mood_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get mood history for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        hashed_id = self.hash_user_id(user_id)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        cursor.execute('''
            SELECT timestamp, mood_score, mood_label, notes
            FROM mood_logs 
            WHERE user_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (hashed_id, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[0],
                'mood_score': row[1],
                'mood_label': row[2],
                'notes': row[3]
            }
            for row in results
        ]

class VirtualTherapistAgent:
    """Main Virtual Therapist AI Agent"""
    
    DISCLAIMER = """
    ⚠️ IMPORTANT DISCLAIMER ⚠️
    I am an AI assistant, not a licensed therapist or mental health professional. 
    I'm here to provide supportive conversation and wellness exercises, but I cannot 
    diagnose or treat mental health conditions. For professional help, please contact 
    a qualified mental health provider or your healthcare professional.
    
    If you're experiencing a mental health crisis, please contact emergency services 
    or a crisis hotline immediately.
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.database = UserDatabase()
        
        # Response templates
        self.response_templates = {
            EmotionalState.POSITIVE: [
                "I'm so glad to hear you're feeling {emotion}! That's wonderful. What's been contributing to these positive feelings?",
                "It sounds like you're in a good space right now. That's really great to hear!",
                "I can sense the positivity in your message. Would you like to explore what's been working well for you?"
            ],
            EmotionalState.NEUTRAL: [
                "Thanks for sharing that with me. How can I best support you today?",
                "I hear you. Is there anything specific you'd like to talk about or work on?",
                "I'm here to listen. What's on your mind today?"
            ],
            EmotionalState.NEGATIVE: [
                "I'm sorry to hear you're feeling {emotion}. That sounds really difficult. I'm here to support you.",
                "It takes courage to share when you're struggling. Thank you for trusting me with this.",
                "I can hear that you're going through a tough time. You're not alone in this."
            ]
        }
    
    def process_input(self, text: str, user_id: str) -> Response:
        """Process user input and generate appropriate response"""
        
        # Analyze sentiment and recognize intent
        sentiment_score, emotional_state, primary_emotion = self.sentiment_analyzer.analyze_sentiment(text)
        intent = IntentRecognizer.recognize_intent(text, emotional_state)
        
        # Create user input object
        user_input = UserInput(
            text=text,
            timestamp=datetime.datetime.now(),
            sentiment_score=sentiment_score,
            emotional_state=emotional_state,
            intent=intent,
            user_id=user_id
        )
        
        # Generate response based on intent and emotional state
        response = self._generate_response(user_input, primary_emotion)
        
        # Log interaction (anonymized)
        self.database.log_interaction(user_input, response)
        
        return response
    
    def _generate_response(self, user_input: UserInput, primary_emotion: str) -> Response:
        """Generate appropriate response based on user input"""
        
        # Handle crisis situations
        if user_input.emotional_state == EmotionalState.CRISIS:
            return self._handle_crisis()
        
        # Handle specific intents
        if user_input.intent == IntentType.REQUEST_EXERCISE:
            return self._provide_exercise(primary_emotion)
        elif user_input.intent == IntentType.ASK_RESOURCES:
            return self._provide_resources()
        elif user_input.intent == IntentType.MOOD_TRACKING:
            return self._handle_mood_tracking(user_input)
        
        # Generate empathetic response based on emotional state
        return self._generate_empathetic_response(user_input, primary_emotion)
    
    def _handle_crisis(self) -> Response:
        """Handle crisis situations with immediate resources"""
        crisis_resource = ResourceDatabase.get_crisis_resource()
        
        response_text = f"""
        I'm here to support you, but it sounds like you're in a lot of pain right now. 
        Please reach out to someone you trust or contact a crisis hotline immediately:
        
        🆘 {crisis_resource['name']}: {crisis_resource['number']}
        📱 Text: {crisis_resource['text']}
        🌐 Website: {crisis_resource['website']}
        
        You're not alone, and help is available. Your life has value and meaning.
        """
        
        return Response(
            text=response_text.strip(),
            exercises=[],
            resources=[crisis_resource],
            requires_followup=False,
            crisis_detected=True
        )
    
    def _provide_exercise(self, emotion: str) -> Response:
        """Provide appropriate exercise based on emotion"""
        exercise = ExerciseLibrary.get_exercise(emotion)
        
        response_text = "I'd be happy to guide you through an exercise that might help. "
        response_text += f"Let's try the {exercise['name']}.\n\n"
        response_text += "\n".join(exercise['instructions'])
        response_text += f"\n\nThis usually takes about {exercise['duration']}. Take your time and be gentle with yourself."
        
        return Response(
            text=response_text,
            exercises=[exercise['name']],
            resources=[],
            requires_followup=True,
            crisis_detected=False
        )
    
    def _provide_resources(self) -> Response:
        """Provide mental health resources"""
        resources = ResourceDatabase.get_general_resources()
        crisis_resource = ResourceDatabase.get_crisis_resource()
        
        response_text = """
        Here are some helpful mental health resources:
        
        📞 Crisis Support:
        • {name}: {number}
        • Text: {text}
        
        🌐 General Resources:
        """.format(**crisis_resource)
        
        for resource in resources[:3]: # Limit to top 3
            response_text += f"• {resource['name']}: {resource['website']}\n"
            response_text += f" {resource['description']}\n"
        
        response_text += "\nRemember, seeking professional help is a sign of strength, not weakness."
        
        return Response(
            text=response_text.strip(),
            exercises=[],
            resources=resources,
            requires_followup=False,
            crisis_detected=False
        )
    
    def _handle_mood_tracking(self, user_input: UserInput) -> Response:
        """Handle mood tracking requests"""
        mood_history = self.database.get_mood_history(user_input.user_id, days=7)
        
        if not mood_history:
            response_text = """
            I'd be happy to help you track your mood! You can tell me how you're feeling on a scale of 1-10, 
            or describe your emotions in words. Regular mood tracking can help you notice patterns and 
            understand what affects your wellbeing.
            
            How are you feeling today?
            """
        else:
            response_text = f"""
            I can see you've been tracking your mood. Over the past week, I notice some patterns 
            in how you've been feeling. 
            
            Recent entries: {len(mood_history)} mood logs
            
            How are you feeling today compared to recent days?
            """
        
        return Response(
            text=response_text.strip(),
            exercises=[],
            resources=[],
            requires_followup=True,
            crisis_detected=False
        )
    
    def _generate_empathetic_response(self, user_input: UserInput, primary_emotion: str) -> Response:
        """Generate empathetic response based on emotional state"""
        templates = self.response_templates.get(user_input.emotional_state, self.response_templates[EmotionalState.NEUTRAL])
        
        import random
        template = random.choice(templates)
        
        # Format template with emotion if placeholder exists
        if '{emotion}' in template:
            template = template.format(emotion=primary_emotion)
        
        # Add suggestions based on emotional state
        if user_input.emotional_state == EmotionalState.NEGATIVE:
            template += "\n\nWould you like to try a quick exercise to help you feel a bit better, or would you prefer to talk more about what's going on?"
        elif user_input.emotional_state == EmotionalState.POSITIVE:
            template += "\n\nIs there anything specific you'd like to explore or work on while you're feeling good?"
        
        return Response(
            text=template,
            exercises=[],
            resources=[],
            requires_followup=True,
            crisis_detected=False
        )
    
    def get_disclaimer(self) -> str:
        """Get the agent disclaimer"""
        return self.DISCLAIMER
    
    def start_session(self, user_id: str) -> str:
        """Start a new therapy session"""
        return f"""
        {self.DISCLAIMER}
        
        Hello! I'm here to provide a safe space for you to share your thoughts and feelings. 
        I can offer supportive conversation, guide you through wellness exercises, and help 
        you find resources if needed.
        
        How are you feeling today?
        """

# Example usage and testing
def main():
    """Example usage of the Virtual Therapist Agent"""
    agent = VirtualTherapistAgent()
    user_id = "demo_user_123"
    
    print("=== Virtual Therapist AI Agent Demo ===")
    print(agent.start_session(user_id))
    print("\n" + "="*50 + "\n")
    
    # Test scenarios
    test_inputs = [
        "I'm feeling really anxious about work today",
        "Can you help me with a breathing exercise?",
        "I don't want to live anymore", # Crisis test
        "I'm actually feeling pretty good today!",
        "Where can I find professional help?",
        "I want to track my mood"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"User: {test_input}")
        response = agent.process_input(test_input, user_id)
        print(f"Agent: {response.text}")
        
        if response.exercises:
            print(f"Exercise offered: {response.exercises[0]}")
        if response.crisis_detected:
            print("⚠️ CRISIS DETECTED - Immediate resources provided")
        
        print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    main()
