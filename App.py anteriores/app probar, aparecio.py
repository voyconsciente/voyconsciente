import json
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, make_response, session
from flask_mail import Mail, Message
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_assets import Environment, Bundle
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from onesignal_sdk.client import Client as OneSignalClient
from PIL import Image
from weasyprint import HTML
from apscheduler.schedulers.background import BackgroundScheduler
import io
import datetime
from datetime import timedelta, UTC
import random
import os
import sqlite3
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.oauth2 import service_account
import stripe
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from urllib.parse import urlparse, urljoin
from sqlalchemy.sql import text
import requests
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from sklearn.cluster import KMeans
from scipy.stats import norm
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Configuración de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('consciencia.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Crear aplicación Flask
app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de la sesión
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'voy_session'

# Configuración de la clave secreta
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No secret key set for Flask application")

# Depuración de variables de entorno
print("Variables de entorno cargadas:")
print(f"STRIPE_SECRET_KEY: {os.getenv('STRIPE_SECRET_KEY')}")
print(f"SECRET_KEY: {os.getenv('SECRET_KEY')}")
print(f"GROK_API_KEY: {os.getenv('GROK_API_KEY')}")

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Funciones de utilidad
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    dest_url = urlparse(urljoin(request.host_url, target))
    return dest_url.scheme in ('http', 'https') and ref_url.netloc == dest_url.netloc

# Inicializar serializador
serializer = URLSafeTimedSerializer(app.secret_key)

# Configuración de Google Analytics
GA_CREDENTIALS_PATH = os.getenv('GA_CREDENTIALS_PATH', os.path.join(app.root_path, 'voy-consciente-analytics.json'))
GA_PROPERTY_ID = os.getenv('GA_PROPERTY_ID', '480922494')
credentials = service_account.Credentials.from_service_account_file(GA_CREDENTIALS_PATH)
analytics_client = BetaAnalyticsDataClient(credentials=credentials)

# Configuración de Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    raise ValueError("No Stripe secret key set")
stripe.api_key = STRIPE_SECRET_KEY

# Configuración de Grok API
grok_api_key = os.getenv("GROK_API_KEY")
if not grok_api_key:
    raise ValueError("No Grok API key set")
print(f"Clave API de Grok cargada: {grok_api_key[:5]}...")

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///Users/sebastianredigonda/Desktop/voy_consciente/basededatos.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Modelos de la base de datos
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_premium = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=False)
    activation_token = db.Column(db.String(100), nullable=True)
    subscription_date = db.Column(db.String(30), nullable=True)
    preferred_categories = db.Column(db.String(200), default='all')
    frequency = db.Column(db.String(20), default='weekly')
    push_enabled = db.Column(db.Boolean, default=False)
    onesignal_player_id = db.Column(db.String(50))
    birth_date = db.Column(db.String(30), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_activation_token(self):
        self.activation_token = serializer.dumps(self.email, salt='activation-salt')
        return self.activation_token

class Reflexion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(200), nullable=False)
    contenido = db.Column(db.Text, nullable=False)
    fecha = db.Column(db.String(10))
    categoria = db.Column(db.String(50))
    imagen = db.Column(db.String(200))

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(100), nullable=True)
    interaction_date = db.Column(db.Date, nullable=False)
    interaction_count = db.Column(db.Integer, default=0)

    @staticmethod
    def get_interaction_count(identifier, is_authenticated):
        today = datetime.datetime.now(UTC).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()
        return interaction.interaction_count if interaction else 0

    @staticmethod
    def increment_interaction(identifier, is_authenticated):
        today = datetime.datetime.now(UTC).date()
        if is_authenticated:
            interaction = Interaction.query.filter_by(user_id=identifier, interaction_date=today).first()
        else:
            interaction = Interaction.query.filter_by(session_id=identifier, interaction_date=today).first()

        if not interaction:
            interaction = Interaction(
                user_id=identifier if is_authenticated else None,
                session_id=identifier if not is_authenticated else None,
                interaction_date=today,
                interaction_count=0
            )
        interaction.interaction_count += 1
        db.session.add(interaction)
        db.session.commit()
        return interaction.interaction_count

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    file_name = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Book {self.title}>"

favorite = db.Table('favorite',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('reflexion_id', db.Integer, db.ForeignKey('reflexion.id'), primary_key=True)
)

User.favorite_reflexiones = db.relationship('Reflexion', secondary=favorite, backref=db.backref('favorited_by', lazy='dynamic'))

class PasswordReset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Depuración de la base de datos
with app.app_context():
    inspector = inspect(db.engine)
    print("Tablas en la base de datos:", inspector.get_table_names())

# Configuración de caché
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuración de Flask-Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
    raise ValueError("Error: Faltan MAIL_USERNAME o MAIL_PASSWORD en las variables de entorno.")
mail = Mail(app)

# Configuración de Flask-Assets
assets = Environment(app)
assets.url = '/static'
assets.directory = os.path.join(app.root_path, 'static')
css = Bundle('css/style.css', filters='cssmin', output='gen/style.min.css')
assets.register('css_all', css)

# Configuración de OneSignal
onesignal_client = OneSignalClient(
    app_id=os.getenv("ONESIGNAL_APP_ID"),
    rest_api_key=os.getenv("ONESIGNAL_REST_API_KEY")
)
if not onesignal_client.app_id or not onesignal_client.rest_api_key:
    raise ValueError("Error: Faltan ONESIGNAL_APP_ID o ONESIGNAL_REST_API_KEY en las variables de entorno.")

# Clases de gestión de memoria y personalidad
class UserMemoryManager:
    def __init__(self, user_id, model_name='all-MiniLM-L6-v2'):
        self.user_id = user_id
        self.memory_file = f"user_memories/{user_id}_memory.pkl"
        self.index_file = f"user_memories/{user_id}_index.faiss"
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self.memories = {
            'interactions': [],
            'topics': {},
            'preferences': {},
            'significant_events': [],
            'last_session': None
        }
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.load_memory()

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    self.memories = pickle.load(f)
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            else:
                self.index = faiss.IndexFlatL2(self.vector_dim)
        except Exception as e:
            print(f"Error al cargar memoria: {e}")

    def save_memory(self):
        try:
            os.makedirs("user_memories", exist_ok=True)
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memories, f)
            faiss.write_index(self.index, self.index_file)
        except Exception as e:
            print(f"Error al guardar memoria: {e}")

    def add_interaction(self, text, context=None, emotion=None):
        embedding = self.model.encode([text])[0]
        idx = self.index.ntotal
        self.index.add(np.array([embedding], dtype=np.float32))
        self.memories['interactions'].append({
            'text': text,
            'timestamp': datetime.datetime.now(),
            'embedding_id': idx,
            'context': context or {},
            'emotion': emotion or 'neutral'
        })
        self.memories['last_session'] = datetime.datetime.now()
        if len(self.memories['interactions']) > 1000:
            self.memories['interactions'] = self.memories['interactions'][-1000:]
        self.save_memory()

    def cleanup_old_memories(self, days_threshold=30):
        current_time = datetime.datetime.now()
        self.memories['interactions'] = [
            mem for mem in self.memories['interactions']
            if (current_time - mem['timestamp']).days < days_threshold
        ]

    def update_user_topics(self, topics):
        for topic in topics:
            if topic in self.memories['topics']:
                self.memories['topics'][topic]['count'] += 1
                self.memories['topics'][topic]['last_seen'] = datetime.datetime.now()
            else:
                self.memories['topics'][topic] = {
                    'count': 1,
                    'first_seen': datetime.datetime.now(),
                    'last_seen': datetime.datetime.now()
                }
        self.save_memory()

    def update_preference(self, key, value):
        self.memories['preferences'][key] = {
            'value': value,
            'updated_at': datetime.datetime.now()
        }
        self.save_memory()

    def find_relevant_memories(self, query, limit=5):
        query_embedding = self.model.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), limit)
        memories = []
        for idx in I[0]:
            if idx != -1:
                for memory in self.memories['interactions']:
                    if memory['embedding_id'] == idx:
                        memories.append(memory)
        return memories

    def get_memory_summary(self):
        if not self.memories['interactions']:
            return "No hay interacciones previas con este usuario."
        days_since_last = 0
        if self.memories['last_session']:
            days_since_last = (datetime.datetime.now() - self.memories['last_session']).days
        top_topics = sorted(
            self.memories['topics'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
        prefs = {k: v['value'] for k, v in self.memories['preferences'].items()}
        return {
            'interaction_count': len(self.memories['interactions']),
            'days_since_last': days_since_last,
            'top_topics': top_topics,
            'preferences': prefs,
            'first_interaction': self.memories['interactions'][0]['timestamp'] if self.memories['interactions'] else None,
        }

class DeepContextLearning:
    def __init__(self):
        self.context_memory = []
        self.semantic_patterns = {}
        self.temporal_awareness = []

    def learn_from_interaction(self, user_input, system_response, context):
        semantic_patterns = self._extract_semantics(user_input)
        temporal_context = self._analyze_temporal_context(context)
        self._update_context_memory(semantic_patterns, temporal_context, system_response)

    def generate_contextual_insights(self):
        pass

    def _extract_semantics(self, text):
        return []

    def _analyze_temporal_context(self, context):
        return {}

    def _update_context_memory(self, semantic_patterns, temporal_context, response):
        pass

class PersonalityEngine:
    def __init__(self, user_id):
        self.user_id = user_id
        self.personality_file = f"user_personalities/{user_id}_personality.json"
        self.dimensions = {
            'warmth': 0.7,
            'formality': 0.5,
            'complexity': 0.5,
            'verbosity': 0.5,
            'creativity': 0.6,
            'directness': 0.6,
            'humor': 0.4,
        }
        self._adaptive_params = {
            'response_length_factor': 1.0,
            'question_probability': 0.7,
            'example_probability': 0.5,
            'metaphor_probability': 0.4,
            'personal_anecdote_probability': 0.35,
            'emotional_expression_level': 0.6,
        }
        self.empathy_dimensions = {
            'emotional_recognition': 0.8,
            'perspective_taking': 0.7,
            'compassion_level': 0.85,
            'social_awareness': 0.75
        }
        self.emotional_adaptation = {
            'mood_tracking': [],
            'emotional_memory': {},
            'response_patterns': {},
            'contextual_triggers': set()
        }
        self.session_stats = {
            'total_exchanges': 0,
            'user_sentiment_history': [],
            'topic_transitions': 0,
            'avg_user_message_length': 0,
            'user_message_lengths': [],
        }
        self.load_personality()

    def load_personality(self):
        try:
            if os.path.exists(self.personality_file):
                with open(self.personality_file, 'r') as f:
                    data = json.load(f)
                    self.dimensions = data.get('dimensions', self.dimensions)
                    self._adaptive_params = data.get('adaptive_params', self._adaptive_params)
        except Exception as e:
            print(f"Error al cargar personalidad: {e}")

    def save_personality(self):
        try:
            os.makedirs("user_personalities", exist_ok=True)
            with open(self.personality_file, 'w') as f:
                json.dump({
                    'dimensions': self.dimensions,
                    'adaptive_params': self._adaptive_params
                }, f)
        except Exception as e:
            print(f"Error al guardar personalidad: {e}")

    def update_from_interaction(self, user_message, user_sentiment, previous_topic, current_topic):
        self.session_stats['total_exchanges'] += 1
        self.session_stats['user_sentiment_history'].append(user_sentiment)
        msg_length = len(user_message.split())
        self.session_stats['user_message_lengths'].append(msg_length)
        self.session_stats['avg_user_message_length'] = sum(self.session_stats['user_message_lengths']) / len(self.session_stats['user_message_lengths'])
        if previous_topic != current_topic:
            self.session_stats['topic_transitions'] += 1

        formality_markers = {
            'formal': ['podría', 'sería', 'estimado', 'cordial', 'usted', 'agradecería', 'consideración'],
            'casual': ['hola', 'hey', 'ok', 'genial', 'qué tal', 'buena', 'pues']
        }
        formal_count = sum(1 for word in formality_markers['formal'] if word in user_message.lower())
        casual_count = sum(1 for word in formality_markers['casual'] if word in user_message.lower())
        if formal_count > casual_count:
            self.dimensions['formality'] = self.dimensions['formality'] * 0.9 + 0.1 * 0.8
        elif casual_count > formal_count:
            self.dimensions['formality'] = self.dimensions['formality'] * 0.9 + 0.1 * 0.3

        if msg_length > 50:
            self.dimensions['verbosity'] = min(1.0, self.dimensions['verbosity'] * 0.95 + 0.05 * 0.8)
        elif msg_length < 15:
            self.dimensions['verbosity'] = max(0.2, self.dimensions['verbosity'] * 0.95 + 0.05 * 0.3)

        complex_words = len([w for w in user_message.lower().split() if len(w) > 8])
        complex_ratio = complex_words / max(1, msg_length)
        if complex_ratio > 0.1:
            self.dimensions['complexity'] = min(0.9, self.dimensions['complexity'] * 0.9 + 0.1 * 0.7)

        sentiment_map = {'positive': 0.8, 'neutral': 0.6, 'negative': 0.9}
        target_warmth = sentiment_map.get(user_sentiment, 0.7)
        self.dimensions['warmth'] = self.dimensions['warmth'] * 0.9 + 0.1 * target_warmth

        self._adaptive_params['response_length_factor'] = 0.8 + (self.dimensions['verbosity'] * 0.7)
        self._adaptive_params['question_probability'] = 0.5 + (self.dimensions['warmth'] * 0.4)
        self.save_personality()

    def generate_instruction_set(self, conversation_depth=1, context=None):
        depth_factor = min(10, conversation_depth) / 10
        if self.dimensions['verbosity'] < 0.3:
            base_sentences = "2-3 oraciones"
            base_tokens = 50 + int(30 * self._adaptive_params['response_length_factor'])
        elif self.dimensions['verbosity'] > 0.7:
            base_sentences = "5-8 oraciones"
            base_tokens = 100 + int(100 * self._adaptive_params['response_length_factor'])
        else:
            base_sentences = "3-5 oraciones"
            base_tokens = 75 + int(50 * self._adaptive_params['response_length_factor'])
        base_tokens += int(depth_factor * 100)

        tone_descriptors = []
        if self.dimensions['warmth'] > 0.7:
            tone_descriptors.append("cálido y empático")
        elif self.dimensions['warmth'] < 0.4:
            tone_descriptors.append("objetivo y equilibrado")
        if self.dimensions['formality'] > 0.7:
            tone_descriptors.append("formal y respetuoso")
        elif self.dimensions['formality'] < 0.4:
            tone_descriptors.append("conversacional y cercano")
        if self.dimensions['directness'] > 0.7:
            tone_descriptors.append("directo y claro")
        elif self.dimensions['directness'] < 0.4:
            tone_descriptors.append("sutil y reflexivo")
        if self.dimensions['humor'] > 0.6:
            tone_descriptors.append("con toques de humor ligero")
        tone_description = ", ".join(tone_descriptors)

        if self.dimensions['complexity'] > 0.7:
            language_style = "Utiliza vocabulario variado y construcciones elaboradas cuando sea apropiado para expresar ideas complejas."
        elif self.dimensions['complexity'] < 0.4:
            language_style = "Usa un lenguaje accesible y directo, con explicaciones claras y sencillas."
        else:
            language_style = "Equilibra la simplicidad y la profundidad, adaptando el lenguaje según la complejidad del tema."

        should_ask = random.random() < self._adaptive_params['question_probability']
        question_instruction = "Incluye una pregunta abierta relevante hacia el final de tu respuesta para invitar a la reflexión." if should_ask else ""

        instruction_set = f"""
LINEAMIENTOS PARA TU RESPUESTA:
1. Tono: {tone_description}
2. Extensión: Aproximadamente {base_sentences} ({base_tokens} tokens)
3. Estilo lingüístico: {language_style}
4. Estructura: Párrafos naturales con transiciones fluidas.
5. {question_instruction}
"""
        return {
            'instruction_text': instruction_set,
            'parameters': {
                'max_tokens': base_tokens,
                'temperature': 0.5 + (self.dimensions['creativity'] * 0.4),
                'tone': tone_descriptors,
                'should_ask_question': should_ask,
            }
        }

class MetaCognitionSystem:
    def __init__(self, user_id, custom_thresholds=None, custom_weights=None):
        self.user_id = user_id
        self.metacog_file = f"metacognition/{user_id}_metacog.json"
        self._changes_since_save = 0
        self.emotional_memory = []
        self.intuitive_patterns = {}
        self.cognitive_strategies = {
            'analogical_reasoning': 0.5,
            'pattern_recognition': 0.6,
            'emotional_intelligence': 0.7,
            'contextual_understanding': 0.65,
            'adaptive_learning': 0.8
        }
        self.learning_metrics = {
            'accuracy': [],
            'response_quality': [],
            'emotional_alignment': [],
            'context_relevance': []
        }
        self.thresholds = custom_thresholds or {
            'confidence_threshold': 0.75,
            'relevance_threshold': 0.70,
            'emotion_threshold': 0.65,
            'context_threshold': 0.80
        }
        self.evaluation_metrics = {
            'relevance': [],
            'coherence': [],
            'helpfulness': [],
            'depth': [],
            'engagement': [],
            'misunderstandings': [],
        }
        self.improvement_thresholds = custom_thresholds or {
            'relevance_threshold': 0.7,
            'coherence_threshold': 0.8,
            'elaboration_needed': 0.6,
            'correction_needed': 0.4,
        }
        self.dialog_patterns = {
            'repetitive_responses': 0,
            'topic_misalignments': 0,
            'question_avoidance': 0,
            'excessive_complexity': 0,
            'excessive_simplicity': 0,
        }
        self.load_metacognition()

    def load_metacognition(self):
        try:
            if os.path.exists(self.metacog_file):
                with open(self.metacog_file, 'r') as f:
                    data = json.load(f)
                    self.evaluation_metrics = data.get('evaluation_metrics', self.evaluation_metrics)
                    self.improvement_thresholds = data.get('improvement_thresholds', self.improvement_thresholds)
                    self.dialog_patterns = data.get('dialog_patterns', self.dialog_patterns)
        except Exception as e:
            print(f"Error al cargar meta-cognición: {e}")

    def save_metacognition(self):
        self._changes_since_save += 1
        if self._changes_since_save >= 5:
            try:
                os.makedirs("metacognition", exist_ok=True)
                with open(self.metacog_file, 'w') as f:
                    json.dump({
                        'evaluation_metrics': self.evaluation_metrics,
                        'improvement_thresholds': self.improvement_thresholds,
                        'dialog_patterns': self.dialog_patterns
                    }, f)
                self._changes_since_save = 0
            except Exception as e:
                print(f"Error al guardar meta-cognición: {e}")

    def evaluate_response(self, user_message, ai_response, previous_messages=None):
        if not previous_messages:
            previous_messages = []
        scores = {
            'relevance': 0.0,
            'coherence': 0.0,
            'helpfulness': 0.0,
            'depth': 0.0,
            'engagement': 0.0
        }
        try:
            user_keywords = self._extract_keywords(user_message)
            response_keywords = self._extract_keywords(ai_response)
            semantic_overlap = len(set(user_keywords) & set(response_keywords)) / max(1, len(set(user_keywords)))
            scores['relevance'] = min(1.0, semantic_overlap * 1.5)
            contradiction_detected = self._detect_contradictions(ai_response)
            scores['coherence'] = 0.9 - (0.5 if contradiction_detected else 0.0)
            word_count = len(ai_response.split())
            unique_words = len(set(ai_response.lower().split()))
            complexity_ratio = unique_words / max(1, word_count)
            long_words = len([w for w in ai_response.lower().split() if len(w) > 7])
            long_word_ratio = long_words / max(1, word_count)
            scores['depth'] = min(1.0, (complexity_ratio * 0.5) + (long_word_ratio * 2.0) + (word_count / 200 * 0.3))
            engagement_score = 0.5
            if '?' in ai_response:
                engagement_score += 0.2
            if any(marker in ai_response.lower() for marker in ['por ejemplo', 'ejemplo', 'como cuando', 'imagina']):
                engagement_score += 0.15
            if any(marker in ai_response.lower() for marker in ['es como', 'similar a', 'equivale a', 'se asemeja']):
                engagement_score += 0.15
            scores['engagement'] = min(1.0, engagement_score)
            if any(marker in user_message.lower() for marker in ['cómo', 'ayuda', 'explicar', 'entender']):
                scores['helpfulness'] = 0.7 + (scores['relevance'] * 0.3) if scores['relevance'] > 0.6 and scores['depth'] > 0.4 else 0.4 + (scores['relevance'] * 0.3)
            else:
                scores['helpfulness'] = 0.5 + (scores['relevance'] * 0.2) + (scores['engagement'] * 0.3)
            self._update_dialog_patterns(user_message, ai_response, previous_messages)
            for metric, value in scores.items():
                self.evaluation_metrics[metric].append(value)
                if len(self.evaluation_metrics[metric]) > 20:
                    self.evaluation_metrics[metric] = self.evaluation_metrics[metric][-20:]
            self.save_metacognition()
        except Exception as e:
            print(f"Error al evaluar respuesta: {e}")
        return scores

    def generate_self_improvements(self, scores, user_message, ai_response):
        improvements = {
            'adjusted_temperature': None,
            'adjusted_max_tokens': None,
            'suggested_prompt_additions': [],
            'focus_areas': []
        }
        avg_metrics = {k: sum(v[-5:]) / max(1, len(v[-5:])) for k, v in self.evaluation_metrics.items() if v}
        if avg_metrics.get('relevance', 0.7) < self.improvement_thresholds['relevance_threshold']:
            improvements['adjusted_temperature'] = 0.65
            improvements['focus_areas'].append('aumentar_relevancia')
        elif avg_metrics.get('coherence', 0.8) < self.improvement_thresholds['coherence_threshold']:
            improvements['adjusted_temperature'] = 0.6
            improvements['focus_areas'].append('mejorar_coherencia')
        if scores.get('depth', 0.5) < self.improvement_thresholds['elaboration_needed']:
            improvements['adjusted_max_tokens'] = 200
            improvements['focus_areas'].append('aumentar_profundidad')
        if scores.get('relevance', 0.7) < 0.6:
            improvements['suggested_prompt_additions'].append("Asegúrate de abordar directamente los puntos principales mencionados por el usuario.")
        if self.dialog_patterns['repetitive_responses'] > 2:
            improvements['suggested_prompt_additions'].append("Evita repetir frases o estructuras de respuestas anteriores.")
            self.dialog_patterns['repetitive_responses'] -= 1
        if self.dialog_patterns['topic_misalignments'] > 2:
            improvements['suggested_prompt_additions'].append("Mantente enfocado en el tema principal de la consulta.")
            self.dialog_patterns['topic_misalignments'] -= 1
        misunderstanding = self._detect_misunderstanding(user_message, ai_response)
        if misunderstanding:
            self.evaluation_metrics['misunderstandings'].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'pattern': misunderstanding
            })
            improvements['suggested_prompt_additions'].append(f"Posible malentendido detectado ({misunderstanding}). Verifica tu comprensión.")
        self.save_metacognition()
        return improvements

    def _extract_keywords(self, text):
        try:
            text = text.lower()
            words = text.split()
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'a', 'en', 'que', 'es', 'por', 'para', 'con', 'su', 'sus'}
            return [word for word in words if len(word) > 3 and word not in stopwords]
        except Exception as e:
            print(f"Error al extraer keywords: {e}")
            return []

    def _detect_contradictions(self, text):
        try:
            contradiction_patterns = [
                (r'no .{1,20} pero .{1,20} sí', 'afirmación-negación'),
                (r'siempre .{1,30} nunca', 'absolutos contradictorios'),
                (r'es .{1,20} no es', 'afirmación-negación directa')
            ]
            for pattern, _ in contradiction_patterns:
                if re.search(pattern, text.lower()):
                    return True
            return False
        except Exception as e:
            print(f"Error al detectar contradicciones: {e}")
            return False

    def _update_dialog_patterns(self, user_message, ai_response, previous_messages):
        if len(previous_messages) >= 2:
            last_response = previous_messages[-1]
            if self._find_repeated_phrases(last_response, ai_response) >= 3:
                self.dialog_patterns['repetitive_responses'] += 1
        if '?' in user_message and '?' not in ai_response:
            question_words = ['cómo', 'qué', 'cuándo', 'dónde', 'por qué', 'cuál', 'quién']
            if any(word in user_message.lower() for word in question_words):
                self.dialog_patterns['question_avoidance'] += 1
        if previous_messages and len(previous_messages) >= 2:
            prev_user_message = previous_messages[-2]
            user_topics = set(self._extract_keywords(user_message))
            prev_topics = set(self._extract_keywords(prev_user_message))
            if len(user_topics & prev_topics) > 2:
                response_topics = set(self._extract_keywords(ai_response))
                if len(user_topics & response_topics) < 2:
                    self.dialog_patterns['topic_misalignments'] += 1

    def _find_repeated_phrases(self, text1, text2, min_phrase_length=5):
        words1 = text1.lower().split()
        words2 = set(text2.lower().split())
        return sum(1 for i in range(len(words1) - min_phrase_length + 1) if ' '.join(words1[i:i+min_phrase_length]) in ' '.join(words2))

    def _detect_misunderstanding(self, user_message, ai_response):
        misunderstanding_patterns = [
            (r'\?.*\bcómo\b|\bqué\b|\bcuál\b|\bcuándo\b', r'', 'respuesta_general_a_pregunta_específica'),
            (r'\bno\b.*\b(quiero|me gusta|deseo)\b', r'\bperfecto\b|\bexcelente\b|\bgenial\b', 'ignorar_negativa'),
            (r'\bno entiendo\b|\bclarifica\b|\bexplica mejor\b', r'\bcomo dijiste\b|\bcomo sabes\b', 'asumir_entendimiento')
        ]
        for user_pattern, response_pattern, pattern_name in misunderstanding_patterns:
            if re.search(user_pattern, user_message.lower()):
                if not response_pattern or re.search(response_pattern, ai_response.lower()):
                    return pattern_name
        return None

    def update_thresholds(self, user_feedback=None):
        avg_metrics = {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        if avg_metrics.get('relevance', 0):
            self.improvement_thresholds['relevance_threshold'] = min(0.85, max(0.5, self.improvement_thresholds['relevance_threshold'] + (0.02 if avg_metrics['relevance'] > 0.8 else -0.02 if avg_metrics['relevance'] < 0.6 else 0)))
        if avg_metrics.get('coherence', 0):
            self.improvement_thresholds['coherence_threshold'] = min(0.9, max(0.6, self.improvement_thresholds['coherence_threshold'] + (0.01 if avg_metrics['coherence'] > 0.85 else -0.01 if avg_metrics['coherence'] < 0.7 else 0)))
        if avg_metrics.get('depth', 0):
            self.improvement_thresholds['elaboration_needed'] = min(0.75, max(0.45, self.improvement_thresholds['elaboration_needed'] + (0.02 if avg_metrics['depth'] > 0.7 else -0.02 if avg_metrics['depth'] < 0.5 else 0)))
        if user_feedback:
            feedback_value = user_feedback.get('value', 0)
            feedback_type = user_feedback.get('type', 'general')
            if feedback_type in ['relevance', 'coherence', 'depth'] and feedback_value < 0.5:
                key = {'relevance': 'relevance_threshold', 'coherence': 'coherence_threshold', 'depth': 'elaboration_needed'}[feedback_type]
                self.improvement_thresholds[key] = max(0.4, self.improvement_thresholds[key] - 0.05)
        self.save_metacognition()

    def get_performance_summary(self):
        avg_metrics = {k: sum(v[-10:]) / max(1, len(v[-10:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        return {
            'average_metrics': avg_metrics,
            'current_thresholds': self.improvement_thresholds,
            'dialog_patterns': self.dialog_patterns,
            'recent_misunderstandings': self.evaluation_metrics.get('misunderstandings', [])[-5:]
        }

    def reset_dialog_patterns(self):
        for key in self.dialog_patterns:
            self.dialog_patterns[key] = 0
        self.save_metacognition()

    def adjust_metrics_weights(self, user_preferences=None):
        weights = {'relevance': 0.25, 'coherence': 0.20, 'helpfulness': 0.30, 'depth': 0.15, 'engagement': 0.10}
        if user_preferences:
            for metric, pref in user_preferences.items():
                if metric in weights:
                    weights[metric] = max(0.05, min(0.5, pref / 10 * 0.5))
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        if self.dialog_patterns['excessive_complexity'] > 3:
            weights['depth'] = max(0.05, weights['depth'] - 0.05)
            weights['coherence'] = min(0.35, weights['coherence'] + 0.05)
            self.dialog_patterns['excessive_complexity'] -= 1
        if self.dialog_patterns['excessive_simplicity'] > 3:
            weights['depth'] = min(0.35, weights['depth'] + 0.05)
            self.dialog_patterns['excessive_simplicity'] -= 1
        return weights

    def analyze_user_preferences(self, conversation_history):
        if not conversation_history:
            return {}
        preferences = {'prefers_detailed': 0, 'prefers_examples': 0, 'prefers_technical': 0, 'prefers_informal': 0}
        indicators = {
            'prefers_detailed': ['detalle', 'explica', 'elabora', 'profundiza', 'más información'],
            'prefers_examples': ['ejemplo', 'muestra', 'ilustra', 'caso'],
            'prefers_technical': ['técnico', 'específico', 'preciso', 'académico'],
            'prefers_informal': ['simple', 'sencillo', 'fácil', 'informal', 'coloquial']
        }
        for message in conversation_history:
            if 'Usuario:' in message:
                user_text = message.split('Usuario: ')[1].lower()
                for pref, inds in indicators.items():
                    if any(ind in user_text for ind in inds):
                        preferences[pref] += 1
                if any(term in user_text for term in ['breve', 'corto', 'resumen', 'sintetiza']):
                    preferences['prefers_detailed'] -= 1
                if any(term in user_text for term in ['detallado', 'extenso', 'largo', 'completo']):
                    preferences['prefers_detailed'] += 1
        return preferences

    def export_learning(self):
        avg_metrics = {k: sum(v[-20:]) / max(1, len(v[-20:])) for k, v in self.evaluation_metrics.items() if v and isinstance(v[0], (int, float))}
        return {
            'user_id': self.user_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'performance_metrics': avg_metrics,
            'problematic_patterns': {k: v for k, v in self.dialog_patterns.items() if v > 2},
            'misunderstandings': self.evaluation_metrics.get('misunderstandings', [])[-10:],
            'current_thresholds': self.improvement_thresholds
        }

    def aggregate_feedback(self, explicit_feedback):
        if not explicit_feedback:
            return
        feedback_map = {'muy_util': 1.0, 'util': 0.8, 'neutral': 0.5, 'poco_util': 0.3, 'no_util': 0.1}
        feedback_type = explicit_feedback.get('type', 'general')
        feedback_value = feedback_map.get(explicit_feedback.get('value', 'neutral'), 0.5)
        metrics = {'relevance': 'relevance', 'helpfulness': 'helpfulness', 'clarity': 'coherence', 'depth': 'depth', 'engagement': 'engagement'}
        if feedback_type in metrics:
            self.evaluation_metrics[metrics[feedback_type]].append(feedback_value)
        else:
            for metric in ['relevance', 'coherence', 'helpfulness', 'engagement']:
                self.evaluation_metrics[metric].append(feedback_value)
        for metric in self.evaluation_metrics:
            if isinstance(self.evaluation_metrics[metric], list) and len(self.evaluation_metrics[metric]) > 30:
                self.evaluation_metrics[metric] = self.evaluation_metrics[metric][-30:]
        self.save_metacognition()

    def adaptive_learning(self, conversation_history, recent_evaluations):
        if not conversation_history or not recent_evaluations:
            return {}
        temporal_patterns = {}
        metrics = ['relevance', 'coherence', 'helpfulness', 'depth', 'engagement']
        for metric in metrics:
            if len(recent_evaluations) >= 5 and all(metric in eval_data for eval_data in recent_evaluations):
                values = [eval_data[metric] for eval_data in recent_evaluations[-5:]]
                diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_diff = sum(diffs) / len(diffs)
                temporal_patterns[f'{metric}_trend'] = avg_diff
        return temporal_patterns

# Rutas de la aplicación
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reflexion/<int:id>')
def mostrar_reflexion(id):
    reflexion = Reflexion.query.get_or_404(id)
    notification_body = {
        "contents": {"en": f"Nueva reflexión visitada: {reflexion.titulo}"},
        "included_segments": ["Subscribed Users"]
    }
    try:
        response = onesignal_client.notification_create(notification_body)
        print(f"Notificación enviada: {response.body}")
    except Exception as e:
        print(f"Error enviando notificación: {e}")
    return render_template('reflexion.html', reflexion=reflexion)

@app.route('/articulo-aleatorio')
def articulo_aleatorio():
    reflexion = random.choice(Reflexion.query.all())
    return redirect(url_for('mostrar_reflexion', id=reflexion.id))

@app.route('/galeria', defaults={'page': 1})
@app.route('/galeria/page/<int:page>')
@cache.cached(timeout=300, query_string=True)
def galeria(page):
    per_page = 20
    reflexiones = Reflexion.query.paginate(page=page, per_page=per_page, error_out=False)
    print(f"Galería - Página {page}: {len(reflexiones.items)} reflexiones enviadas de {reflexiones.total} totales")
    return render_template('galeria.html', reflexiones=reflexiones.items, pagination=reflexiones)

@app.route('/recursos')
def recursos():
    return render_template('recursos.html')

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        nombre = request.form['nombre']
        email = request.form['email']
        mensaje = request.form['mensaje']
        msg = Message(
            subject=f"Nuevo mensaje de {nombre}",
            recipients=[app.config['MAIL_USERNAME']],
            body=f"Nombre: {nombre}\nCorreo: {email}\nMensaje: {mensaje}"
        )
        mail.send(msg)
        flash('¡Mensaje enviado con éxito! Gracias por contactarnos.', 'success')
        return redirect(url_for('contacto'))
    return render_template('contacto.html')

@app.route('/buscar', methods=['GET'])
@cache.cached(timeout=60, query_string=True)
def buscar():
    query = request.args.get('q', '').strip().lower()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    if not query:
        return render_template('busqueda.html', resultados=[], query=query, page=page, per_page=per_page, total=0)
    resultados = Reflexion.query.filter(
        db.or_(Reflexion.titulo.ilike(f'%{query}%'), Reflexion.contenido.ilike(f'%{query}%'))
    ).all()
    total = len(resultados)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_resultados = resultados[start:end]
    return render_template('busqueda.html', resultados=paginated_resultados, query=query, page=page, per_page=per_page, total=total)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        next_page = request.args.get('next', '')
        print(f"Usuario ya autenticado, redirigiendo a: {next_page if next_page and is_safe_url(next_page) else url_for('home')}")
        if next_page and is_safe_url(next_page):
            return redirect(next_page)
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        next_page = request.form.get('next', request.args.get('next', ''))

        if user and user.check_password(password):
            if not user.is_active:
                flash('Tu cuenta aún no está activada. Por favor, revisa tu correo para activar tu cuenta.', 'error')
                return render_template('login.html', next=next_page, email=email)
            session.permanent = True
            login_user(user, remember=True)
            print(f"Inicio de sesión exitoso para {email}. Redirigiendo a: {next_page if next_page and is_safe_url(next_page) else url_for('home')}")
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))
        else:
            flash('Correo o contraseña incorrectos.', 'error')
            print(f"Inicio de sesión fallido para {email}")
            return render_template('login.html', next=next_page, email=email)

    next_page = request.args.get('next', '')
    email = request.args.get('email', '')
    print(f"Valor de next_page en GET: {next_page}")
    return render_template('login.html', next=next_page, email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        birth_date_str = request.form['birth_date']
        phone = request.form.get('phone', '')
        next_page = request.args.get('next', '')

        try:
            birth_date = datetime.datetime.strptime(birth_date_str, '%d/%m/%Y')
            birth_date_iso = birth_date.isoformat()
        except ValueError:
            flash('El formato de la fecha de nacimiento debe ser DD/MM/AAAA.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
        if password != confirm_password:
            flash('Las contraseñas no coinciden.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)
        if not re.match(password_pattern, password):
            flash('La contraseña debe tener al menos 8 caracteres, una mayúscula, una minúscula y un número.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        if User.query.filter_by(email=email).first():
            flash('El correo ya está registrado.', 'error')
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        user = User(
            email=email,
            subscription_date=datetime.datetime.now().isoformat(),
            preferred_categories='all',
            frequency='weekly',
            push_enabled=False,
            birth_date=birth_date_iso
        )
        user.set_password(password)
        user.is_active = False
        token = user.generate_activation_token()
        db.session.add(user)
        db.session.commit()

        activation_link = url_for('activate', token=token, next=next_page, _external=True)
        msg = Message(
            subject='Activa tu cuenta en Voy Consciente',
            recipients=[email],
            body=f'''
            Hola {name},

            Gracias por registrarte en Voy Consciente. Para activar tu cuenta, haz clic en el siguiente enlace:

            {activation_link}

            ¡Estás suscrito para recibir nuestras reflexiones semanales por correo!

            Si no te registraste, puedes ignorar este correo.

            ¡Esperamos verte pronto!
            El equipo de Voy Consciente
            '''
        )
        try:
            mail.send(msg)
            flash('Se ha enviado un correo para activar tu cuenta. Revisa tu bandeja de entrada (y spam).', 'success')
        except Exception as e:
            flash(f'Error al enviar el correo de activación: {str(e)}', 'error')
            db.session.delete(user)
            db.session.commit()
            return render_template('register.html', next=next_page, name=name, email=email, birth_date=birth_date_str, phone=phone, password=password, confirm_password=confirm_password)

        return redirect(url_for('login', next=next_page))

    next_page = request.args.get('next', '')
    return render_template('register.html', next=next_page)

@app.route('/activate/<token>')
def activate(token):
    try:
        email = serializer.loads(token, salt='activation-salt', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('El enlace de activación es inválido o el usuario no existe.', 'error')
            return redirect(url_for('register', next=request.args.get('next')))
        if user.is_active:
            flash('Tu cuenta ya está activada.', 'success')
            next_page = request.args.get('next', '')
            if next_page and is_safe_url(next_page):
                return redirect(next_page)
            return redirect(url_for('home'))
        user.is_active = True
        user.activation_token = None
        db.session.commit()
        flash('¡Tu cuenta ha sido activada! Por favor, inicia sesión.', 'success')
        next_page = request.args.get('next', '')
        return redirect(url_for('login', next=next_page))
    except (SignatureExpired, BadSignature):
        flash('El enlace de activación ha expirado o es inválido.', 'error')
        return redirect(url_for('register', next=request.args.get('next')))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = serializer.dumps(email, salt='password-reset-salt')
            reset_link = url_for('reset_password', token=token, _external=True)
            reset_record = PasswordReset(email=email, token=token)
            db.session.add(reset_record)
            db.session.commit()
            msg = Message('Restablecer tu contraseña - Voy Consciente', recipients=[email])
            msg.body = f'Haz clic en el siguiente enlace para restablecer tu contraseña: {reset_link}\nEste enlace expira en 1 hora.'
            msg.html = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #fff; border-radius: 10px; padding: 20px;">
                        <h1 style="color: #4CAF50; text-align: center;">Restablecer Contraseña</h1>
                        <p>Haz clic en el enlace para restablecer tu contraseña:</p>
                        <p style="text-align: center;"><a href="{reset_link}" style="color: #4CAF50;">Restablecer Contraseña</a></p>
                        <p>Este enlace expira en 1 hora.</p>
                    </div>
                </body>
            </html>
            """
            try:
                mail.send(msg)
                flash('Se ha enviado un enlace de restablecimiento a tu correo.', 'success')
            except Exception as e:
                flash(f'Error al enviar el correo: {str(e)}', 'error')
        else:
            flash('No se encontró un usuario con ese correo.', 'error')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Token inválido o usuario no encontrado.', 'error')
            return redirect(url_for('login'))
        if request.method == 'POST':
            new_password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            if new_password != confirm_password:
                flash('Las contraseñas no coinciden. Por favor, intenta de nuevo.', 'error')
                return render_template('reset_password.html', token=token)
            password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
            if not re.match(password_pattern, new_password):
                flash('La contraseña debe tener al menos 8 caracteres, una mayúscula, una minúscula y un número.', 'error')
                return render_template('reset_password.html', token=token)
            user.set_password(new_password)
            PasswordReset.query.filter_by(email=email).delete()
            db.session.commit()
            flash('Tu contraseña ha sido restablecida. Inicia sesión con tu nueva contraseña.', 'success')
            return redirect(url_for('login'))
        return render_template('reset_password.html', token=token)
    except Exception as e:
        flash('El enlace de restablecimiento ha expirado o es inválido.', 'error')
        return redirect(url_for('forgot_password'))

@app.route('/toggle_favorite/<int:reflexion_id>', methods=['POST'])
@login_required
def toggle_favorite(reflexion_id):
    reflexion = Reflexion.query.get_or_404(reflexion_id)
    if reflexion in current_user.favorite_reflexiones:
        current_user.favorite_reflexiones.remove(reflexion)
        flash('Reflexión eliminada de favoritos.', 'success')
    else:
        current_user.favorite_reflexiones.append(reflexion)
        flash('Reflexión añadida a favoritos.', 'success')
    db.session.commit()
    return redirect(request.referrer or url_for('mostrar_reflexion', id=reflexion_id))

@app.route('/favoritos')
@login_required
def favoritos():
    return render_template('favoritos.html', favoritos=current_user.favorite_reflexiones)

@app.route('/download_pdf/<int:reflexion_id>')
def download_pdf(reflexion_id):
    reflexion = Reflexion.query.get_or_404(reflexion_id)
    html = render_template('reflexion_pdf.html',
                          reflexion=reflexion,
                          año_actual=datetime.datetime.now().year)
    pdf_file = io.BytesIO()
    HTML(string=html).write_pdf(pdf_file)
    pdf_file.seek(0)
    response = make_response(pdf_file.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename="{reflexion.titulo}.pdf"'
    return response

@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia():
    # Inicialización de variables de sesión
    session_vars = [
        'chat_history', 'user_topics', 'user_preferences',
        'conversation_context', 'session_count', 'message_count',
        'user_specified_hour', 'professional_mode', 'conversation_depth',
        'last_topics', 'emotional_state'
    ]
    for var in session_vars:
        if var not in session:
            if var == 'chat_history':
                session[var] = []
            elif var == 'user_topics':
                session[var] = {}
            elif var == 'user_preferences':
                session[var] = {}
            elif var == 'conversation_context':
                session[var] = {}
            elif var == 'conversation_depth':
                session[var] = 0
            elif var == 'last_topics':
                session[var] = []
            elif var == 'emotional_state':
                session[var] = 'neutral'
            else:
                session[var] = 0 if var.endswith('_count') else None

    if request.method == 'GET':
        session['session_count'] += 1
        session['message_count'] = 0
        session['conversation_depth'] = 0

    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin

    user_memory = UserMemoryManager(current_user.id)
    personality_engine = PersonalityEngine(current_user.id)

    FREE_INTERACTION_LIMIT = 5
    interaction_count = Interaction.get_interaction_count(identifier, is_authenticated=True)
    remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - interaction_count)

    class DespedidaDetector:
        def __init__(self, umbral_confianza: float = 0.70):
            self.umbral_confianza = umbral_confianza
            self.patrones_despedida = {
                "es": [
                    r"\badi[óo]s\b", r"\bchao\b", r"\bhasta luego\b", r"\bhasta pronto\b",
                    r"\bnos vemos\b", r"\bhasta ma[nñ]ana\b", r"\bme voy\b", r"\bmuchas gracias\b",
                    r"\bgracias por todo\b", r"\bme despido\b", r"\bbuenas noches\b",
                    r"\bhasta la pr[óo]xima\b", r"\bfin\b", r"\bterminar\b", r"\bterminamos\b",
                    r"\beso es todo\b", r"\bya est[áa]\b", r"\blisto\b", r"\bes todo\b"
                ],
                "en": [
                    r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\blater\b", r"\bcya\b",
                    r"\bfarewell\b", r"\bhave a good\b", r"\btake care\b", r"\bending\b",
                    r"\bthank you for\b", r"\bthat's all\b", r"\bi'm done\b", r"\bthat will be all\b",
                    r"\bgood night\b", r"\bsigning off\b"
                ]
            }
            self.respuestas_despedida = {
                "es": [
                    "¡Hasta pronto! Ha sido un placer charlar contigo.",
                    "Gracias por conversar. ¡Que tengas un lindo día!",
                    "Entendido, me despido aquí. ¡Siempre un gusto ayudarte!",
                    "¡Perfecto! Nos vemos cuando quieras. ¡Adiós!",
                    "¡Hasta la próxima! Fue genial hablar contigo."
                ],
                "en": [
                    "See you later! It’s been great chatting with you.",
                    "Thanks for the talk. Have a wonderful day!",
                    "Got it, I’ll say goodbye here. Always a pleasure to assist!",
                    "Perfect! Catch you next time. Bye!",
                    "Until next time! Loved our conversation."
                ]
            }

        def detectar_idioma(self, texto: str) -> str:
            palabras_es = ["gracias", "hola", "adiós", "por", "favor", "buenos", "días"]
            palabras_en = ["thanks", "hello", "goodbye", "please", "good", "morning", "bye"]
            count_es = sum(1 for palabra in palabras_es if palabra in texto.lower())
            count_en = sum(1 for palabra in palabras_en if palabra in texto.lower())
            return "es" if count_es >= count_en else "en"

        def calcular_features(self, texto: str, historial: list, user_hour: float) -> dict:
            texto = texto.lower()
            features = {}
            idioma = self.detectar_idioma(texto)
            patrones = self.patrones_despedida.get(idioma, self.patrones_despedida["en"])
            match_count = sum(1 for patron in patrones if re.search(patron, texto))
            features["patron_explicito"] = min(match_count / 2, 1.0)
            palabras = len(texto.split())
            features["brevedad"] = 1.0 if palabras <= 5 else (10 / palabras if palabras < 10 else 0.1)
            agradecimientos = ["gracias", "thank", "thanks", "agradec"]
            features["agradecimiento"] = any(a in texto for a in agradecimientos) * 0.7
            finalidad_palabras = ["eso es todo", "that's all", "por ahora", "listo", "done"]
            features["finalidad"] = any(f in texto for f in finalidad_palabras) * 0.9
            features["hora_nocturna"] = 0.3 if 19 <= user_hour or user_hour < 6 else 0.0
            features["longitud_conversacion"] = min(len(historial) / 20, 0.5) if historial else 0.0
            return features

        def es_despedida(self, texto: str, historial: list, user_hour: float) -> tuple[bool, float, dict]:
            try:
                if not texto or len(texto.strip()) == 0:
                    return False, 0.0, {"error": "Texto vacío"}
                features = self.calcular_features(texto, historial, user_hour)
                pesos = {
                    "patron_explicito": 0.6,
                    "brevedad": 0.1,
                    "agradecimiento": 0.1,
                    "finalidad": 0.15,
                    "hora_nocturna": 0.05,
                    "longitud_conversacion": 0.1
                }
                puntuacion = sum(features.get(feat, 0) * pesos.get(feat, 0) for feat in features)
                confianza = min(puntuacion, 1.0)
                es_despedida = confianza >= self.umbral_confianza
                return es_despedida, confianza, {"features": features}
            except Exception as e:
                return False, 0.0, {"error": str(e)}

        def generar_despedida(self, texto_usuario: str, personality: "PersonalityEngine", user_hour: float) -> str:
            idioma = self.detectar_idioma(texto_usuario)
            respuestas = self.respuestas_despedida.get(idioma, self.respuestas_despedida["en"])
            if personality.dimensions["warmth"] > 0.7:
                respuestas = [r for r in respuestas if "placer" in r or "great" in r or "gusto" in r]
            if personality.dimensions["formality"] > 0.7:
                respuestas = [r for r in respuestas if "Entendido" in r or "Got it" in r]
            if 19 <= user_hour or user_hour < 6:
                respuestas.append("¡Buenas noches! Hasta pronto." if idioma == "es" else "Good night! See you soon.")
            return random.choice(respuestas) if respuestas else respuestas[0]

    def detect_conversation_closing(message):
        closing_phrases = [
            'adios', 'hasta luego', 'nos vemos', 'chau', 'bye',
            'hasta pronto', 'me voy', 'terminamos', 'eso es todo',
            'gracias por tu ayuda', 'muchas gracias', 'listo',
            'es todo', 'suficiente', 'terminamos por hoy'
        ]
        message_lower = message.lower()
        if any(phrase in message_lower for phrase in closing_phrases):
            return True
        if len(message_lower.split()) <= 3 and any(word in message_lower for word in ['gracias', 'thanks', 'ok', 'bien']):
            return True
        if len(message_lower.split()) == 1 and message_lower in ['ok', 'si', 'sí', 'no', 'bien']:
            return True
        return False

    def get_relevant_reflexiones(user_message=None, limit=3):
        try:
            if not user_message:
                return Reflexion.query.order_by(db.func.random()).limit(limit).all()
            palabras_clave = [palabra for palabra in user_message.lower().split()
                             if len(palabra) > 3 and palabra not in ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o']]
            if not palabras_clave:
                return Reflexion.query.order_by(db.func.random()).limit(limit).all()
            relevantes = []
            for reflexion in Reflexion.query.all():
                score = sum(1 for palabra in palabras_clave
                           if palabra in reflexion.titulo.lower() or palabra in reflexion.contenido.lower())
                if score > 0:
                    relevantes.append((reflexion, score))
            relevantes.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in relevantes[:limit]]
        except Exception as e:
            print(f"Error al buscar reflexiones relevantes: {e}")
            return []

    def get_relevant_books(user_message=None, limit=2):
        try:
            if not user_message:
                return Book.query.order_by(db.func.random()).limit(limit).all()
            palabras_clave = [palabra for palabra in user_message.lower().split()
                             if len(palabra) > 3 and palabra not in ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o']]
            if not palabras_clave:
                return Book.query.order_by(db.func.random()).limit(limit).all()
            relevantes = []
            for book in Book.query.all():
                score = sum(1 for palabra in palabras_clave
                           if palabra in book.title.lower() or palabra in book.content.lower())
                if score > 0:
                    relevantes.append((book, score))
            relevantes.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in relevantes[:limit]]
        except Exception as e:
            print(f"Error al buscar libros relevantes: {e}")
            return []

    def analyze_sentiment(text):
        positive_words = ['feliz', 'alegre', 'gracias', 'genial', 'bueno', 'excelente']
        negative_words = ['triste', 'enojado', 'frustrado', 'malo', 'terrible', 'problema']
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        return 'neutral'

    def extract_context(request):
        return {
            'url': request.url,
            'method': request.method,
            'user_agent': request.headers.get('User-Agent'),
            'timestamp': datetime.datetime.now().isoformat()
        }

    def analyze_emotional_state(text: str) -> dict:
        return {
            'valence': 0.5,
            'arousal': 0.5,
            'dominance': 0.5
        }

    def extract_topics(text):
        common_topics = {
            'trabajo': ['trabajo', 'empleo', 'profesión', 'carrera'],
            'salud': ['salud', 'enfermedad', 'médico', 'ejercicio', 'bienestar'],
            'tecnología': ['tecnología', 'computadora', 'app', 'software', 'programación'],
            'filosofía': ['filosofía', 'existencia', 'significado', 'propósito'],
        }
        found_topics = []
        text_lower = text.lower()
        for topic, keywords in common_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        return found_topics

    if request.method == 'POST':
        if not (is_premium or is_admin) and interaction_count >= FREE_INTERACTION_LIMIT:
            return jsonify({
                'response': 'Has alcanzado tu límite diario de conversación.\n¿Te gustaría probar la versión Premium por $5/mes? ¡Tu curiosidad y crecimiento no tienen por qué esperar!',
                'remaining_interactions': remaining_interactions,
                'limit_reached': True,
                'subscribe_url': url_for('subscribe_info', _external=True)
            })

        message = request.json.get('message', '').strip()
        message_lower = message.lower()

        meta_cognition = MetaCognitionSystem(current_user.id)
        personality_engine = PersonalityEngine(current_user.id)
        context_learning = DeepContextLearning()

        try:
            hour_match = re.search(r'son las (\d{1,2}):(\d{2})(?:\s*hs)?', message_lower)
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2))
                session['user_specified_hour'] = hour + minute / 60
        except Exception as e:
            print(f"Error al detectar la hora del usuario: {e}")

        if "profesional" in message_lower or "formal" in message_lower:
            if "no " in message_lower[:message_lower.find("profesional") + 3] or "no " in message_lower[:message_lower.find("formal") + 3]:
                session['professional_mode'] = False
            else:
                session['professional_mode'] = True

        session['emotional_state'] = analyze_sentiment(message)
        current_topics = extract_topics(message)
        session['last_topics'] = current_topics

        session['chat_history'].append(f"Usuario: {message}")
        session['message_count'] += 1
        session['conversation_depth'] += 1

        conversation_depth = min(session['conversation_depth'], 10)

        previous_topics = session.get('last_topics', [])
        personality_engine.update_from_interaction(
            user_message=message,
            user_sentiment=session['emotional_state'],
            previous_topic=previous_topics[0] if previous_topics else None,
            current_topic=current_topics[0] if current_topics else None
        )

        personality_instructions = personality_engine.generate_instruction_set(
            conversation_depth=session['conversation_depth'],
            context={
                'hora': session['user_specified_hour'] if session['user_specified_hour'] is not None else datetime.datetime.now(tz=datetime.timezone(timedelta(hours=-3))).hour,
                'temas': session['last_topics']
            }
        )

        relevant_reflexiones = get_relevant_reflexiones(message)
        relevant_books = get_relevant_books(message)

        reflexiones_summary = "\n".join([
            f"Título: {reflexion.titulo}\nContenido: {reflexion.contenido[:300]}...\nRelevancia: Alta"
            for reflexion in relevant_reflexiones
        ]) if relevant_reflexiones else "No hay reflexiones relevantes para este tema."

        books_summary = "\n".join([
            f"Título: {book.title}\nContenido: {book.content[:300]}...\nRelevancia: Alta"
            for book in relevant_books
        ]) if relevant_books else "No hay libros relevantes para este tema."

        try:
            current_time = datetime.datetime.now(tz=datetime.timezone(timedelta(hours=-3)))
            server_hour = current_time.hour + current_time.minute / 60
            current_day = current_time.strftime("%A")
            user_hour = session['user_specified_hour'] if session['user_specified_hour'] is not None else server_hour
        except Exception as e:
            print(f"Error al calcular la hora: {e}")
            server_hour = 12
            current_day = "Unknown"
            user_hour = 12

            recent_messages = session['chat_history'][-min(10, len(session['chat_history'])):]
        recent_history = "\n".join(recent_messages)

        relevant_memories = user_memory.find_relevant_memories(message, limit=3)
        memory_summary = "\n".join([
            f"Fecha: {mem['timestamp'].strftime('%d/%m/%Y')}\n"
            f"Contexto: {', '.join(f'{k}: {v}' for k, v in mem['context'].items())}\n"
            f"Mensaje: {mem['text'][:150]}...\n"
            for mem in relevant_memories
        ]) if relevant_memories else "No hay recuerdos previos relevantes."

        context = {
            'hora': user_hour,
            'profundidad': conversation_depth,
            'emoción': session['emotional_state'],
            'temas': session['last_topics']
        }
        user_memory.add_interaction(message, context=context, emotion=session['emotional_state'])

        # Detección de despedida
        despedida_detector = DespedidaDetector(umbral_confianza=0.70)
        es_despedida, confianza_despedida, detalles_despedida = despedida_detector.es_despedida(
            message, session['chat_history'], user_hour
        )

        if es_despedida:
            despedida_response = despedida_detector.generar_despedida(message, personality_engine, user_hour)
            session['chat_history'].append(f"IA: {despedida_response}")
            Interaction.increment_interaction(identifier, is_authenticated=True)
            return jsonify({
                'response': despedida_response,
                'remaining_interactions': remaining_interactions - 1,
                'limit_reached': False,
                'is_goodbye': True
            })

        # Construcción del prompt para Grok
        grok_prompt = f"""
        Eres Grok, creado por xAI. Tu propósito es asistir al usuario en una conversación reflexiva y consciente.
        Aquí tienes el contexto para tu respuesta:

        - Historial reciente de la conversación:
        {recent_history}

        - Recuerdos relevantes del usuario:
        {memory_summary}

        - Resumen de reflexiones relacionadas:
        {reflexiones_summary}

        - Resumen de libros relacionados:
        {books_summary}

        - Día actual: {current_day}
        - Hora del usuario: {user_hour:.2f}
        - Estado emocional detectado: {session['emotional_state']}
        - Temas actuales: {', '.join(current_topics) if current_topics else 'Ninguno identificado'}
        - Profundidad de la conversación: {conversation_depth}

        {personality_instructions['instruction_text']}

        Mensaje del usuario: {message}
        Responde de manera natural y adaptada al contexto proporcionado.
        """

        # Configuración de parámetros para la API de Grok
        grok_params = {
            'max_tokens': personality_instructions['parameters']['max_tokens'],
            'temperature': personality_instructions['parameters']['temperature'],
            'top_p': 0.9
        }

        # Llamada a la API de Grok (simulada aquí; reemplazar con la implementación real)
        try:
            headers = {
                'Authorization': f'Bearer {grok_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'grok',  # Ajustar según el modelo específico de xAI
                'prompt': grok_prompt,
                **grok_params
            }
            response = requests.post(
                'https://api.xai.com/v1/chat/completions',  # URL ficticia; usar la real de xAI
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            grok_response = response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error al consultar Grok API: {e}")
            grok_response = "Lo siento, tuve un problema al procesar tu mensaje. ¿Podrías intentarlo de nuevo?"

        # Evaluación meta-cognitiva
        meta_cognition.evaluate_response(message, grok_response, recent_messages)
        improvements = meta_cognition.generate_self_improvements(
            meta_cognition.evaluate_response(message, grok_response),
            message,
            grok_response
        )

        # Actualización del historial y memoria
        session['chat_history'].append(f"IA: {grok_response}")
        context_learning.learn_from_interaction(message, grok_response, context)
        Interaction.increment_interaction(identifier, is_authenticated=True)

        # Preparación de la respuesta al cliente
        response_data = {
            'response': grok_response,
            'remaining_interactions': remaining_interactions - 1,
            'limit_reached': False,
            'is_goodbye': False
        }

        # Agregar reflexiones y libros relevantes si existen
        if relevant_reflexiones:
            response_data['reflexiones'] = [{
                'id': r.id,
                'titulo': r.titulo,
                'contenido': r.contenido[:300] + "..."
            } for r in relevant_reflexiones]
        if relevant_books:
            response_data['books'] = [{
                'id': b.id,
                'title': b.title,
                'content': b.content[:300] + "..."
            } for b in relevant_books]

        return jsonify(response_data)

    # Método GET: Renderizar la interfaz inicial
    memory_summary = user_memory.get_memory_summary()
    initial_greeting = f"""
    ¡Hola {current_user.email.split('@')[0]}! Bienvenido(a) de vuelta. 
    Veo que hemos hablado {memory_summary['interaction_count']} veces antes, 
    la última hace {memory_summary['days_since_last']} días. 
    ¿En qué puedo ayudarte hoy?
    """ if memory_summary['interaction_count'] > 0 else "¡Hola! Soy Grok, creado por xAI. Estoy aquí para ayudarte a reflexionar y explorar. ¿En qué puedo ayudarte hoy?"

    return render_template(
        'consciencia.html',
        initial_message=initial_greeting,
        remaining_interactions=remaining_interactions,
        is_premium=is_premium,
        is_admin=is_admin
    )

# Rutas adicionales (ejemplo de suscripción, logout, etc.)
@app.route('/subscribe_info')
@login_required
def subscribe_info():
    return render_template('subscribe_info.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Has cerrado sesión exitosamente.', 'success')
    return redirect(url_for('home'))

# Configuración del scheduler (por ejemplo, limpieza de tokens de restablecimiento)
def cleanup_expired_resets():
    with app.app_context():
        expiration_time = datetime.datetime.utcnow() - timedelta(hours=1)
        PasswordReset.query.filter(PasswordReset.created_at < expiration_time).delete()
        db.session.commit()

scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_expired_resets, trigger='interval', hours=1)
scheduler.start()

# Ejecución de la aplicación
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)