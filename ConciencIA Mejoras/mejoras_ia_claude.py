@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia():
    # Inicializar variables de sesión con una estructura más rica y semántica
    session_vars = {
        'chat_history': [],              # Histórico completo
        'user_topics': {},               # Temas de interés con ponderación
        'user_preferences': {            # Preferencias estructuradas
            'communication_style': 'auto', # formal, casual, empático, analítico
            'response_depth': 'adaptive',  # conciso, moderado, profundo
            'interests': [],             # Intereses detectados
            'cognitive_preferences': [], # Visual, analítico, emocional, etc.
        },
        'conversation_context': {        # Contexto enriquecido
            'thread_id': str(uuid.uuid4()),
            'current_topic': None,       
            'topic_retention': 0,        # 0-10 cuánto persiste en un tema
            'cognitive_load': 0,         # 0-10 complejidad percibida
            'unanswered_questions': [],  # Preguntas pendientes de respuesta
        },
        'conversation_dynamics': {       # Métricas dinámicas
            'depth': 0,                  # Profundidad actual 
            'engagement': 0,             # Nivel de compromiso 0-10
            'initiative_ratio': 0.5,     # Balance entre usuario/IA
            'topic_continuity': 0,       # Continuidad temática
        },
        'emotional_context': {           # Análisis emocional complejo
            'current_state': 'neutral',
            'intensity': 0,              # 0-10
            'valence': 0,                # -5 (muy negativo) a +5 (muy positivo)
            'arousal': 0,                # 0-10 nivel de activación
            'trend': 'stable',           # increasing, decreasing, fluctuating
            'history': []                # Registro histórico de emociones
        },
        'session_metrics': {             # Métricas ampliadas
            'session_count': 0,
            'message_count': 0,
            'avg_response_time': 0,
            'avg_message_length': 0,
            'interaction_quality': 0,    # 0-10 calidad percibida
        },
        'temporal_context': {            # Contexto temporal
            'user_timezone': None,
            'user_specified_hour': None,
            'session_start_time': time.time(),
            'last_interaction_time': time.time(),
        }
    }
    
    # Inicializar o recuperar estructura de sesión
    for category, values in session_vars.items():
        if category not in session:
            session[category] = values
    
    # Actualizar métricas de sesión (solo en GET)
    if request.method == 'GET':
        session['session_metrics']['session_count'] += 1
        session['session_metrics']['message_count'] = 0
        session['conversation_dynamics']['depth'] = 0
        session['temporal_context']['session_start_time'] = time.time()
        session['conversation_context']['thread_id'] = str(uuid.uuid4())
        
        # Mantener preferencias y emociones para continuidad
        
    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin

    # Sistema adaptativo de límites de interacción
    FREE_INTERACTION_LIMIT = 5
    PREMIUM_DAILY_LIMIT = 50
    
    interaction_count = Interaction.get_interaction_count(identifier, is_authenticated=True)
    
    # Cálculo de límites con bonificaciones
    user_score = UserScore.get_score(identifier)
    bonus_interactions = min(5, user_score // 100)  # Bonificación por puntuación
    
    effective_limit = FREE_INTERACTION_LIMIT + bonus_interactions
    
    if is_premium:
        effective_limit = PREMIUM_DAILY_LIMIT
    elif is_admin:
        effective_limit = 999999
        
    remaining_interactions = max(0, effective_limit - interaction_count)

    # Sistemas de análisis semántico avanzado
    
    # 1. Análisis de sentimiento con modelo NLP contextual
    def analyze_sentiment(text, history=None):
        """
        Análisis de sentimiento avanzado usando NLP
        Retorna un diccionario completo con diferentes dimensiones emocionales
        """
        try:
            # Versión simplificada - en producción usar un modelo NLP completo
            positive_words = {
                'feliz': 0.8, 'alegre': 0.7, 'gracias': 0.6, 'genial': 0.8, 
                'bueno': 0.5, 'excelente': 0.9, 'contento': 0.7, 'satisfecho': 0.6,
                'optimista': 0.7, 'energía': 0.6, 'emocionado': 0.8
            }
            
            negative_words = {
                'triste': -0.7, 'enojado': -0.8, 'frustrado': -0.7, 'malo': -0.6, 
                'terrible': -0.9, 'problema': -0.5, 'preocupado': -0.6, 
                'ansioso': -0.7, 'molesto': -0.6, 'decepcionado': -0.7
            }
            
            # Palabras de alta intensidad emocional
            high_arousal = {
                'emocionado', 'furioso', 'extasiado', 'aterrorizado', 'eufórico',
                'ansioso', 'pánico', 'urgente', 'crisis'
            }
            
            text_lower = text.lower()
            words = re.findall(r'\w+', text_lower)
            
            # Análisis de valencia (positivo/negativo)
            valence_scores = []
            for word in words:
                if word in positive_words:
                    valence_scores.append(positive_words[word])
                elif word in negative_words:
                    valence_scores.append(negative_words[word])
            
            # Calcular valencia promedio
            valence = sum(valence_scores) / max(1, len(valence_scores)) if valence_scores else 0
            
            # Intensidad emocional (presencia de palabras intensas o signos de exclamación)
            intensity = min(10, (sum(1 for word in words if word in high_arousal) * 2) + 
                          (text.count('!') * 1.5) + 
                          (sum(1 for c in text if c.isupper()) / max(1, len(text)) * 10))
            
            # Determinar estado emocional
            if valence > 0.3:
                state = 'positive'
            elif valence < -0.3:
                state = 'negative'
            else:
                state = 'neutral'
                
            # Añadir matices
            if state == 'positive' and intensity > 7:
                state = 'excited'
            elif state == 'negative' and intensity > 7:
                state = 'distressed'
            
            # Nivel de activación (arousal)
            arousal = min(10, intensity + abs(valence) * 5)
            
            # Análisis de tendencia con historia previa
            trend = 'stable'
            if history and len(history) > 2:
                prev_valence = history[-1]['valence']
                if valence > prev_valence + 0.3:
                    trend = 'improving'
                elif valence < prev_valence - 0.3:
                    trend = 'deteriorating'
            
            return {
                'state': state,
                'valence': round(valence * 5, 1),  # Escala -5 a +5
                'intensity': round(intensity, 1),
                'arousal': round(arousal, 1),
                'trend': trend
            }
                
        except Exception as e:
            print(f"Error en análisis de sentimiento: {e}")
            return {'state': 'neutral', 'valence': 0, 'intensity': 0, 'arousal': 0, 'trend': 'stable'}

    # 2. Extracción avanzada de temas con categorización jerárquica
    def extract_topics(text, history=None):
        """
        Extracción avanzada de temas utilizando análisis semántico
        Retorna diccionario con temas primarios y secundarios con ponderación
        """
        # Taxonomía jerárquica de temas
        topic_taxonomy = {
            'desarrollo_personal': {
                'keywords': ['crecimiento', 'desarrollo', 'mejora', 'hábitos', 'metas'],
                'subtopics': {
                    'productividad': ['productividad', 'eficiencia', 'organización', 'gestión', 'tiempo'],
                    'meditación': ['meditación', 'mindfulness', 'atención', 'presente', 'calma'],
                    'propósito': ['propósito', 'significado', 'misión', 'valores', 'objetivos']
                }
            },
            'filosofía': {
                'keywords': ['filosofía', 'existencia', 'significado', 'propósito', 'ética'],
                'subtopics': {
                    'estoicismo': ['estoicismo', 'marcus', 'aurelio', 'seneca', 'epicteto'],
                    'existencialismo': ['existencialismo', 'sartre', 'existencia', 'libertad', 'absurdo'],
                    'metafísica': ['metafísica', 'realidad', 'ser', 'existencia', 'ontología']
                }
            },
            'psicología': {
                'keywords': ['psicología', 'mente', 'conducta', 'comportamiento', 'cerebro'],
                'subtopics': {
                    'emociones': ['emoción', 'sentimiento', 'afecto', 'gestión emocional'],
                    'cognitivo': ['cognitivo', 'pensamiento', 'creencia', 'sesgo', 'mental'],
                    'personalidad': ['personalidad', 'carácter', 'temperamento', 'rasgo']
                }
            },
            'relaciones': {
                'keywords': ['relación', 'amistad', 'pareja', 'interpersonal', 'social'],
                'subtopics': {
                    'comunicación': ['comunicación', 'diálogo', 'conversar', 'escucha'],
                    'conflictos': ['conflicto', 'desacuerdo', 'pelea', 'resolución'],
                    'intimidad': ['intimidad', 'conexión', 'vulnerabilidad', 'confianza']
                }
            },
            'trabajo': {
                'keywords': ['trabajo', 'empleo', 'carrera', 'profesión', 'laboral'],
                'subtopics': {
                    'liderazgo': ['liderazgo', 'líder', 'dirigir', 'equipo', 'influencia'],
                    'satisfacción': ['satisfacción', 'realización', 'propósito', 'motivación'],
                    'estrés': ['estrés', 'burnout', 'presión', 'ansiedad', 'agotamiento']
                }
            },
            'salud': {
                'keywords': ['salud', 'bienestar', 'enfermedad', 'médico', 'cuerpo'],
                'subtopics': {
                    'mental': ['mental', 'depresión', 'ansiedad', 'terapia', 'psicológico'],
                    'física': ['física', 'ejercicio', 'nutrición', 'descanso', 'energía'],
                    'hábitos': ['hábito', 'rutina', 'disciplina', 'consistencia', 'cambio']
                }
            },
            'tecnología': {
                'keywords': ['tecnología', 'digital', 'app', 'software', 'computadora'],
                'subtopics': {
                    'ia': ['ia', 'inteligencia artificial', 'machine learning', 'algoritmo'],
                    'internet': ['internet', 'web', 'online', 'digital', 'virtual'],
                    'programación': ['programación', 'código', 'desarrollo', 'software']
                }
            }
        }
        
        try:
            text_lower = text.lower()
            words = set(re.findall(r'\w+', text_lower))
            
            # Preprocesamiento: Eliminar stopwords
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'a', 'de', 'en', 'que', 'por', 'con'}
            words = words - stopwords
            
            # Puntuar temas primarios
            primary_scores = {}
            subtopic_scores = {}
            
            for topic, data in topic_taxonomy.items():
                # Puntuar tema principal
                topic_score = sum(2 for keyword in data['keywords'] if keyword in text_lower or 
                                  any(keyword in word for word in words))
                
                if topic_score > 0:
                    primary_scores[topic] = topic_score
                    
                    # Puntuar subtemas
                    for subtopic, keywords in data['subtopics'].items():
                        subtopic_score = sum(1 for keyword in keywords if keyword in text_lower or
                                           any(keyword in word for word in words))
                        
                        if subtopic_score > 0:
                            full_topic = f"{topic}.{subtopic}"
                            subtopic_scores[full_topic] = subtopic_score
            
            # Combinar resultados en formato jerárquico
            results = {
                'primary': {topic: score for topic, score in primary_scores.items() if score > 0},
                'secondary': {topic: score for topic, score in subtopic_scores.items() if score > 0}
            }
            
            # Si hay historia, aumentar score de temas previos para continuidad
            if history and isinstance(history, list) and history:
                for prev_topic in history:
                    if prev_topic in results['primary']:
                        results['primary'][prev_topic] += 1
                    elif prev_topic in results['secondary']:
                        results['secondary'][prev_topic] += 0.5
            
            # Si no se detecta ningún tema, usar un valor por defecto
            if not results['primary'] and not results['secondary']:
                results['primary']['general'] = 1
                
            return results
                
        except Exception as e:
            print(f"Error en extracción de temas: {e}")
            return {'primary': {'general': 1}, 'secondary': {}}

    # 3. Detector de intención conversacional
    def detect_intent(text, context=None):
        """
        Detecta la intención principal del usuario en su mensaje
        """
        try:
            text_lower = text.lower()

            # 3. Detector de intención conversacional
    def detect_intent(text, context=None):
        """
        Detecta la intención principal del usuario en su mensaje
        """
        try:
            text_lower = text.lower()
            
            # Patrones de intención con expresiones regulares y ponderación
            intent_patterns = {
                'pregunta_abierta': (r'\b(qué|cuál|cómo|por qué|para qué|dónde|cuándo|quién).*\?', 3),
                'pregunta_cerrada': (r'.*\b(es|son|está|están|puede|pueden|ha|han).*\?', 2),
                'solicitud_info': (r'\b(dime|cuéntame|explícame|necesito saber|quiero aprender)\b', 3),
                'solicitud_consejo': (r'\b(aconséjame|deberíamos|debería|qué (harías|hago|me recomiendas))\b', 4),
                'expresión_emocional': (r'\b(me siento|estoy (triste|feliz|preocupado|ansioso|emocionado))\b', 4),
                'reflexión': (r'\b(reflexión|pensar|contemplar|filosofar|meditar|considerar)\b', 3),
                'agradecimiento': (r'\b(gracias|agradezco|te lo agradezco)\b', 3),
                'despedida': (r'\b(adiós|hasta luego|nos vemos|chao|hasta pronto)\b', 4),
                'saludo': (r'\b(hola|buenos días|buenas tardes|buenas noches|saludos)\b', 4),
                'desacuerdo': (r'\b(no estoy de acuerdo|difiero|creo que no|incorrecto|equivocado)\b', 3),
                'acuerdo': (r'\b(estoy de acuerdo|coincido|exacto|así es|correcto)\b', 3),
                'clarificación': (r'\b(no entiendo|podrías aclarar|qué quieres decir|a qué te refieres)\b', 3),
                'preferencia': (r'\b(prefiero|me gusta más|mejor|en lugar de)\b', 2),
                'meta_conversación': (r'\b(hablemos de|cambiemos de tema|me gustaría discutir)\b', 3),
                'configuración': (r'\b(habla más (formal|profesional|casual|informal|cercano))\b', 4),
            }
            
            # Detectar intenciones por patrones
            found_intents = {}
            for intent, (pattern, weight) in intent_patterns.items():
                matches = re.findall(pattern, text_lower)
                if matches:
                    found_intents[intent] = weight * len(matches)
            
            # Añadir detección contextual
            if '?' in text:
                found_intents['pregunta'] = found_intents.get('pregunta', 0) + 2
                
            if '!' in text:
                found_intents['exclamación'] = 2
                
            # Usar contexto previo para mejorar detección
            if context and 'previous_intent' in context:
                prev_intent = context['previous_intent']
                # Dar continuidad a intenciones secuenciales
                if prev_intent in ['pregunta_abierta', 'solicitud_info'] and len(text.split()) < 5:
                    found_intents['continuación'] = 3
            
            # Determinar intención principal
            if found_intents:
                primary_intent = max(found_intents.items(), key=lambda x: x[1])
                secondary_intents = sorted([(i, s) for i, s in found_intents.items() if i != primary_intent[0]], 
                                          key=lambda x: x[1], reverse=True)[:2]
                
                return {
                    'primary': primary_intent[0],
                    'secondary': [i[0] for i in secondary_intents],
                    'confidence': min(10, primary_intent[1])
                }
            else:
                # Si no se detecta intención específica
                return {
                    'primary': 'declarativa',
                    'secondary': [],
                    'confidence': 1
                }
                
        except Exception as e:
            print(f"Error en detección de intención: {e}")
            return {'primary': 'indeterminada', 'secondary': [], 'confidence': 0}

    # 4. Análisis de estilo cognitivo del usuario
    def analyze_cognitive_style(text, history=None):
        """
        Analiza el estilo cognitivo y comunicativo del usuario para adaptar respuestas
        """
        try:
            text_lower = text.lower()
            
            # Indicadores de diferentes estilos cognitivos
            style_indicators = {
                'analítico': [
                    r'\b(analizar|análisis|lógico|datos|evidencia|estadística|precisión|exactitud)\b',
                    r'\b(por (un|otro) lado|sin embargo|no obstante|en contraste)\b'
                ],
                'emocional': [
                    r'\b(siento|sentir|emoción|me afecta|me duele|me alegra|me entristece)\b',
                    r'\b(realmente|sinceramente|honestamente|de corazón)\b'
                ],
                'pragmático': [
                    r'\b(funciona|utilidad|práctico|aplicación|implementar|solución|resultado)\b',
                    r'\b(cómo hago|cómo puedo|pasos para|método)\b'
                ],
                'conceptual': [
                    r'\b(concepto|idea|teoría|filosofía|significado|sentido|esencia)\b',
                    r'\b(¿qué significa|¿por qué|fundamentalmente|en esencia)\b'
                ],
                'visual': [
                    r'\b(ver|imagen|visualizar|imaginar|escenario|panorama|perspectiva)\b',
                    r'\b(mostrar|ilustrar|dibujar|representar)\b'
                ],
                'verbal': [
                    r'\b(explicar|describir|narrar|contar|relatar|expresar)\b',
                    r'\b(en palabras|forma de decir|manera de expresar)\b'
                ],
                'detallista': [
                    r'\b(detalle|específico|precisamente|exactamente|concretamente)\b',
                    r'\b(paso a paso|punto por punto|uno por uno)\b'
                ],
                'holístico': [
                    r'\b(general|panorama|completo|todo|entero|conjunto|sistema)\b',
                    r'\b(en general|como un todo|visión global|gran esquema)\b'
                ]
            }
            
            # Detectar presencia de indicadores
            style_scores = {style: 0 for style in style_indicators}
            for style, patterns in style_indicators.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    style_scores[style] += len(matches)
            
            # Analizar complejidad lingüística (indicador de estilo cognitivo)
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            
            complexity = 0
            if len(words) > 30:
                complexity += 1
            if avg_word_length > 6:
                complexity += 1
            if len(re.findall(r'[,;:]', text)) > 3:
                complexity += 1
            if len(re.findall(r'\b(sin embargo|no obstante|por consiguiente|en consecuencia)\b', text_lower)) > 0:
                complexity += 2
                
            # Determinar estilos dominantes
            dominant_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            
            result = {
                'primary_style': dominant_styles[0][0] if dominant_styles[0][1] > 0 else 'neutral',
                'secondary_style': dominant_styles[1][0] if len(dominant_styles) > 1 and dominant_styles[1][1] > 0 else None,
                'complexity': complexity,
                'detail_orientation': style_scores['detallista'] - style_scores['holístico'],
                'abstraction_level': style_scores['conceptual'] - style_scores['pragmático'],
                'emotional_analytical_balance': style_scores['emocional'] - style_scores['analítico']
            }
            
            # Incorporar historia si está disponible
            if history and isinstance(history, list) and len(history) > 0:
                # Suavizar cambios para evitar oscilaciones bruscas
                prev_primary = history[-1].get('primary_style')
                if prev_primary and prev_primary != result['primary_style'] and style_scores[prev_primary] > 0:
                    # Dar algo de inercia al estilo previo
                    result['secondary_style'] = result['primary_style']
                    result['primary_style'] = prev_primary
            
            return result
            
        except Exception as e:
            print(f"Error en análisis de estilo cognitivo: {e}")
            return {
                'primary_style': 'neutral',
                'secondary_style': None,
                'complexity': 1,
                'detail_orientation': 0,
                'abstraction_level': 0,
                'emotional_analytical_balance': 0
            }

    # 5. Análisis de contexto y coherencia conversacional
    def analyze_conversation_dynamics(user_message, history, context):
        """
        Analiza la dinámica, coherencia y progresión de la conversación
        """
        try:
            # Extraer métricas básicas
            current_time = time.time()
            time_since_last = current_time - context.get('last_interaction_time', current_time)
            is_new_session = time_since_last > 30 * 60  # 30 minutos
            
            # Analizar longitud de mensaje
            msg_length = len(user_message.split())
            msg_complexity = len(re.findall(r'[.;:]', user_message))
            
            # Evaluar continuidad temática
            topics = extract_topics(user_message, context.get('last_topics', []))
            primary_topics = list(topics['primary'].keys())
            
            topic_continuity = 0
            topic_shift = True
            
            if context.get('current_topic') and context['current_topic'] in primary_topics:
                topic_continuity = min(10, context.get('topic_retention', 0) + 2)
                topic_shift = False
            elif context.get('last_topics') and any(t in primary_topics for t in context['last_topics']):
                topic_continuity = min(10, context.get('topic_retention', 0) + 1)
                topic_shift = False
            else:
                topic_continuity = max(0, context.get('topic_retention', 0) - 3)
                topic_shift = True
            
            # Medir profundidad conversacional
            current_depth = context.get('depth', 0)
            
            # Aumenta con: longitud de mensaje, complejidad, continuidad temática
            depth_change = 0
            
            if msg_length > 30:  # Mensaje largo
                depth_change += 1
            if msg_complexity > 2:  # Mensaje complejo
                depth_change += 1
            if topic_continuity > 5:  # Alta continuidad temática
                depth_change += 1
            if context.get('initiative_ratio', 0.5) < 0.4:  # Usuario toma más iniciativa
                depth_change += 1
                
            new_depth = min(10, current_depth + depth_change)
            if topic_shift:
                new_depth = max(1, new_depth - 1)  # Reducir profundidad en cambios de tema
                
            # Medir nivel de compromiso (engagement)
            engagement = min(10, (
                (msg_length / 20) +        # Longitud normalizada (0-2.5 aprox)
                (msg_complexity) +         # Complejidad (0-5 aprox)
                (depth_change * 2) +       # Cambio en profundidad (0-4)
                (0 if is_new_session else 2)  # Continuidad de sesión
            ))
            
            # Determinar quién está tomando la iniciativa
            # < 0.5 significa que el usuario está dirigiendo más
            # > 0.5 significa que la IA está dirigiendo más
            
            # Calcular basado en preguntas, solicitudes, etc.
            user_intent = detect_intent(user_message, {'previous_intent': context.get('last_intent')})
            
            initiative_markers = {
                'pregunta_abierta': -0.1,    # Usuario toma iniciativa con preguntas
                'solicitud_info': -0.1,      # Usuario solicita información
                'meta_conversación': -0.2,   # Usuario dirige explícitamente la conversación
                'continuación': 0.1,         # Respuesta a iniciativa previa de la IA
            }
            
            # Ajustar ratio de iniciativa basado en intención detectada
            initiative_shift = initiative_markers.get(user_intent['primary'], 0)
            for intent in user_intent['secondary']:
                initiative_shift += initiative_markers.get(intent, 0) * 0.5
                
            # Suavizar cambios para evitar oscilaciones bruscas
            new_initiative = min(1.0, max(0.0, context.get('initiative_ratio', 0.5) + initiative_shift))
            
            return {
                'depth': new_depth,
                'engagement': round(engagement, 1),
                'initiative_ratio': round(new_initiative, 2),
                'topic_continuity': topic_continuity,
                'topic_shift': topic_shift,
                'current_topic': primary_topics[0] if primary_topics else context.get('current_topic'),
                'last_topics': primary_topics[:3],
                'topic_retention': topic_continuity,
                'last_intent': user_intent['primary'],
                'message_length': msg_length,
                'message_complexity': msg_complexity
            }
            
        except Exception as e:
            print(f"Error en análisis de dinámica conversacional: {e}")
            return {
                'depth': min(context.get('depth', 0) + 1, 10),
                'engagement': 5.0,
                'initiative_ratio': 0.5,
                'topic_continuity': 0,
                'topic_shift': True,
                'current_topic': 'general',
                'last_topics': ['general'],
                'topic_retention': 0
            }

   
# 6. Selección avanzada de recursos relevantes
    def get_relevant_resources(user_message, context, limit_reflexiones=3, limit_books=2):
        """
        Sistema avanzado de recuperación de recursos relevantes con vectorización semántica
        """
        try:
            # Si no hay mensaje de usuario, seleccionar al azar
            if not user_message:
                return {
                    'reflexiones': Reflexion.query.order_by(db.func.random()).limit(limit_reflexiones).all(),
                    'books': Book.query.order_by(db.func.random()).limit(limit_books).all()
                }
            
            # Extracción de temas para mejorar la búsqueda
            topics = extract_topics(user_message)
            intent = detect_intent(user_message)
            
            # Vectorización simple de palabras clave (en producción, usar embeddings)
            palabras_clave = []
            
            # Añadir palabras clave de temas primarios
            for topic in topics['primary']:
                if topic in topic_taxonomy:
                    palabras_clave.extend(topic_taxonomy[topic]['keywords'])
            
            # Añadir palabras clave de temas secundarios
            for topic in topics['secondary']:
                if '.' in topic:  # Formato "principal.secundario"
                    main, sub = topic.split('.')
                    if main in topic_taxonomy and sub in topic_taxonomy[main]['subtopics']:
                        palabras_clave.extend(topic_taxonomy[main]['subtopics'][sub])
            
            # Añadir palabras del mensaje original (eliminando stopwords)
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'a', 'de', 'en', 'que', 'por', 'con'}
            palabras_originales = [palabra.lower() for palabra in user_message.split() 
                                 if len(palabra) > 3 and palabra.lower() not in stopwords]
            
            palabras_clave.extend(palabras_originales)
            
            # Eliminar duplicados y normalizar
            palabras_clave = list(set(palabras_clave))
            
            # Función de puntuación basada en relevancia
            def score_reflexion(reflexion):
                titulo_lower = reflexion.titulo.lower()
                contenido_lower = reflexion.contenido.lower()
                
                # Puntuación por presencia de palabras clave
                keyword_score = sum(2 if palabra in titulo_lower else 1 
                                   for palabra in palabras_clave 
                                   if palabra in titulo_lower or palabra in contenido_lower)
                
                # Bonus por relevancia temática
                topic_score = 0
                for topic in topics['primary']:
                    if topic in reflexion.tags:
                        topic_score += 3
                        
                # Bonus por intención del usuario
                intent_score = 0
                if intent['primary'] in ['pregunta_abierta', 'solicitud_info'] and 'informativo' in reflexion.tags:
                    intent_score += 2
                elif intent['primary'] in ['solicitud_consejo'] and 'consejos' in reflexion.tags:
                    intent_score += 3
                elif intent['primary'] in ['expresión_emocional'] and 'emocional' in reflexion.tags:
                    intent_score += 3
                    
                # Bonus por contexto conversacional
                context_score = 0
                if context.get('depth', 0) > 6 and 'profundo' in reflexion.tags:
                    context_score += 2
                elif context.get('depth', 0) < 3 and 'básico' in reflexion.tags:
                    context_score += 2
                
                # Freshness - preferir contenido no mostrado recientemente
                recency_score = 0
                if hasattr(reflexion, 'last_shown'):
                    days_since_shown = (datetime.datetime.now() - reflexion.last_shown).days
                    recency_score = min(5, days_since_shown)
                else:
                    recency_score = 5  # Máximo para contenido nunca mostrado
                
                return keyword_score + topic_score + intent_score + context_score + recency_score
            
            # Función similar para libros
            def score_book(book):
                title_lower = book.title.lower()
                content_lower = book.content.lower()
                
                keyword_score = sum(2 if palabra in title_lower else 1 
                                   for palabra in palabras_clave 
                                   if palabra in title_lower or palabra in content_lower)
                
                # Puntuación adicional similar a reflexiones
                topic_score = sum(2 for topic in topics['primary'] if topic in book.tags)
                
                return keyword_score + topic_score
            
            # Buscar y puntuar todas las reflexiones y libros
            reflexiones_scores = [(reflexion, score_reflexion(reflexion)) 
                                 for reflexion in Reflexion.query.all()]
            
            books_scores = [(book, score_book(book)) 
                           for book in Book.query.all()]
            
            # Ordenar por puntuación y seleccionar los mejores
            reflexiones_relevantes = [item[0] for item in sorted(reflexiones_scores, 
                                                               key=lambda x: x[1], 
                                                               reverse=True)[:limit_reflexiones]]
            
            books_relevantes = [item[0] for item in sorted(books_scores, 
                                                         key=lambda x: x[1], 
                                                         reverse=True)[:limit_books]]
            
            # Actualizar timestamp de última visualización
            for reflexion in reflexiones_relevantes:
                reflexion.last_shown = datetime.datetime.now()
                db.session.add(reflexion)
            
            db.session.commit()
            
            return {
                'reflexiones': reflexiones_relevantes,
                'books': books_relevantes
            }
            
        except Exception as e:
            print(f"Error al buscar recursos relevantes: {e}")
            return {
                'reflexiones': Reflexion.query.order_by(db.func.random()).limit(limit_reflexiones).all(),
                'books': Book.query.order_by(db.func.random()).limit(limit_books).all()
            }

    # 7. Gestión de preferencias de usuario adaptativa
    def update_user_preferences(user_id, message, context):
        """
        Actualiza las preferencias del usuario basadas en el mensaje y comportamiento
        """
        try:
            preferences = session.get('user_preferences', {})
            
            # Detector de preferencias de estilo de comunicación
            style_keywords = {
                'formal': ['formal', 'profesional', 'académico', 'técnico', 'riguroso'],
                'casual': ['casual', 'relajado', 'coloquial', 'informal', 'cercano'],
                'empático': ['empático', 'comprensivo', 'sensible', 'emocional', 'cálido'],
                'analítico': ['analítico', 'lógico', 'racional', 'estructurado', 'sistemático']
            }
            
            msg_lower = message.lower()
            
            # Detectar preferencias explícitas
            for style, keywords in style_keywords.items():
                for keyword in keywords:
                    if f"más {keyword}" in msg_lower or f"habla {keyword}" in msg_lower:
                        preferences['communication_style'] = style
                        break
            
            # Detectar preferencias de profundidad
            depth_keywords = {
                'conciso': ['breve', 'corto', 'conciso', 'resumido', 'directo'],
                'moderado': ['moderado', 'balanceado', 'equilibrado'],
                'profundo': ['profundo', 'detallado', 'extenso', 'elaborado', 'completo']
            }
            
            for depth, keywords in depth_keywords.items():
                for keyword in keywords:
                    if f"sé {keyword}" in msg_lower or f"responde {keyword}" in msg_lower:
                        preferences['response_depth'] = depth
                        break
            
            # Actualizar intereses detectados
            topics = extract_topics(message)
            current_interests = preferences.get('interests', [])
            
            # Añadir nuevos temas de interés
            for topic in topics['primary']:
                if topic not in current_interests and topic != 'general':
                    current_interests.append(topic)
            
            # Limitar a los 5 intereses más recientes
            preferences['interests'] = current_interests[-5:]
            
            # Actualizar preferencias cognitivas
            cognitive_style = analyze_cognitive_style(message)
            current_cognitive = preferences.get('cognitive_preferences', [])
            
            if cognitive_style['primary_style'] not in current_cognitive:
                current_cognitive.append(cognitive_style['primary_style'])
                
            # Limitar a los 3 estilos cognitivos más recientes
            preferences['cognitive_preferences'] = current_cognitive[-3:]
            
            # Actualizar en la sesión
            session['user_preferences'] = preferences
            
            # Si el usuario está autenticado, también actualizar en base de datos
            if user_id:
                user = User.query.get(user_id)
                if user:
                    user.preferences = json.dumps(preferences)
                    db.session.add(user)
                    db.session.commit()
            
            return preferences
            
        except Exception as e:
            print(f"Error al actualizar preferencias: {e}")
            return session.get('user_preferences', {})

    # Procesamiento de solicitudes POST (interacción con la IA)
    if request.method == 'POST':
        # Verificar límite de interacciones
        if not (is_premium or is_admin) and interaction_count >= effective_limit:
            return jsonify({
                'response': 'Has alcanzado tu límite diario de conversación.\n\n'
                            'Te invito a explorar nuestra versión Premium que ofrece:\n'
                            '- Conversaciones ilimitadas\n'
                            '- Acceso a recursos exclusivos\n'
                            '- Análisis de personalidad avanzado\n'
                            '- Retroalimentación personalizada\n\n'
                            '¡Tu crecimiento personal no tiene por qué detenerse!',
                'remaining_interactions': remaining_interactions,
                'limit_reached': True,
                'subscribe_url': url_for('subscribe_info', _external=True)
            })

        # Obtener el mensaje del usuario
        message = request.json.get('message', '').strip()
        
        # Análisis completo del mensaje y contexto
        start_time = time.time()
        
        # 1. Actualizar contexto temporal
        temporal_context = session.get('temporal_context', {})
        temporal_context['last_interaction_time'] = time.time()
        
        # Detectar hora especificada por el usuario
        try:
            hour_match = re.search(r'son las (\d{1,2}):(\d{2})(?:\s*hs)?', message.lower())
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2))
                temporal_context['user_specified_hour'] = hour + minute / 60
        except Exception as e:
            print(f"Error al detectar hora: {e}")
        
        session['temporal_context'] = temporal_context
        
        # 2. Análisis emocional
        emotional_context = session.get('emotional_context', {})
        current_emotion = analyze_sentiment(message, emotional_context.get('history', []))
        
        # Actualizar historial de emociones
        emotion_history = emotional_context.get('history', [])
        emotion_history.append(current_emotion)
        emotional_context['history'] = emotion_history[-5:]  # Mantener solo las 5 más recientes
        
        # Actualizar estado emocional actual
        emotional_context.update({
            'current_state': current_emotion['state'],
            'valence': current_emotion['valence'],
            'intensity': current_emotion['intensity'],
            'arousal': current_emotion['arousal'],
            'trend': current_emotion['trend']
        })
        
        session['emotional_context'] = emotional_context
        
        # 3. Actualizar métricas de conversación
        conversation_dynamics = session.get('conversation_dynamics', {})
        updated_dynamics = analyze_conversation_dynamics(
            message, 
            session.get('chat_history', []), 
            conversation_dynamics
        )
        conversation_dynamics.update(updated_dynamics)
        session['conversation_dynamics'] = conversation_dynamics
        
        # 4. Actualizar preferencias de usuario
        user_preferences = update_user_preferences(
            identifier, 
            message, 
            session.get('conversation_context', {})
        )
        
        # 5. Actualizar contexto de conversación
        conversation_context = session.get('conversation_context', {})
        conversation_context.update({
            'current_topic': updated_dynamics['current_topic'],
            'topic_retention': updated_dynamics['topic_retention'],
            'last_topics': updated_dynamics['last_topics'],
            'last_intent': updated_dynamics['last_intent'],
            'cognitive_load': min(10, conversation_context.get('cognitive_load', 0) + 
                                (1 if updated_dynamics['message_complexity'] > 2 else 0))
        })
        session['conversation_context'] = conversation_context
        
        # 6. Actualizar historial y contadores
        chat_history = session.get('chat_history', [])
        chat_history.append(f"Usuario: {message}")
        session['chat_history'] = chat_history[-20:]  # Mantener solo los 20 mensajes más recientes
        
        session_metrics = session.get('session_metrics', {})
        session_metrics['message_count'] = session_metrics.get('message_count', 0) + 1
        session['session_metrics'] = session_metrics
        
        # 7. Buscar recursos relevantes
        relevant_resources = get_relevant_resources(
            message,
            conversation_context,
            limit_reflexiones=3,
            limit_books=2
        )
        
        # Preparar resúmenes de recursos
        reflexiones_summary = "\n\n".join([
            f"Título: {reflexion.titulo}\n"
            f"Contenido: {reflexion.contenido[:300]}...\n"
            f"Relevancia: Alta\n"
            f"Tags: {', '.join(reflexion.tags)}"
            for reflexion in relevant_resources['reflexiones']
        ]) if relevant_resources['reflexiones'] else "No hay reflexiones relevantes para este tema."
        
        books_summary = "\n\n".join([
            f"Título: {book.title}\n"
            f"Contenido: {book.content[:300]}...\n"
            f"Relevancia: Alta\n"
            f"Tags: {', '.join(book.tags)}"
            for book in relevant_resources['books']
        ]) if relevant_resources['books'] else "No hay libros relevantes para este tema."

        # Calcular contexto temporal enriquecido
        try:
            current_time = datetime.datetime.now(tz=datetime.timezone(timedelta(hours=-3)))  # ART
            server_



















