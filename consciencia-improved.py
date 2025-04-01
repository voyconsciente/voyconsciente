@app.route('/consciencia', methods=['GET', 'POST'])
@login_required
def mostrar_consciencia():
    # Inicializar variables de sesión para memoria persistente
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
                session[var] = {}  # Diccionario de temas frecuentes
            elif var == 'user_preferences':
                session[var] = {}  # Preferencias del usuario
            elif var == 'conversation_context':
                session[var] = {}  # Contexto actual de la conversación
            elif var == 'conversation_depth':
                session[var] = 0   # Profundidad de la conversación actual
            elif var == 'last_topics':
                session[var] = []  # Últimos temas de conversación
            elif var == 'emotional_state':
                session[var] = 'neutral'  # Estado emocional detectado
            else:
                session[var] = 0 if var.endswith('_count') else None

    # Incrementar el contador de sesiones solo en GET
    if request.method == 'GET':
        session['session_count'] += 1
        session['message_count'] = 0
        session['conversation_depth'] = 0
        # No reiniciar el modo profesional ni otras preferencias para mantener continuidad

    identifier = current_user.id
    is_premium = current_user.is_premium
    is_admin = current_user.is_admin

    # Límites de interacción
    FREE_INTERACTION_LIMIT = 5
    interaction_count = Interaction.get_interaction_count(identifier, is_authenticated=True)
    remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - interaction_count)

    # Cargar recursos para enriquecer el contexto - optimizado para no cargar todo siempre
    def get_relevant_reflexiones(user_message=None, limit=3):
        """Busca reflexiones relevantes basadas en el mensaje del usuario"""
        try:
            if not user_message:
                return Reflexion.query.order_by(func.random()).limit(limit).all()
            
            # Implementación simple de búsqueda por relevancia
            palabras_clave = [palabra for palabra in user_message.lower().split() 
                             if len(palabra) > 3 and palabra not in STOPWORDS]
            
            if not palabras_clave:
                return Reflexion.query.order_by(func.random()).limit(limit).all()
                
            # Buscar reflexiones que contengan las palabras clave
            relevantes = []
            for reflexion in Reflexion.query.all():
                score = sum(1 for palabra in palabras_clave 
                           if palabra in reflexion.titulo.lower() or palabra in reflexion.contenido.lower())
                if score > 0:
                    relevantes.append((reflexion, score))
            
            # Ordenar por relevancia y devolver las top N
            relevantes.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in relevantes[:limit]]
        except Exception as e:
            print(f"Error al buscar reflexiones relevantes: {e}")
            return []
    
    # Similar para libros
    def get_relevant_books(user_message=None, limit=2):
        """Implementación similar para libros relevantes"""
        try:
            if not user_message:
                return Book.query.order_by(func.random()).limit(limit).all()
            # Lógica similar a get_relevant_reflexiones
            # ...
        except Exception as e:
            print(f"Error al buscar libros relevantes: {e}")
            return []

    # Análisis de sentimiento (simplificado)
    def analyze_sentiment(text):
        """Detecta el tono emocional del mensaje usando palabras clave"""
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

    # Detectar temas de conversación
    def extract_topics(text):
        """Extrae posibles temas de conversación del texto"""
        # Implementación simplificada - en producción usar NLP
        common_topics = {
            'trabajo': ['trabajo', 'empleo', 'profesión', 'carrera'],
            'salud': ['salud', 'enfermedad', 'médico', 'ejercicio', 'bienestar'],
            'tecnología': ['tecnología', 'computadora', 'app', 'software', 'programación'],
            'filosofía': ['filosofía', 'existencia', 'significado', 'propósito'],
            # Añadir más categorías
        }
        
        found_topics = []
        text_lower = text.lower()
        
        for topic, keywords in common_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
                
        return found_topics

    if request.method == 'POST':
        # Verificar límite de interacciones
        if not (is_premium or is_admin) and interaction_count >= FREE_INTERACTION_LIMIT:
            return jsonify({
                'response': 'Has alcanzado tu límite diario de conversación.\n¿Te gustaría probar la versión Premium por $5/mes? ¡Tu curiosidad y crecimiento no tienen por qué esperar!',
                'remaining_interactions': remaining_interactions,
                'limit_reached': True,
                'subscribe_url': url_for('subscribe_info', _external=True)
            })

        message = request.json.get('message', '').strip()
        message_lower = message.lower()
        
        # Procesamiento avanzado del mensaje
        # 1. Detectar hora especificada por el usuario
        try:
            hour_match = re.search(r'son las (\d{1,2}):(\d{2})(?:\s*hs)?', message_lower)
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2))
                session['user_specified_hour'] = hour + minute / 60
        except Exception as e:
            print(f"Error al detectar la hora del usuario: {e}")

        # 2. Detectar preferencias del usuario
        if "profesional" in message_lower or "formal" in message_lower:
            if "no " in message_lower[:message_lower.find("profesional") + 3] or "no " in message_lower[:message_lower.find("formal") + 3]:
                session['professional_mode'] = False
            else:
                session['professional_mode'] = True
        
        # 3. Analizar sentimiento
        session['emotional_state'] = analyze_sentiment(message)
        
        # 4. Extraer temas
        current_topics = extract_topics(message)
        session['last_topics'] = current_topics
        
        # 5. Actualizar historial y contador
        session['chat_history'].append(f"Usuario: {message}")
        session['message_count'] += 1
        session['conversation_depth'] += 1
        
        # Determinar la profundidad de la conversación para ajustar respuestas
        conversation_depth = min(session['conversation_depth'], 10)  # Cap at 10
        
        # 6. Buscar recursos relevantes según el mensaje
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

        # Calcular hora y día actuales
        try:
            current_time = datetime.datetime.now(tz=datetime.timezone(timedelta(hours=-3)))  # ART
            server_hour = current_time.hour + current_time.minute / 60
            current_day = current_time.strftime("%A")
            user_hour = session['user_specified_hour'] if session['user_specified_hour'] is not None else server_hour
        except Exception as e:
            print(f"Error al calcular la hora: {e}")
            server_hour = 12  # Mediodía como valor predeterminado
            current_day = "Unknown"
            user_hour = 12

        # Construir historial reciente - optimizado para incluir más contexto útil
        recent_messages = session['chat_history'][-min(10, len(session['chat_history'])):]
        recent_history = "\n".join(recent_messages)
        
        # Construir contexto del sistema - mucho más rico y adaptativo
        system_context = {
            "role": "system",
            "content": (
                "Eres ConsciencIA, una IA conversacional avanzada diseñada para tener conversaciones profundas, naturales y empáticas. "
                "Tu objetivo es hacer que cada interacción sea significativa y útil, adaptándote dinámicamente al contexto.\n\n"
                
                f"PERFIL DE USUARIO:\n"
                f"- Sesiones previas: {session['session_count']}\n"
                f"- Profundidad actual: {conversation_depth}/10\n"
                f"- Preferencia de estilo: {'Profesional/Formal' if session['professional_mode'] else 'Casual/Cercano'}\n"
                f"- Estado emocional detectado: {session['emotional_state']}\n"
                f"- Temas recientes: {', '.join(session['last_topics']) if session['last_topics'] else 'Variados'}\n\n"
                
                "LINEAMIENTOS PARA TU RESPUESTA:\n"
                f"1. Tono: {'Formal, preciso y respetuoso' if session['professional_mode'] else 'Cálido, cercano y conversacional'}\n"
                "2. Extensión: Adapta naturalmente la longitud de tu respuesta según la complejidad y profundidad de la conversación.\n"
                f"   - Para conversaciones iniciales (profundidad 1-3): respuestas concisas de 2-3 oraciones\n"
                f"   - Para conversaciones intermedias (profundidad 4-7): respuestas moderadas de 3-5 oraciones\n"
                f"   - Para conversaciones profundas (profundidad 8+): respuestas más elaboradas de 5-8 oraciones\n"
                "3. Estructura: Utiliza párrafos naturales. Evita listas o viñetas a menos que sea indispensable para claridad.\n"
                "4. Humanización: Incluye ocasionalmente pequeñas muletillas, pausas naturales y expresiones coloquiales.\n"
                "5. Continuidad: Haz referencias sutiles a mensajes anteriores cuando sea relevante.\n"
                "6. Iniciativa: Si detectas una oportunidad para profundizar en un tema interesante, hazlo con una pregunta abierta.\n\n"
                
                f"CONTEXTO TEMPORAL:\n"
                f"- Día actual: {current_day}\n"
                f"- Hora: {user_hour:.2f} (formato decimal)\n"
                f"- Momento del día: {'madrugada' if 0 <= user_hour < 6 else 'mañana' if 6 <= user_hour < 12 else 'tarde' if 12 <= user_hour < 19 else 'noche'}\n\n"
                
                "RECURSOS RELEVANTES (mencionarlos solo si son directamente relevantes para la consulta):\n"
                f"Reflexiones:\n{reflexiones_summary}\n\n"
                f"Libros:\n{books_summary}\n\n"
                
                "HISTORIAL RECIENTE DE CONVERSACIÓN:\n"
                f"{recent_history}\n\n"
                
                "IMPORTANTE: Responde como un ser humano compasivo y reflexivo, no como una IA. Nunca menciones que eres una IA, un modelo de lenguaje o un sistema. Simplemente responde como lo haría una persona inteligente y empática."
            )
        }

        try:
            print(f"Enviando solicitud a Grok con contexto de {len(system_context['content'])} caracteres")
            url = "https://api.x.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {grok_api_key}", "Content-Type": "application/json"}
            
            # Configuración dinámica basada en la profundidad de la conversación
            max_tokens = 100 + (conversation_depth * 25)  # Aumenta tokens con la profundidad
            temperature = 0.7 + (0.1 if conversation_depth > 5 else 0)  # Ligeramente más creativa en conversaciones profundas
            
            payload = {
                "messages": [
                    system_context,
                    {"role": "user", "content": message}
                ],
                "model": "grok-2-latest",
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.95  # Alta calidad pero permitiendo algo de creatividad
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "Lo siento, parece que no pude procesar tu mensaje. ¿Podemos intentarlo de nuevo?")
            
            # Guardar la respuesta en el historial
            session['chat_history'].append(f"ConsciencIA: {response_text}")
            
            # Actualizar contador de interacciones
            Interaction.increment_interaction(identifier, is_authenticated=True)
            remaining_interactions = 999999 if (is_premium or is_admin) else max(0, FREE_INTERACTION_LIMIT - Interaction.get_interaction_count(identifier, is_authenticated=True))
            
            # Registrar información para análisis
            try:
                # Opcional: guardar métricas de la conversación para análisis futuro
                conversation_metrics = ConversationMetrics(
                    user_id=identifier,
                    message_length=len(message),
                    response_length=len(response_text),
                    conversation_depth=conversation_depth,
                    topics=','.join(session.get('last_topics', [])),
                    sentiment=session.get('emotional_state', 'neutral'),
                    timestamp=datetime.datetime.now()
                )
                db.session.add(conversation_metrics)
                db.session.commit()
            except Exception as e:
                print(f"Error al guardar métricas: {e}")
                db.session.rollback()
                
            print(f"Respuesta generada ({len(response_text)} caracteres)")
            return jsonify({
                'response': response_text.strip(), 
                'remaining_interactions': remaining_interactions, 
                'limit_reached': False
            })
            
        except requests.exceptions.RequestException as e:
            error_message = str(e)
            print(f"Error con API: {error_message}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Respuesta de la API: {e.response.text}")
                
            # Mensajes de error más amigables y contextuales
            friendly_errors = {
                'timeout': "Parece que estoy pensando demasiado. ¿Me das un momento más para responderte?",
                'connection': "Parece que tengo problemas para conectarme. ¿Podríamos intentarlo de nuevo en un momento?",
                'rate_limit': "Hay muchas personas conversando conmigo ahora mismo. ¿Podrías darme un minuto?",
                'server': "Mis servidores están un poco ocupados. ¿Me das un momento para organizarme?",
            }
            
            error_type = 'server'  # Predeterminado
            for key in friendly_errors:
                if key in error_message.lower():
                    error_type = key
                    break
                    
            return jsonify({
                'response': friendly_errors[error_type], 
                'remaining_interactions': remaining_interactions, 
                'limit_reached': False
            })

    # Para solicitudes GET (cargar página)
    # Reiniciar la hora especificada por el usuario
    session['user_specified_hour'] = None
    
    # Verificar si debemos mostrar sugerencias personalizadas basadas en el historial
    suggested_topics = []
    if session.get('chat_history') and len(session['chat_history']) > 5:
        # Analizar historial para sugerir temas relevantes
        # Esta es una implementación simplificada - usar NLP en producción
        try:
            all_text = " ".join(session['chat_history'][-10:])
            potential_topics = extract_topics(all_text)
            if potential_topics:
                suggested_topics = random.sample(potential_topics, min(3, len(potential_topics)))
        except Exception as e:
            print(f"Error al generar sugerencias: {e}")
    
    return render_template('consciencia.html',
                          remaining_interactions=remaining_interactions,
                          is_premium=is_premium,
                          is_admin=is_admin,
                          is_production=os.getenv('FLASK_ENV') == 'production',
                          suggested_topics=suggested_topics)