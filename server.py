from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import json
import logging
from typing import Optional, Dict, Any
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Text Analyzer Server")

class TextData(BaseModel):
    text: str
    user_id: int
    username: Optional[str] = None
    chat_id: Optional[int] = None
    timestamp: Optional[str] = None

class AnalysisResult(BaseModel):
    toxicity_level: str
    confidence: float
    category: str
    flags: list[str]

# Папки для сохранения
OUTPUT_DIR = "saved_texts"
LOG_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class TextAnalyzer:
    """Класс для анализа текста с помощью AI модели"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Загрузка AI модели"""
        try:
            # Здесь должна быть загрузка твоей модели
            self.model = load_model('toxicity_classifier_model (1).h5')
            with open('toxicity_classifier_tokenizer (1).pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open('toxicity_classifier_config (1).json', 'r') as f:
                self.config = json.load(f)
            # Это пример - замени на свою реализацию
            logger.info("🔄 Загружаю AI модель...")
            
            # Пример заглушки - замени на свою модель
            self.model_loaded = True
            logger.info("✅ AI модель готова к работе")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.model_loaded = False
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """Анализ текста на токсичность"""
        try:
            # if not self.model_loaded:
            #     return self._fallback_analysis(text)
            
            # ЗДЕСЬ ВСТАВЬ КОД ТВОЕЙ МОДЕЛИ
            # Пример реализации:
            #toxicity_score = self._predict_toxicity(text)

            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.config['max_len'], padding='post', truncating='post')
            
            # Предсказание
            prediction = self.model.predict(padded_sequence, verbose=0)
            
            # Возвращаем номер класса с максимальной вероятностью
            result = np.argmax(prediction[0])
            
            # Определяем уровень токсичности
            if result == 2:
                category = "toxic"
            elif result == 1:
                category = "offensive"
            else:
                category = "neutral"
            
            return AnalysisResult(
                toxicity_level=str(result),
                confidence=0.0,
                category=category,
                flags=[]
            )
            
        except Exception as e:
            logger.error(f"Ошибка анализа текста: {e}")
            return "None"
    
    # def _predict_toxicity(self, text: str) -> float:
    #     """Заглушка для предсказания токсичности - замени на свою модель"""
    #     # Пример простой эвристики
    #     toxic_words = ['дурак', 'идиот', 'мудак', 'сволочь', 'ублюдок']
    #     text_lower = text.lower()
        
    #     for word in toxic_words:
    #         if word in text_lower:
    #             return 0.9
        
    #     if any(char in text for char in '!@#$%^&*') and len(text) < 20:
    #         return 0.7
            
    #     return 0.1
    
    # def _fallback_analysis(self, text: str) -> AnalysisResult:
    #     """Резервный анализ если модель не загружена"""
    #     return AnalysisResult(
    #         toxicity_level="unknown",
    #         confidence=0.0,
    #         category="not_analyzed", 
    #         flags=["model_not_available"]
    #     )

# Инициализация анализатора
analyzer = TextAnalyzer()

def save_to_json(data: dict, filename: str):
    """Сохранение данных в JSON файл"""
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath
    except Exception as e:
        logger.error(f"Ошибка сохранения JSON: {e}")
        return None

def log_user_activity(user_id: int, username: str, text: str, analysis: AnalysisResult):
    """Логирование активности пользователя"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "username": username,
        "text_length": len(text),
        "toxicity_level": analysis.toxicity_level,
        "category": analysis.category,
        "confidence": analysis.confidence,
        "flags": analysis.flags
    }
    
    # Лог в файл
    log_file = os.path.join(LOG_DIR, f"user_activity.log")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    # Также в консоль
    logger.info(f"👤 User {user_id} ({username}): {analysis.toxicity_level} label")

@app.post("/analyze_text")
async def analyze_text(data: TextData):
    """Основной endpoint для анализа текста"""
    try:
        start_time = datetime.now()
        
        # Добавляем timestamp если не указан
        if not data.timestamp:
            data.timestamp = datetime.now().isoformat()
        
        # Анализируем текст
        analysis_result = analyzer.analyze_text(data.text)
        
        # Готовим данные для сохранения
        save_data = {
            "user_info": {
                "user_id": data.user_id,
                "username": data.username,
                "chat_id": data.chat_id
            },
            "text_data": {
                "text": data.text,
                "timestamp": data.timestamp,
                "length": len(data.text)
            },
            "analysis_result": {
                "toxicity_level": analysis_result.toxicity_level,
                "confidence": analysis_result.confidence,
                "category": analysis_result.category,
                "flags": analysis_result.flags,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        # Сохраняем в файл
        filename = f"user_{data.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = save_to_json(save_data, filename)
        
        # Логируем активность
        log_user_activity(
            data.user_id, 
            data.username or "unknown", 
            data.text, 
            analysis_result
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Текст проанализирован за {processing_time:.2f}с")
        
        return {
            "status": "success",
            "filename": filename,
            "analysis": analysis_result.dict(),
            "processing_time": processing_time,
            "message": "Text analyzed and saved successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа текста: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Получение статистики сервера"""
    try:
        # Считаем файлы
        json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
        
        # Анализируем лог активности
        activity_log = os.path.join(LOG_DIR, "user_activity.log")
        user_activities = []
        
        if os.path.exists(activity_log):
            with open(activity_log, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        user_activities.append(json.loads(line))
        
        # Статистика по токсичности
        toxicity_stats = {}
        for activity in user_activities:
            level = activity.get('toxicity_level', 'unknown')
            toxicity_stats[level] = toxicity_stats.get(level, 0) + 1
        
        return {
            "total_texts_analyzed": len(json_files),
            "toxicity_statistics": toxicity_stats,
            "unique_users": len(set(a.get('user_id') for a in user_activities)),
            "server_uptime": "active"  # Можно добавить реальное время
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return {"error": "Could not retrieve statistics"}

@app.get("/user_activity/{user_id}")
async def get_user_activity(user_id: int, limit: int = 10):
    """Получение активности конкретного пользователя"""
    try:
        activity_log = os.path.join(LOG_DIR, "user_activity.log")
        user_activities = []
        
        if os.path.exists(activity_log):
            with open(activity_log, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        activity = json.loads(line)
                        if activity.get('user_id') == user_id:
                            user_activities.append(activity)
        
        return {
            "user_id": user_id,
            "total_activities": len(user_activities),
            "recent_activities": user_activities[-limit:]
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения активности пользователя: {e}")
        return {"error": "Could not retrieve user activity"}

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Audio Text Analyzer Server is running", 
        "endpoints": {
            "analyze_text": "POST /analyze_text",
            "stats": "GET /stats", 
            "user_activity": "GET /user_activity/{user_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Запуск сервера анализа текста...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
