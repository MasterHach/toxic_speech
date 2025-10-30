import os
import requests
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import io

# Конфигурация
BOT_TOKEN = ""
SERVER_URL = "http://localhost:8000/analyze_text"


FFMPEG_PATH = "C:\\Users\\Admin\\Documents\\ffmpeg-2025-10-27-git-68152978b5-full_build\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\Users\\Admin\\Documents\\ffmpeg-2025-10-27-git-68152978b5-full_build\\bin\\ffprobe.exe"

# # Принудительно указываем пути для pydub
import pydub
pydub.AudioSegment.ffmpeg = FFMPEG_PATH
pydub.AudioSegment.ffprobe = FFPROBE_PATH

# # Также добавляем в PATH
os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ["PATH"]

class AudioToTextBot:
    def __init__(self):
        self.application = Application.builder().token(BOT_TOKEN).build()
        self.recognizer = sr.Recognizer()
        
        # Настройка обработчиков
        self.setup_handlers()
    
    def setup_handlers(self):
        # Обработчик голосовых сообщений
        self.application.add_handler(
            MessageHandler(filters.VOICE, self.handle_voice_message)
        )
        
        # Обработчик аудио файлов
        self.application.add_handler(
            MessageHandler(filters.AUDIO, self.handle_audio_message)
        )
        
        # Обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message)
        )
    
    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка голосовых сообщений с анализом"""
        try:
            await update.message.reply_text("🔄 Обрабатываю аудио сообщение...")
            
            # Конвертируем аудио в текст (твоя существующая функция)
            voice_file = await update.message.voice.get_file()
            text = await self.audio_to_text(voice_file.file_path)
            
            if text:
                # Отправляем на сервер для анализа
                analysis_result = await self.send_to_server(
                    text, 
                    update.effective_user.id,
                    update.effective_user.username,
                    update.message.chat_id
                )
                
                if analysis_result:
                    # Формируем красивый ответ
                    response_message = self.format_analysis_response(text, analysis_result)
                    await update.message.reply_text(response_message, parse_mode='HTML')
                else:
                    await update.message.reply_text("❌ Ошибка при анализе текста на сервере")
            else:
                await update.message.reply_text("❌ Не удалось распознать текст")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")
    
    async def handle_audio_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка аудио файлов"""
        try:
            await update.message.reply_text("🔄 Обрабатываю аудио файл...")
            
            audio_file = await update.message.audio.get_file()
            text = await self.audio_to_text(audio_file.file_path)
            
            if text:
                success = await self.send_to_server(text, update.effective_user.id)
                
                if success:
                    await update.message.reply_text(f"✅ Текст успешно сохранен!\n\n📝 Распознанный текст:\n{text}")
                else:
                    await update.message.reply_text("❌ Ошибка при сохранении на сервер")
            else:
                await update.message.reply_text("❌ Не удалось распознать текст")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений (просто сохраняем на сервер)"""
        try:
            text = update.message.text
            success = await self.send_to_server(text, update.effective_user.id)
            
            if success:
                await update.message.reply_text("✅ Текст успешно сохранен на сервер!")
            else:
                await update.message.reply_text("❌ Ошибка при сохранении на сервер")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")
    
    async def audio_to_text(self, audio_url: str) -> str:
        """Конвертирует аудио в текст"""
        try:
            # Скачиваем аудио файл
            import urllib.request
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as tmp_file:
                urllib.request.urlretrieve(audio_url, tmp_file.name)
                
                # Конвертируем OGG в WAV
                audio = AudioSegment.from_ogg(tmp_file.name)
                wav_data = io.BytesIO()
                audio.export(wav_data, format="wav")
                wav_data.seek(0)
            
            # Распознаем текст
            with sr.AudioFile(wav_data) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='ru-RU')
                
            # Удаляем временный файл
            os.unlink(tmp_file.name)
            
            return text
            
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Error in audio_to_text: {e}")
            return None
    
    async def send_to_server(self, text: str, user_id: int, username: str, chat_id: int) -> dict:
        """Отправляет текст на сервер для анализа"""
        try:
            data = {
                "text": text,
                "user_id": user_id,
                "username": username,
                "chat_id": chat_id
            }
            
            response = requests.post(SERVER_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('analysis', {})
            else:
                print(f"Server error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error sending to server: {e}")
            return None
        
    def format_analysis_response(self, text: str, analysis: dict) -> str:
        """Форматирует ответ с анализом текста"""
        # Эмодзи для разных уровней токсичности


        toxic_label = analysis.get('toxicity_level', 'unknown')
        if toxic_label == "0":
            emoji = "🟢"
        elif toxic_label == "1":
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        #emoji = toxicity_emojis.get(analysis.get('toxicity_level', 'unknown'), '⚪')
        
        response = f"""
            {emoji} <b>Анализ текста завершен</b>

            📝 <b>Распознанный текст:</b>
            {text}

            📊 <b>Результат анализа:</b>
            • Уровень токсичности: <b>{analysis.get('toxicity_level', 'unknown')}</b>
            • Категория: <b>{analysis.get('category', 'unknown')}</b>
            """
        return response
    
    def run(self):
        """Запускает бота"""
        print("Бот запущен...")
        self.application.run_polling()

if __name__ == "__main__":
    # Замени на свой токен бота
    BOT_TOKEN = ""
    
    bot = AudioToTextBot()
    bot.run()
