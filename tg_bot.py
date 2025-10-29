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
SERVER_URL = "http://localhost:8000/save_text"


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
        """Обработка голосовых сообщений"""
        try:
            await update.message.reply_text("🔄 Обрабатываю аудио сообщение...")
            
            # Получаем файл голосового сообщения
            voice_file = await update.message.voice.get_file()
            
            # Конвертируем в текст
            text = await self.audio_to_text(voice_file.file_path)
            
            if text:
                # Отправляем на сервер
                success = await self.send_to_server(text, update.effective_user.id)
                
                if success:
                    await update.message.reply_text(f"Распознанный текст:\n{text}")
                else:
                    await update.message.reply_text("❌ Ошибка при сохранении на сервер")
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
                    await update.message.reply_text(f"Распознанный текст:\n{text}")
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
    
    async def send_to_server(self, text: str, user_id: int) -> bool:
        """Отправляет текст на сервер"""
        try:
            data = {
                "text": text,
                "user_id": user_id
            }
            
            response = requests.post(SERVER_URL, json=data)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending to server: {e}")
            return False
    
    def run(self):
        """Запускает бота"""
        print("Бот запущен...")
        self.application.run_polling()

if __name__ == "__main__":
    BOT_TOKEN = ""
    
    bot = AudioToTextBot()
    bot.run()
