from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI(title="Audio Text Server")

class TextData(BaseModel):
    text: str
    user_id: int = None
    timestamp: str = None

# Папка для сохранения файлов
OUTPUT_DIR = "saved_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/save_text")
async def save_text(data: TextData):
    try:
        # Добавляем timestamp если не указан
        if not data.timestamp:
            data.timestamp = datetime.now().isoformat()
        
        # Создаем имя файла
        filename = f"user_{data.user_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Сохраняем в файл
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"User ID: {data.user_id}\n")
            f.write(f"Timestamp: {data.timestamp}\n")
            f.write(f"Text: {data.text}\n")
        
        return {"status": "success", "filename": filename, "message": "Text saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving text: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Audio Text Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
