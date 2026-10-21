import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")
modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto="Hola, Bienvenidos a Inteligencia Artificial generativa."

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es", #expresividad o empción
    exaggeration=0.7, #Afecta velocidad y ritmo
    cfg_weight=0.4, #Controla variabilidad
    temperature=0.8, #Penaliza repeticiones extrañas
    repetition_penalty=1.2, #Filtra opciones improbables durante la generación
    min_p=0.05, #Filtra opciones improbables durante la generacion
    top_p=1.0   # Controla el muestreo probabilistico
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)
print("Audio generado.")
