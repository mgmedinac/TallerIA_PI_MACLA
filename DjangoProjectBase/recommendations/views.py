from django.shortcuts import render
from dotenv import load_dotenv
import os
import json
import numpy as np
from openai import OpenAI
from movie.models import Movie  # Asegúrate de que este sea el modelo correcto

# Carga las API keys y otros valores de entorno
_ = load_dotenv('api_keys_1.env')
client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Ruta base de tu proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Actualiza la ruta del archivo JSON a la correcta
json_file_path = os.path.join('C:\\Users\\maria\\Desktop\\p1_new_project\\TallerIA_PI_MACLA', 'movie_descriptions_embeddings.json')

# Abre el archivo JSON
with open(json_file_path, 'r') as file:
    movies = json.load(file)


# Función para obtener el embedding de un texto utilizando OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Función para calcular la similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Vista de Django para recomendar películas
def recommend(request):
    recommended_movies = []  # Lista de películas recomendadas
    prompt = ""  # Inicializa el prompt

    if request.method == 'POST':
        prompt = request.POST.get('prompt')  # Obtén el prompt ingresado por el usuario
        prompt_embedding = get_embedding(prompt)  # Genera el embedding del prompt

        # Calcula la similitud de coseno entre el embedding del prompt y los embeddings de las películas
        similarities = []
        for movie in movies:
            similarity = cosine_similarity(prompt_embedding, movie['embedding'])
            similarities.append((similarity, movie))  # Almacena la película completa

        # Ordena las películas por similitud (de mayor a menor)
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Opcional: Limita el número de recomendaciones (puedes ajustar esto)
        top_recommendations = similarities[:10]  # Obtén las 5 mejores

        # Añade las películas recomendadas a la lista
        for _, movie in top_recommendations:
            # Aquí asumimos que el modelo Movie tiene el mismo título que en el JSON
            db_movie = Movie.objects.filter(title__icontains=movie['title']).first()  # Obtiene la primera coincidencia
            if db_movie:
                recommended_movies.append(db_movie)

    # Renderiza la plantilla con las películas recomendadas
    return render(request, 'recommend.html', {'Movies': recommended_movies, 'searchTerm': prompt})
