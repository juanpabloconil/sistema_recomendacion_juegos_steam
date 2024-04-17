from fastapi import FastAPI
import pandas as pd 
from fastapi.responses import JSONResponse
import pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Commandos
#Ruta de acceso a la API local:
#http://127.0.0.1:8000/
#Ruta para acceder a interfaz de usuario:
#http://127.0.0.1:8000/docs#/
#Código para activar la API:
#API_env\Scripts\activate
#Código para cargar API:
#uvicorn main:app --reload
#deactivate 

#Se instancia la aplicación
app = FastAPI()


# http://127.0.0.1:8000/docs#/ #-> ruta para API local
#codigo para inicializar API --> uvicorn main:app --reload

#Funcion de bienvenida:

#Funcion para la presentacion
@app.get('/')
def mensaje():
    return "Bienved@!\ Esta es una API para realizar consultas sobre juegos de STEAM."

#------------------------------------------------------------------------------------------------------
#                                            Consulta 01
#------------------------------------------------------------------------------------------------------

#Esta consulta devuelve una tabla que indica la cantidad de items y porcentaje de contenido Free por año según la empresa desarrolladora.
#http://127.0.0.1:8000/developer/?desarrollador=Poppermost%20Productions

@app.get('/developer/')
def get_developer_stats(desarrollador: str):

    df = pd.read_parquet('./Consulta_Data/df_games_clean.parquet')


    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')


    developer_df = df[df['developer'] == desarrollador]


    items_por_año = developer_df.groupby(df['release_date'].dt.year).size().reset_index(name='Cantidad de Items')


    free_por_año = developer_df[developer_df['price'] == 'Free'].groupby(df['release_date'].dt.year).size().reset_index(name='Contenido Free')


    result_df = items_por_año.merge(free_por_año, on='release_date', how='left')
    result_df['Contenido Free'] = (result_df['Contenido Free'] / result_df['Cantidad de Items'] * 100).fillna(0).astype(int).astype(str) + '%'


    result_df.rename(columns={'release_date': 'Año'}, inplace=True)

    return result_df.to_dict(orient='records')


#---------------------------------------------------------------------------------------------------------------
#                                              Consulta 02
#---------------------------------------------------------------------------------------------------------------
#Esta consulta devuelve un diccionario con cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
#Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
#http://127.0.0.1:8000/userdata/?User_id=76561198070234207

@app.get("/userdata/")
def get_user_data(User_id: str):
    df_games = pd.read_parquet('./Consulta_Data/df_games_clean.parquet')
    df_items = pd.read_parquet('./Consulta_Data/df_items_clean.parquet')
    df_reviews = pd.read_parquet('./Consulta_Data/df_reviews_clean.parquet')

    games_copy = df_games.copy()
    items_copy = df_items.copy()
    reviews_copy = df_reviews.copy()

    # Convertir la columna 'user_id' a str
    items_copy['user_id'] = items_copy['user_id'].astype(str)
    reviews_copy['user_id'] = reviews_copy['user_id'].astype(str)

    # Filtrar df_Items por el user_id dado
    user_items = items_copy[items_copy['user_id'] == str(User_id)]

    # Calcular la cantidad de dinero gastado por el usuario
    # Convertir la columna 'price' a numérica
    games_copy['price'] = pd.to_numeric(games_copy['price'], errors='coerce')
    money_spent = user_items.merge(games_copy[['id', 'price']], left_on='item_id', right_on='id')['price'].sum()

    # Calcular la cantidad total de items del usuario
    total_items = user_items['items_count'].sum()

    # Filtrar df_reviews por el user_id dado
    user_reviews = reviews_copy[reviews_copy['user_id'] == str(User_id)]
#Calcular el porcentaje de recomendación promedio del usuario
    if user_reviews.shape[0] > 0:
        recommendation_percentage = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
    else:
        recommendation_percentage = 0

    # Convertir valores de numpy.int64 a tipos de datos estándar
    money_spent = float(money_spent) if not pd.isnull(money_spent) else 0.0  # Convertir a float, manejar NaN si es necesario
    recommendation_percentage = float(recommendation_percentage) if not pd.isnull(recommendation_percentage) else 0.0  # Convertir a float, manejar NaN si es necesario

#Crear el diccionario de resultados
    result = {
        "Usuario": str(User_id),
        "Dinero gastado": f"{money_spent:.2f} USD",  # Ajustamos el formato para mostrar dos decimales
        "% de recomendación": f"{recommendation_percentage:.2f}%",
        "Cantidad de items": int(total_items)
    }

    return JSONResponse(content=result)

#------------------------------------------------------------------------------------------------------------------------
#                                                      Consulta 03:
#------------------------------------------------------------------------------------------------------------------------
#Esta consulta devuelve devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
#http://127.0.0.1:8000/user-for-genre/?genero=Action

@app.get("/user-for-genre/")
def user_for_genre(genero: str):
    df_games = pd.read_parquet('./Consulta_Data/df_games_clean.parquet')
    df_items = pd.read_parquet('./Consulta_Data/df_items_clean.parquet')

    df_games_copy = df_games.copy()
    df_items_copy = df_items.copy()

    # Tu código existente de la función UserForGenre
    if 'genres' not in df_games_copy.columns:
        raise ValueError("El DataFrame df_games_copy no tiene una columna llamada 'genre'.")

    df_games_copy['release_date'] = pd.to_datetime(df_games_copy['release_date'], errors='coerce')

    juegos_genero = df_games_copy[df_games_copy['genres'] == genero]
    juegos_usuario = juegos_genero.merge(df_items_copy, left_on='id', right_on='item_id')
    horas_por_usuario = juegos_usuario.groupby('user_id')['playtime_forever'].sum().reset_index()
    usuario_max_horas = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax()]['user_id']
    horas_por_año = juegos_usuario.groupby(juegos_usuario['release_date'].dt.year)['playtime_forever'].sum().reset_index()
    horas_por_año.rename(columns={'playtime_forever': 'Horas'}, inplace=True)
    horas_por_año = horas_por_año.to_dict('records')

    result = {
        "Usuario con más horas jugadas para {}: ".format(genero): usuario_max_horas,
        "Horas jugadas": horas_por_año
    }

    return result 
#----------------------------------------------------------------------------------------------------------------------------
#                                                    CONSULTA 05
#----------------------------------------------------------------------------------------------------------------------------
#Esta consulta devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de rese;as de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo, en caso de no estar categorizados o encontrarse arroja el mensaje "No se encontró información sobre el desarrollador '...'".
#http://127.0.0.1:8000/developer-reviews-analysis/?desarrollador=Kotoshiro
@app.get("/developer-reviews-analysis/")


def developer_reviews_analysis(desarrollador: str):

    games = pd.read_parquet('./Consulta_data/user_games_clean.parquet')
    sentiment = pd.read_parquet('./Consulta_Data/Sentiment_Analysis_parquet')
    games_copy = games.copy()
    sentiment_copy = sentiment.copy()

#Combinar conjuntos de datos en las columnas apropiadas ('item_id' en reviews y 'id' en games)
    merged_data = pd.merge(sentiment_copy, games_copy, left_on='item_id', right_on='id')

#Filtrar filas donde el puntaje de sentimiento es positivo (2) o negativo (0)
    filtered_data = merged_data[merged_data['sentiment_analysis'] != 1]  # Excluir sentimiento neutral

#Agrupar por desarrolladora y puntaje de sentimiento, contar la cantidad de resenas
    grouped_data = filtered_data.groupby(['developer', 'sentiment_analysis']).size().unstack(fill_value=0)

#Verificar si la desarrolladora está presente en el DataFrame
    if desarrollador in grouped_data.index:
        # Extraer cantidad de resenas positivas y negativas para la desarrolladora especificada
        developer_reviews = grouped_data.loc[desarrollador]

        # Convertir cantidades a formato de lista con claves especificadas
        developer_reviews_list = [
            {"Negativas": developer_reviews.get(0, 0)},
            {"Positivas": developer_reviews.get(2, 0)}
        ]

        return {desarrollador: developer_reviews_list}
    else:
        return f"No se encontró información sobre la desarrolladora {desarrollador}"




# ------------------------------------------------------------------------------------
#                                       CONSULTA 4 
#-------------------------------------------------------------------------------------
df_merged = pd.read_parquet('./Consulta_Data/df_merged.parquet')

@app.get('/best_developer_year/')

def best_developer_year(año:int):
    """
    Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
    Se tienen en cuenta recommend = True + sentiment_analysis con mayor puntuación.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        año (int): Año para el que se desea obtener el top 3 de desarrolladores.

    Returns:
        dict: Diccionario ordenado con los top 3 desarrolladores para el año dado, en formato {1er puesto: primer juego, 2do puesto: segundo juego, 3er puesto: tercer juego}.
    """
    # Filtrar el DataFrame para el año dado
    df_año = df_merged[df_merged['year_posted'] == año]

    # Filtrar por recomendaciones verdaderas y sentimiento de análisis más alto
    df_filtrado = df_año[df_año['recommend'] & (df_año['sentiment_analysis'] == df_año['sentiment_analysis'].max())]

    # Agrupar por desarrollador y contar las recomendaciones
    top_developers = df_filtrado.groupby('developer')['recommend'].sum()

    # Ordenar los desarrolladores por número de recomendaciones en orden descendente
    top_developers = top_developers.sort_values(ascending=False)

    # Tomar los top 3 desarrolladores
    top_3_developers = top_developers.head(3)

    # Crear el diccionario ordenado con los top 3 desarrolladores
    top_developers_dict = {}
    for i, (developer, recomendaciones) in enumerate(top_3_developers.items(), 1):
        puesto = f"{i}er puesto"
        top_developers_dict[puesto] = developer

    return top_developers_dict

#--------------------------------------------------------------------------------------------------------------
#                                           MODELO SISTEMA DE RECOMENDACION
#--------------------------------------------------------------------------------------------------------------
#774276 ---> usuario de prueba 

df_genres = pd.read_parquet('./Consulta_Data/df_dummies.parquet')
df_games = pd.read_parquet('./Consulta_Data/df_games_clean.parquet')

df_merged = df_games.merge(df_genres, on='id', how='left')
features = ['release_date'] + list(df_genres.columns[1:])
scaler = StandardScaler()
df_merged['release_date'] = scaler.fit_transform(df_merged[['release_date']])
df_final = df_merged[['id'] + features]
df_final= df_final.merge(df_games[['id', 'app_name']], on='id', how='left')

df_sampled = df_final.sample(frac=0.2, random_state=42)
similarity_matrix = cosine_similarity(df_sampled[features].fillna(0))
similarity_matrix = np.nan_to_num(similarity_matrix)

#Definir la variable global top_n
top_n = 5

@app.get("/recomendacion-juegos/")
def recomendar_juegos(game_id: str):
    ids_juegos_muestreados = df_sampled['id'].unique()

#Verificar si el ID del juego está en los juegos muestreados
    if game_id not in ids_juegos_muestreados:
        return f"No se encontraron recomendaciones: {game_id} no está en los datos muestreados."

#Obtener el índice del juego en los datos muestreados
    indice_juego = df_sampled.index[df_sampled['id'] == game_id].tolist()
#Verificar si se encontró el juego en los datos muestreados
    if not indice_juego:
        return f"No se encontraron recomendaciones: {game_id} no está en los datos muestreados."

    indice_juego = indice_juego[0]

    # Calcular los puntajes de similitud entre juegos
    puntajes_similitud = list(enumerate(similarity_matrix[indice_juego]))
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los juegos similares
    indices_juegos_similares = [i for i, puntaje in puntajes_similitud[1:top_n+1]]

#Obtener los nombres de los juegos similares
    nombres_juegos_similares = df_sampled['app_name'].iloc[indices_juegos_similares].tolist()

    # Mensaje de recomendación
    mensaje_recomendacion = f"Juegos recomendados basados en el ID del juego {game_id} - {df_sampled['app_name'].iloc[indice_juego]}:"

    return [mensaje_recomendacion] + nombres_juegos_similares 




