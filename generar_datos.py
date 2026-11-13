import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 1. Definición de las micro-habilidades (Skills) para SAT e IB DP
SKILLS = {
    "SAT": [
        "SAT_Algebra_Linear_Eq",
        "SAT_Passport_Advanced_Math",
        "SAT_Problem_Solving_Data",
        "SAT_Geometry_Trig"
    ],
    "IB": [
        "IB_Math_AA_Integration",
        "IB_Math_AA_Differentiation",
        "IB_Math_AI_Distributions",
        "IB_Math_AI_Hypothesis_Testing"
    ]
}

def generate_synthetic_bkt_data(num_students=150, interactions_per_student=60):
    data = []
    
    # 2. Parámetros simulados realistas: (Prior, Learn Rate, Guess, Slip)
    # Prior: Probabilidad inicial de conocer el tema antes de practicar
    # Learn Rate: Tasa a la que aprenden tras cada intento
    # Guess: Probabilidad de acertar por suerte (ej. opción múltiple)
    # Slip: Probabilidad de equivocarse en un cálculo a pesar de saber la teoría
    skill_params = {
        "SAT_Algebra_Linear_Eq": (0.40, 0.15, 0.25, 0.10),
        "SAT_Passport_Advanced_Math": (0.20, 0.10, 0.20, 0.15),
        "SAT_Problem_Solving_Data": (0.50, 0.20, 0.25, 0.05),
        "SAT_Geometry_Trig": (0.30, 0.12, 0.20, 0.10),
        "IB_Math_AA_Integration": (0.10, 0.08, 0.05, 0.20),   # Guess bajo por ser pregunta abierta
        "IB_Math_AA_Differentiation": (0.20, 0.10, 0.05, 0.15),
        "IB_Math_AI_Distributions": (0.30, 0.15, 0.10, 0.10),
        "IB_Math_AI_Hypothesis_Testing": (0.15, 0.12, 0.10, 0.10)
    }
    
    for student in range(1, num_students + 1):
        # Asignar al estudiante a un programa específico
        program = random.choice(["SAT", "IB"])
        student_skills = SKILLS[program]
        
        # Simular la aptitud general del estudiante (añade varianza a la data)
        student_aptitude = random.uniform(-0.1, 0.2) 
        
        # Inicializar el estado de conocimiento oculto
        knowledge_state = {}
        for skill in student_skills:
            prior = max(0.01, min(0.99, skill_params[skill][0] + student_aptitude))
            knowledge_state[skill] = prior
            
        # Configurar fecha de inicio de clases simulada
        current_time = datetime(2026, 8, 1, 15, 0)
        
        for _ in range(interactions_per_student):
            # El tutor o sistema asigna un ejercicio de un tema aleatorio
            skill = random.choice(student_skills)
            _, learn_rate, guess, slip = skill_params[skill]
            
            # Determinar si en este instante el estudiante "sabe" resolver el tema
            knows_skill = random.random() < knowledge_state[skill]
            
            # Generar el resultado (1 = Correcto, 0 = Incorrecto)
            if knows_skill:
                # Si sabe, solo falla si comete un "Slip" (error tonto)
                correct = 0 if random.random() < slip else 1
            else:
                # Si no sabe, solo acierta por "Guess" (suerte)
                correct = 1 if random.random() < guess else 0
                
            # Actualizar la probabilidad de que haya aprendido el tema para el futuro
            if not knows_skill:
                knowledge_state[skill] = knowledge_state[skill] + ((1 - knowledge_state[skill]) * learn_rate)
                
            # Simular un ID de pregunta específica (ej. Banco de 20 preguntas por tema)
            question_id = f"Q_{skill}_{random.randint(1, 20)}"
            
            data.append({
                "student_id": f"STU_{student:04d}",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "skill_name": skill,
                "question_id": question_id,
                "correct": correct
            })
            
            # Avanzar el tiempo para la siguiente interacción (clases o tareas)
            current_time += timedelta(hours=random.randint(1, 48), minutes=random.randint(0, 59))
            
    # Convertir a DataFrame y ordenar cronológicamente por estudiante
    df = pd.DataFrame(data)
    df = df.sort_values(by=['student_id', 'timestamp']).reset_index(drop=True)
    return df

# Ejecución de la simulación
df_simulated = generate_synthetic_bkt_data(num_students=200, interactions_per_student=50)

# Exportar a CSV para ser consumido por el modelo de Machine Learning
df_simulated.to_csv("synthetic_bkt_dataset.csv", index=False)

# Imprimir una muestra para validar la estructura
print("Dataset Generado. Total de registros:", len(df_simulated))
print(df_simulated.head(10))