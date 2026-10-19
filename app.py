import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(page_title="EdTech ML: BKT Simulator", layout="wide")

st.title("Adaptive Learning Engine (Knowledge Tracing)")
st.markdown("""
**Proyecto de Portfolio de Willy Olaya | Data Scientist & EdTech Specialist**
Esta aplicación simula un modelo de *Bayesian Knowledge Tracing* (BKT). Evalúa en tiempo real 
la probabilidad de que un estudiante domine una habilidad matemática basándose en sus respuestas.
""")

# 2. Diccionario de parámetros aprendidos (Simulando lo que nos dio pyBKT)
# Formato: 'Skill': (Prior, Learn, Guess, Slip)
SKILL_PARAMS = {
    "SAT_Algebra_Linear_Eq": (0.40, 0.15, 0.25, 0.10),
    "SAT_Geometry_Trig": (0.30, 0.12, 0.20, 0.10),
    "IB_Math_AA_Integration": (0.10, 0.08, 0.05, 0.20)
}

# 3. Inicialización del estado de la sesión (Session State)
if 'skill' not in st.session_state:
    st.session_state.skill = "SAT_Algebra_Linear_Eq"
    st.session_state.current_skill_selection = st.session_state.skill
    st.session_state.prob_mastery = SKILL_PARAMS[st.session_state.skill][0]
    st.session_state.history = [st.session_state.prob_mastery]
    st.session_state.attempts = 0

# Función para reiniciar la simulación si se cambia de tema
def reset_state(new_skill):
    st.session_state.skill = new_skill
    st.session_state.prob_mastery = SKILL_PARAMS[new_skill][0]
    st.session_state.history = [st.session_state.prob_mastery]
    st.session_state.attempts = 0

def reset_selected_skill():
    reset_state(st.session_state.current_skill_selection)

# 4. Lógica de Actualización Bayesiana (El núcleo del modelo)
def update_knowledge(correct):
    prior = st.session_state.prob_mastery
    _, learn, guess, slip = SKILL_PARAMS[st.session_state.skill]
    
    # Teorema de Bayes para calcular la probabilidad dado el resultado de la pregunta
    if correct:
        # P(L | Correct) = P(Correct | L) * P(L) / P(Correct)
        prob_obs = (prior * (1 - slip)) + ((1 - prior) * guess)
        post_prob = (prior * (1 - slip)) / prob_obs
    else:
        # P(L | Incorrect) = P(Incorrect | L) * P(L) / P(Incorrect)
        prob_obs = (prior * slip) + ((1 - prior) * (1 - guess))
        post_prob = (prior * slip) / prob_obs
        
    # Añadir la probabilidad de aprender el tema durante la resolución
    new_mastery = post_prob + ((1 - post_prob) * learn)
    
    # Actualizar variables globales
    st.session_state.prob_mastery = new_mastery
    st.session_state.history.append(new_mastery)
    st.session_state.attempts += 1

# 5. Interfaz de Usuario (UI)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuración")
    
    # Selección de Habilidad
    selected_skill = st.selectbox(
        "Selecciona el tema a evaluar:",
        list(SKILL_PARAMS.keys()),
        index=list(SKILL_PARAMS.keys()).index(st.session_state.skill),
        key="current_skill_selection",
        on_change=reset_selected_skill,
    )

    # Mostrar parámetros de la habilidad
    prior, learn, guess, slip = SKILL_PARAMS[selected_skill]
    st.write(f"- **Prior:** {prior:.2f} (Base line)")
    st.write(f"- **Learn:** {learn:.2f} (Tasa de aprendizaje)")
    st.write(f"- **Guess:** {guess:.2f} (Prob. de acertar por suerte)")
    st.write(f"- **Slip:** {slip:.2f} (Prob. de equivocarse por descuido)")
    
    st.divider()
    st.header("Interacción")
    st.write(f"Intento actual: #{st.session_state.attempts + 1}")
    
    # Botones de simulación
    if st.button("El estudiante respondió CORRECTAMENTE", use_container_width=True):
        update_knowledge(correct=True)
    if st.button("El estudiante respondió INCORRECTAMENTE", use_container_width=True):
        update_knowledge(correct=False)
        
    if st.button("Reiniciar Simulación", type="secondary"):
        reset_state(selected_skill)
        st.rerun()

with col2:
    st.header("Nivel de Dominio (Knowledge State)")
    
    # Métrica principal grande
    current_prob = st.session_state.prob_mastery
    delta_value = None
    if len(st.session_state.history) > 1:
        delta_value = f"{(current_prob - st.session_state.history[-2]):.1%}"

    st.metric(
        label="Probabilidad de que domine el tema",
        value=f"{current_prob:.1%}",
        delta=delta_value,
    )
    
    # Gráfico interactivo con Plotly
    df_chart = pd.DataFrame({
        "Intento": range(len(st.session_state.history)),
        "Probabilidad": st.session_state.history
    })
    
    fig = px.line(
        df_chart, 
        x="Intento", 
        y="Probabilidad", 
        markers=True,
        title=f"Evolución del Aprendizaje: {selected_skill}",
        range_y=[0, 1]
    )
    
    # Línea de umbral (Mastery Threshold)
    fig.add_hline(y=0.85, line_dash="dash", line_color="green", annotation_text="Dominio Alcanzado (85%)")
    fig.update_traces(line_color="#1f77b4", marker=dict(size=10))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Lógica de Recomendación Automática
    if current_prob >= 0.85:
        st.success("**Sistema de Recomendación:** El estudiante ha dominado el tema. Avanzar al siguiente módulo de la currícula.")
    elif current_prob < 0.20 and st.session_state.attempts > 3:
        st.error("**Sistema de Recomendación:** El estudiante está frustrado. Detener práctica y recomendar un video teórico o repaso guiado.")
    else:
        st.info("**Sistema de Recomendación:** Continuar asignando ejercicios de este mismo tema.")