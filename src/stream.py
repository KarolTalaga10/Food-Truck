import streamlit as st
from sa_implementation import Event, Solution, generate_init_route
# from PIL import Image
st.title('Find best path for Food - Truck!')
# img = Image.open('../images/ft.png')
# st.image(img, width=None)


def print_solution(solution):
    st.write('Your initial solution is: ')
    for event in solution:
        st.write('* {0}, {1}, {2}'.format(event.name, event.time_start, event.time_stay))


st.sidebar.subheader('Set parameters')
st.sidebar.selectbox('Alpha', (0.999, 0.99, 0.9))
T_init = st.sidebar.slider('Initial temperature', 0, 5000, 0, step=10)
T_stop = st.sidebar.slider("Stop temperature", 0, 500, 0, step=1)
chain_len = st.sidebar.slider('Length of Markov chain', 0, 100, 0, step=5)
num_days = st.sidebar.slider('Number of cycle duration', 1, 31, 1, step=1)
num_events = st.sidebar.slider('Number of events hosts', 1, 8, 1, step=1)
init_select = st.sidebar.selectbox(
    "How would you like to create initial solution?",
    ("Randomly", "Manually")
)

if init_select == 'Randomly':
    init_route = generate_init_route(num_events, num_days)
else:
    pass

search_btn = st.sidebar.button("Search!")
if search_btn:
    print_solution(init_route)
else:
    pass
