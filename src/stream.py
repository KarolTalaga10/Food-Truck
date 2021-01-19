import streamlit as st
from sa_implementation import Event, SolutionApp, SolMethod, generate_init_route
from data_prep import NAMES
from PIL import Image
import base64

st.title('Find best path for Food - Truck!')
file_ = open("../images/loading.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
img = Image.open('../images/ft2.jpg')
start = st.image(img, width=500)

st.sidebar.subheader('Set parameters')
params = {'alpha': st.sidebar.selectbox('Alpha', (0.999, 0.99, 0.9))}
T_init = st.sidebar.slider('Initial temperature', 0, 5000, 0, step=10)
T_stop = st.sidebar.slider("Stop temperature", 0, 100, 0, step=1)
chain_len = st.sidebar.slider('Length of Markov chain', 0, 100, 0, step=5)
init_select = st.sidebar.selectbox(
    "How would you like to create initial solution?",
    ('-', "Randomly", "Manually")
)

if init_select == 'Randomly':
    num_events = st.sidebar.slider('Number of events hosts', 1, 8, 1, step=1)
    num_days = st.sidebar.slider('Number of cycle duration', num_events * 2, 31, num_events * 2, step=1)
    init_route = generate_init_route(num_events, num_days)

elif init_select == 'Manually':
    num_events = st.sidebar.slider('Number of events hosts', 1, 8, 1, step=1)
    sum_days = 0
    init_route = []
    day_of_selling = []
    day_start = 1
    for i in range(num_events):
        names = {'events': st.selectbox('{0} event name '.format(i + 1), NAMES)}
        day_of_selling.append(
            st.slider("How many days do you want to be in {0} event?".format(i + 1), 1, 31, 1, step=1))
        sum_days += day_of_selling[i]
        init_route.append(Event(names['events'], day_start, day_of_selling[i]))
        day_start += 1 + day_of_selling[i]

else:
    pass

show_agenda = st.sidebar.checkbox("Show agenda")
search = st.sidebar.button("Search!")

if show_agenda:
    start.empty()
    img = Image.open('../images/agenda.png')
    st.subheader('Agenda')
    st.image(img, width=600)

if search:
    st.subheader('Your initial solution is: ')
    SolutionApp.print_solution(init_route)
    a = st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    final_solution = SolutionApp(init_route, T_init, params['alpha'], T_stop, SolMethod.GEO, chain_len, init_route, 0)
    final_solution.sym_ann_algorithm()
    a.empty()
else:
    pass
