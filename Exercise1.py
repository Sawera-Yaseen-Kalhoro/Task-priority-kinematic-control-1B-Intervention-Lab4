from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)                                 # displacement along Z-axis
theta = np.array([0.2,0.5,0.1]).reshape(3,1)    # rotation around Z-axis
alpha = np.zeros(3)                             # rotation around X-axis
a = np.array([0.75,0.5,0.3])                    # displacement along X-axis
revolute = [True,True,True]                     # flags specifying the type of joints
robot = Manipulator(d, theta.flatten(), a, alpha, revolute) # Manipulator object

# Task hierarchy definition
tasks = [ 
            # Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1)),
            # JointPosition("Joint 1 Position",np.array([[0.0]])),
            # Orientation2D("End-effector orientation", np.array([[0.0]]))
            Configuration2D("End-effector configuration", np.array([1.0, 0.0, 0.0]).reshape(3, 1))
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# initialize a memory to store the error evolution
error_evol = []
if isinstance(tasks[0],Configuration2D):
    error_evol = [[],[]] # initialize 2 lists to store position and orientation
else:
    for i in range(len(tasks)):
        error_evol.append([])

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # initialize random desired EE position
    desired_EE_random = np.random.uniform(-1.2, 1.2, (2, 1))
    if isinstance(tasks[0],Position2D):
        tasks[0].setDesired(desired_EE_random)
    elif isinstance(tasks[0],Configuration2D):
        tasks[0].setDesired(np.vstack((desired_EE_random,[0.0]))) # fix the desired orientation to 0

    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global error_evol

    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    Pi_1 = np.eye(robot.getDOF())
    # Initialize output vector (joint velocity)
    dqi_1 = np.zeros((robot.getDOF(),1))
    # Loop over tasks
    for i in range(len(tasks)):
        task = tasks[i]
        # Update task state
        task.update(robot)
        Ji = task.getJacobian()
        erri = task.getError()

        # store the error evolution
        if isinstance(task,Configuration2D):
            error_evol[0].append(np.linalg.norm(erri[:2]))
            error_evol[1].append(erri[2,0])
        elif erri.shape[0] > 1:
            error_evol[i].append(np.linalg.norm(erri))
        else:
            error_evol[i].append(erri[0,0])

        # Compute augmented Jacobian
        Ji_bar = Ji @ Pi_1
        # Compute task velocity
        dqi = DLS(Ji_bar,0.1) @ (erri - Ji@dqi_1)
        # Accumulate velocity
        dq = dqi_1 + dqi
        # Update null-space projector
        Pi = Pi_1 - np.linalg.pinv(Ji_bar)@Ji_bar

        # Store the current P and dq for the next iteration
        Pi_1 = Pi
        dqi_1 = dqi
    ###

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Function to generate the second plot
def error_plot(tasks, error_evol):
    # define simulation time (after repeats)
    tfinal = len(error_evol[0])*dt
    tt = np.arange(0,tfinal,dt)

    # Create a second plot for the joint positions
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, xlim=(0, tfinal))
    ax2.set_title('Task Priority - Error Evolution')
    ax2.set_xlabel('Time[s]')
    ax2.set_ylabel('Error')
    # ax2.set_aspect('equal')
    ax2.grid()

    labels = [] # list of labels for legend

    for i in range(len(error_evol)):
        ax2.plot(tt, error_evol[i])

        # generate label for legend
        if isinstance(tasks[0],Configuration2D):
            labels = ["Error_1 (End-effector position)","Error_2 (End-effector orientation)"]
        else:
            labels.append(f"Error_{i+1} ({tasks[i].name})")

    ax2.legend(labels)
    plt.show()


# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Show the plot of error evolution
error_plot(tasks,error_evol)