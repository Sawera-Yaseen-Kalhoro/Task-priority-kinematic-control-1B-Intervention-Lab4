from lab2_robotics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''

    # 1. Initialize J and O.
    J = np.zeros((6, len(T) - 1))  # -1 because T includes the base transformation, which adds 1 more matrix
    O = T[-1][:3, -1].reshape((3, 1))  # from base to n

    # 2. For each joint of the robot
    for i in range(len(T) - 1):
        if i < link:  #(For joints before the specified link compute jacobian because they contribute to end-effector motion)
            # a. Extract z and o.
            Ti = T[i]
            z = Ti[:3, 2].reshape((3, 1))
            o = Ti[:3, -1].reshape((3, 1))  # from base to i-1

            # b. Check joint type.
            rhoi = int(revolute[i])

            # c. Modify corresponding column of J.
            Ji = np.block([
                [np.array([[np.cross(rhoi * z, (O - o), axis=0) + (1 - rhoi) * z]])],
                [np.array([[rhoi * z]])]
            ])
            J[:, i] = Ji.reshape((6,))

        else: # Joints beyond the specific linnk do not contribute to robot's motion hence corresponding jacobian columns are 0
            J[:, i] = np.zeros(6)

    return J


'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def _init_(self, name, desired):
        super()._init_(name, desired)
        self.J = np.zeros(0) # Initialize with proper dimensions
        self.err = np.zeros(len(desired,1)) # Initialize with proper dimensions
        self.err_evolution = []  # Initialize error evolution
        self.label = name  # Set label for the task
        
    def update(self, robot):

        J_ee = robot.getEEJacobian() #Get end-effector position
        self.J = J_ee[:2,:] # Update task Jacobian
        ee_position = robot.getEETransform()[:2,-1].reshape((2,1)) # get the end-effector position
        self.err =  self.sigma_d - ee_position # Update task error
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def _init_(self, name, desired):
        super()._init_(name, desired)
        self.J = np.zeros(0) # Initialize with proper dimensions
        self.err = np.zeros(len(desired,1))  # Initialize with proper dimensions        
        
    def update(self, robot):
        #Get the end-effector Jacobian
        J_ee = robot.getEEJacobian()
        self.J = np.array([J_ee[-1,:]]) # last row of of the Jacobian, corresponds to rotation around z
        #Get the end-effector orientation
        ee_orientation = robot.getEETransform()[:2, :2]
        # Update the task error as the deifference between the desired orientation and the end-effector orientation
        self.err = self.sigma_d - np.arctan2(ee_orientation[1,0], ee_orientation[0,0])
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((3, 3))  # Initialize with proper dimensions
        self.err = np.zeros((3,1))  # Initialize with proper dimensions
        
    def update(self, robot):
            J_pos = robot.getEEJacobian()[:2, :]  # Get position Jacobian (translational component)
            J_ori = np.array([robot.getEEJacobian()[-1,:]])  # Get orientation Jacobian (rotational component)
            self.J = np.vstack((J_pos, J_ori))  # Update the task Jacobian (Stack position and orientation Jacobians vertically)
            
            # Calculate task error
            ee_transform = robot.getEETransform()  # Get end-effector transformation
            ee_position = ee_transform[:2, -1].reshape((2, 1))  # Extract position component
            ee_orientation = np.arctan2(ee_transform[1, 0], ee_transform[0, 0])  # Extract orientation component
            self.err = np.vstack((self.sigma_d[:2] - ee_position, self.sigma_d[2] - ee_orientation))  # Stack errors vertically

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((1,3)) # Initialize with proper dimensions
        self.err = np.zeros((1,1)) # Initialize with proper dimensions
        
    def update(self, robot):
        self.J = np.array([[1,0,0]]) # Update task Jacobian
        self.err = self.getDesired() - robot.q[0] # Update task error
